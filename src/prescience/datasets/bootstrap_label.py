"""Bootstrap labeling workflow with model-assisted approvals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from prescience.datasets.manifest import (
    LabelManifest,
    load_or_create_manifest,
    save_manifest,
    should_process,
    upsert_record,
)
from prescience.datasets.yolo import (
    TrainConfig,
    build_yolo_dataset,
    collect_labeled_images,
    list_images,
    train_yolo_model,
)


@dataclass(frozen=True)
class BootstrapParams:
    sku: str
    frames_dir: Path
    labels_dir: Path
    sections: int = 6
    manual_per_section: int = 2
    approve_per_section: int = 5
    class_id: int = 0
    base_model: str = "yolov8n.pt"
    imgsz: int = 960
    conf_propose: float = 0.10
    epochs_stage1: int = 30
    retrain_after_approve: bool = True
    epochs_stage2: int = 60
    pick_largest: bool = True
    overwrite: bool = False
    allow_negatives: bool = True


def _xyxy_to_yolo_norm(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    box_w = x2 - x1
    box_h = y2 - y1
    xc = x1 + box_w / 2.0
    yc = y1 + box_h / 2.0

    return (xc / width, yc / height, box_w / width, box_h / height)


def _label_path_for_image(labels_dir: Path, image_path: Path) -> Path:
    return labels_dir / f"{image_path.stem}.txt"


def _write_positive_label(label_path: Path, class_id: int, yolo_box: tuple[float, float, float, float]) -> None:
    xc, yc, bw, bh = yolo_box
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")


def _write_negative_label(label_path: Path) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("", encoding="utf-8")


class _ManualBoxDrawer:
    def __init__(self, window_name: str):
        self.window_name = window_name
        self.dragging = False
        self.box_ready = False
        self.start = (0, 0)
        self.end = (0, 0)
        cv2.setMouseCallback(window_name, self._on_mouse)

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.box_ready = False
            self.start = (x, y)
            self.end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.end = (x, y)
            self.box_ready = True

    def current_box(self) -> tuple[int, int, int, int] | None:
        if not self.box_ready:
            return None
        return (self.start[0], self.start[1], self.end[0], self.end[1])


def _label_one_image_manual(
    img_path: Path,
    label_path: Path,
    class_id: int,
    allow_negatives: bool,
) -> str:
    """Manual labeling UI. Returns positive, negative, skipped."""
    image = cv2.imread(str(img_path))
    if image is None:
        return "skipped"

    height, width = image.shape[:2]
    window = "Manual Label (drag box, Enter=save, 0=negative, Esc=skip)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    drawer = _ManualBoxDrawer(window)

    while True:
        disp = image.copy()
        cv2.putText(
            disp,
            "Draw box | Enter=save | 0=negative | Esc=skip",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        if drawer.dragging or drawer.current_box() is not None:
            x1, y1 = drawer.start
            x2, y2 = drawer.end
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 200, 255), 2)

        cv2.imshow(window, disp)
        key = cv2.waitKey(20) & 0xFF

        if key == 13:
            box = drawer.current_box()
            if box is None:
                continue
            yolo_box = _xyxy_to_yolo_norm(*box, width, height)
            _write_positive_label(label_path, class_id, yolo_box)
            cv2.destroyWindow(window)
            return "positive"

        if key == ord("0") and allow_negatives:
            _write_negative_label(label_path)
            cv2.destroyWindow(window)
            return "negative"

        if key == 27:
            cv2.destroyWindow(window)
            return "skipped"


def _choose_best_box_xyxy(
    boxes_xyxy: np.ndarray,
    confidences: np.ndarray,
    pick_largest: bool,
) -> tuple[int, int, int, int] | None:
    if boxes_xyxy.size == 0:
        return None

    if pick_largest:
        areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
        idx = int(np.argmax(areas))
    else:
        idx = int(np.argmax(confidences))

    x1, y1, x2, y2 = boxes_xyxy[idx]
    return (int(x1), int(y1), int(x2), int(y2))


def _approve_one_image_with_model(
    img_path: Path,
    label_path: Path,
    model: YOLO,
    class_id: int,
    conf: float,
    imgsz: int,
    pick_largest: bool,
    allow_negatives: bool,
) -> str:
    """Approval UI. Returns positive, negative, skipped, quit."""
    image = cv2.imread(str(img_path))
    if image is None:
        return "skipped"

    height, width = image.shape[:2]

    results = model.predict(source=image, conf=conf, imgsz=imgsz, verbose=False)
    result = results[0]

    best: tuple[int, int, int, int] | None = None
    if result.boxes is not None and len(result.boxes) > 0:
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        best = _choose_best_box_xyxy(boxes_xyxy, confs, pick_largest=pick_largest)

    window = "Approve (y=accept, n=manual, x=negative, s=skip, q=quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while True:
        disp = image.copy()

        if best is not None:
            x1, y1, x2, y2 = best
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(disp, "PROPOSED", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(disp, "No proposal. Press n for manual.", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(
            disp,
            "y=accept n=manual x=negative s=skip q=quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        cv2.imshow(window, disp)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            cv2.destroyWindow(window)
            return "quit"

        if key == ord("s"):
            cv2.destroyWindow(window)
            return "skipped"

        if key == ord("x") and allow_negatives:
            _write_negative_label(label_path)
            cv2.destroyWindow(window)
            return "negative"

        if key == ord("y") and best is not None:
            yolo_box = _xyxy_to_yolo_norm(*best, width, height)
            _write_positive_label(label_path, class_id, yolo_box)
            cv2.destroyWindow(window)
            return "positive"

        if key == ord("n"):
            cv2.destroyWindow(window)
            return _label_one_image_manual(
                img_path=img_path,
                label_path=label_path,
                class_id=class_id,
                allow_negatives=allow_negatives,
            )


def _split_into_sections(paths: list[Path], sections: int) -> list[list[Path]]:
    if sections <= 0:
        raise ValueError("sections must be > 0")
    n = len(paths)
    out: list[list[Path]] = []
    start = 0
    for i in range(sections):
        size = n // sections + (1 if i < (n % sections) else 0)
        out.append(paths[start : start + size])
        start += size
    return out


def _pick_evenly(section_paths: list[Path], k: int, selectable: set[str]) -> list[Path]:
    available = [path for path in section_paths if path.name in selectable]
    if not available:
        return []
    if k >= len(available):
        return available
    idxs = np.linspace(0, len(available) - 1, num=k, dtype=int).tolist()
    return [available[i] for i in idxs]


def _sync_manifest_from_existing_labels(manifest: LabelManifest, labels_dir: Path, images: list[Path]) -> None:
    for image in images:
        label_path = _label_path_for_image(labels_dir, image)
        if not label_path.exists() or image.name in manifest.records:
            continue
        text = label_path.read_text(encoding="utf-8").strip()
        status = "negative" if not text else "positive"
        upsert_record(manifest, image.name, status, str(label_path))


def run_bootstrap_labeling(params: BootstrapParams) -> None:
    images = list_images(params.frames_dir)
    if not images:
        raise RuntimeError(f"No images found in {params.frames_dir}")

    params.labels_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = params.labels_dir.parent / "manifest.json"
    manifest = load_or_create_manifest(
        manifest_path=manifest_path,
        sku=params.sku,
        frames_dir=params.frames_dir,
        labels_dir=params.labels_dir,
    )
    _sync_manifest_from_existing_labels(manifest, params.labels_dir, images)
    save_manifest(manifest_path, manifest)

    sections = _split_into_sections(images, params.sections)

    selectable = {
        image.name
        for image in images
        if should_process(image.name, params.overwrite, set(manifest.records.keys()))
    }

    stage1_candidates: list[Path] = []
    for section in sections:
        stage1_candidates.extend(_pick_evenly(section, params.manual_per_section, selectable))

    for image in stage1_candidates:
        label_path = _label_path_for_image(params.labels_dir, image)
        result = _label_one_image_manual(
            img_path=image,
            label_path=label_path,
            class_id=params.class_id,
            allow_negatives=params.allow_negatives,
        )
        if result in {"positive", "negative", "skipped"}:
            upsert_record(manifest, image.name, result, str(label_path))
            save_manifest(manifest_path, manifest)

    labeled_stage1 = collect_labeled_images(params.frames_dir, params.labels_dir)
    if not labeled_stage1:
        raise RuntimeError("No labels available after stage1; aborting")

    dataset_stage1 = Path(f"data/datasets/yolo/{params.sku}_bootstrap_stage1")
    data_yaml_stage1 = build_yolo_dataset(
        labeled_images=labeled_stage1,
        labels_dir=params.labels_dir,
        out_dataset_dir=dataset_stage1,
        class_name="product",
    )

    model_dir_stage1 = Path(f"data/models/yolo/{params.sku}_bootstrap_stage1")
    best_stage1 = train_yolo_model(
        data_yaml=data_yaml_stage1,
        model_out_dir=model_dir_stage1,
        config=TrainConfig(
            base_model=params.base_model,
            imgsz=params.imgsz,
            epochs=params.epochs_stage1,
            conf=params.conf_propose,
        ),
    )
    manifest.stage1_model = str(best_stage1)
    save_manifest(manifest_path, manifest)

    proposer = YOLO(str(best_stage1))

    selectable_stage2 = {
        image.name
        for image in images
        if should_process(image.name, params.overwrite, set(manifest.records.keys()))
    }
    stage2_candidates: list[Path] = []
    for section in sections:
        stage2_candidates.extend(_pick_evenly(section, params.approve_per_section, selectable_stage2))

    for image in stage2_candidates:
        label_path = _label_path_for_image(params.labels_dir, image)
        result = _approve_one_image_with_model(
            img_path=image,
            label_path=label_path,
            model=proposer,
            class_id=params.class_id,
            conf=params.conf_propose,
            imgsz=params.imgsz,
            pick_largest=params.pick_largest,
            allow_negatives=params.allow_negatives,
        )

        if result == "quit":
            break

        if result in {"positive", "negative", "skipped"}:
            upsert_record(manifest, image.name, result, str(label_path))
            save_manifest(manifest_path, manifest)

    if params.retrain_after_approve:
        labeled_stage2 = collect_labeled_images(params.frames_dir, params.labels_dir)
        dataset_stage2 = Path(f"data/datasets/yolo/{params.sku}_bootstrap_stage2")
        data_yaml_stage2 = build_yolo_dataset(
            labeled_images=labeled_stage2,
            labels_dir=params.labels_dir,
            out_dataset_dir=dataset_stage2,
            class_name="product",
        )

        model_dir_stage2 = Path(f"data/models/yolo/{params.sku}_bootstrap_stage2")
        best_stage2 = train_yolo_model(
            data_yaml=data_yaml_stage2,
            model_out_dir=model_dir_stage2,
            config=TrainConfig(
                base_model=str(best_stage1),
                imgsz=params.imgsz,
                epochs=params.epochs_stage2,
                conf=params.conf_propose,
            ),
        )
        manifest.stage2_model = str(best_stage2)
        save_manifest(manifest_path, manifest)
