"""Onboarding labeling workflow with model-assisted approvals."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import tempfile

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
    get_ultralytics_version,
    list_images,
    train_yolo_model,
)
from prescience.training.state import (
    apply_training_state_update,
    can_resume_from_state,
    compute_dataset_hash,
    load_or_create_train_state,
    save_train_state,
    select_training_names,
    train_state_path_for_labels,
)
from prescience.training.strategy import dynamic_quick_epochs, resolve_onboarding_training_config


@dataclass(frozen=True)
class OnboardingParams:
    sku: str
    frames_dir: Path
    labels_dir: Path
    sections: int = 6
    manual_per_section: int = 4
    approve_per_section: int = 5
    class_id: int = 0
    base_model: str = "auto"
    mode: str = "quick"
    dataset_scope: str | None = None
    core_size: int | None = None
    imgsz: int | None = None
    conf_propose: float = 0.03
    epochs_stage1: int | None = None
    retrain_after_approve: bool = True
    epochs_stage2: int | None = None
    patience: int | None = None
    freeze: int | None = None
    workers: int | None = None
    resume: bool = False
    pick_largest: bool = True
    overwrite: bool = False
    allow_negatives: bool = True
    version: int | None = None


VERSION_DIR_PATTERN_TEMPLATE = r"^{sku}_v(\d+)$"


def _next_model_version(sku: str, models_root: Path) -> int:
    """Pick next available integer version for SKU model directory naming."""
    models_root.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(VERSION_DIR_PATTERN_TEMPLATE.format(sku=re.escape(sku)))
    max_seen = 0
    for child in models_root.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if not match:
            continue
        max_seen = max(max_seen, int(match.group(1)))
    return max_seen + 1


def _has_model_versions(sku: str, models_root: Path) -> bool:
    """Return True when SKU already has at least one trained version."""
    if not models_root.exists():
        return False
    pattern = re.compile(VERSION_DIR_PATTERN_TEMPLATE.format(sku=re.escape(sku)))
    for child in models_root.iterdir():
        if not child.is_dir():
            continue
        if pattern.match(child.name) and (child / "best.pt").exists():
            return True
    return False


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


def _write_positive_labels(label_path: Path, class_id: int, yolo_boxes: list[tuple[float, float, float, float]]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}" for (xc, yc, bw, bh) in yolo_boxes]
    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_negative_label(label_path: Path) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("", encoding="utf-8")


class _ManualBoxDrawer:
    def __init__(self, window_name: str):
        self.window_name = window_name
        self.dragging = False
        self.start = (0, 0)
        self.end = (0, 0)
        self.boxes: list[tuple[int, int, int, int]] = []
        cv2.setMouseCallback(window_name, self._on_mouse)

    @staticmethod
    def _normalize_xyxy(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int, int]:
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return (x1, y1, x2, y2)

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start = (x, y)
            self.end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.end = (x, y)
            x1, y1, x2, y2 = self._normalize_xyxy(self.start[0], self.start[1], self.end[0], self.end[1])
            if (x2 - x1) >= 2 and (y2 - y1) >= 2:
                self.boxes.append((x1, y1, x2, y2))

    def preview_box(self) -> tuple[int, int, int, int] | None:
        if not self.dragging:
            return None
        return self._normalize_xyxy(self.start[0], self.start[1], self.end[0], self.end[1])

    def undo_last(self) -> None:
        if self.boxes:
            self.boxes.pop()

    def clear_all(self) -> None:
        self.boxes.clear()


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
    window = "Manual Label (drag boxes, Enter=save, u=undo, c=clear, 0=negative, Esc=skip)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    drawer = _ManualBoxDrawer(window)

    while True:
        disp = image.copy()
        cv2.putText(
            disp,
            "Draw multiple boxes | Enter=save | u=undo c=clear | 0=negative | Esc=skip",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            disp,
            f"Boxes: {len(drawer.boxes)}",
            (10, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        for idx, (x1, y1, x2, y2) in enumerate(drawer.boxes, start=1):
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 220, 120), 2)
            cv2.putText(
                disp,
                str(idx),
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 220, 120),
                2,
            )

        preview = drawer.preview_box()
        if preview is not None:
            x1, y1, x2, y2 = preview
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 200, 255), 2)

        cv2.imshow(window, disp)
        key = cv2.waitKey(20) & 0xFF

        if key == 13:
            if not drawer.boxes:
                continue
            yolo_boxes = [_xyxy_to_yolo_norm(*box, width, height) for box in drawer.boxes]
            _write_positive_labels(label_path, class_id, yolo_boxes)
            cv2.destroyWindow(window)
            return "positive"

        if key == ord("u"):
            drawer.undo_last()

        if key == ord("c"):
            drawer.clear_all()

        if key == ord("0") and allow_negatives:
            _write_negative_label(label_path)
            cv2.destroyWindow(window)
            return "negative"

        if key == 27:
            cv2.destroyWindow(window)
            return "skipped"


def _collect_proposed_boxes_xyxy(
    boxes_xyxy: np.ndarray,
    confidences: np.ndarray,
    pick_largest: bool,
) -> list[tuple[int, int, int, int]]:
    if boxes_xyxy.size == 0:
        return []

    if pick_largest:
        areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
        order = np.argsort(areas)[::-1]
    else:
        order = np.argsort(confidences)[::-1]

    out: list[tuple[int, int, int, int]] = []
    for idx in order.tolist():
        x1, y1, x2, y2 = boxes_xyxy[int(idx)]
        out.append((int(x1), int(y1), int(x2), int(y2)))
    return out


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

    proposals: list[tuple[int, int, int, int]] = []
    if result.boxes is not None and len(result.boxes) > 0:
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        proposals = _collect_proposed_boxes_xyxy(boxes_xyxy, confs, pick_largest=pick_largest)

    window = "Approve (y=accept, n=manual, x=negative, s=skip, q=quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while True:
        disp = image.copy()

        if proposals:
            for idx, (x1, y1, x2, y2) in enumerate(proposals, start=1):
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    disp,
                    f"P{idx}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )
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

        if key == ord("y") and proposals:
            yolo_boxes = [_xyxy_to_yolo_norm(*box, width, height) for box in proposals]
            _write_positive_labels(label_path, class_id, yolo_boxes)
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


def _append_history_new_files(frames_dir: Path) -> set[str]:
    meta_path = frames_dir.parent / "meta.json"
    if not meta_path.exists():
        return set()
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()

    history = raw.get("append_history")
    if not isinstance(history, list):
        return set()

    # Focus only on the most recent append batch so each retrain round
    # reviews frames from the newest video, not all historical appends.
    for item in reversed(history):
        if not isinstance(item, dict):
            continue
        files = item.get("new_files")
        if not isinstance(files, list):
            continue
        selected = {value for value in files if isinstance(value, str)}
        if selected:
            return selected
    return set()


def _select_processable_image_names(
    images: list[Path],
    processable_names: set[str],
    frames_dir: Path,
    overwrite: bool,
) -> set[str]:
    """Use appended-frame pool when append history exists; otherwise use all processable."""
    if overwrite:
        return processable_names

    image_names = {image.name for image in images}
    appended_names = _append_history_new_files(frames_dir)
    if not appended_names:
        return processable_names
    return processable_names.intersection(image_names).intersection(appended_names)


def run_onboarding_labeling(params: OnboardingParams) -> None:
    resolved = resolve_onboarding_training_config(
        mode=params.mode,
        dataset_scope=params.dataset_scope,
        core_size=params.core_size,
        imgsz=params.imgsz,
        epochs_stage1=params.epochs_stage1,
        epochs_stage2=params.epochs_stage2,
        patience=params.patience,
        freeze=params.freeze,
        workers=params.workers,
    )
    base_model = params.base_model if params.base_model != "auto" else "yolov8n.pt"
    print(
        "[onboarding train] "
        f"mode={resolved.mode} scope={resolved.dataset_scope} "
        f"base_model={base_model} imgsz={resolved.imgsz} "
        f"epochs_stage1={resolved.epochs_stage1} epochs_stage2={resolved.epochs_stage2} "
        f"patience={resolved.patience} freeze={resolved.freeze} workers={resolved.workers} "
        f"core_size={resolved.core_size}"
    )

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

    processable_names = {
        image.name
        for image in images
        if should_process(image.name, params.overwrite, set(manifest.records.keys()))
    }
    selectable = _select_processable_image_names(
        images=images,
        processable_names=processable_names,
        frames_dir=params.frames_dir,
        overwrite=params.overwrite,
    )
    if selectable != processable_names:
        print(f"Focusing onboarding candidates on appended unlabeled frames ({len(selectable)} images).")
    section_source = [image for image in images if image.name in selectable] or images
    sections = _split_into_sections(section_source, params.sections)

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

    labeled_stage1_all = collect_labeled_images(params.frames_dir, params.labels_dir)
    if not labeled_stage1_all:
        raise RuntimeError("No labels available after stage1; aborting")

    train_state_path = train_state_path_for_labels(labels_dir=params.labels_dir)
    train_state = load_or_create_train_state(train_state_path)
    ultralytics_version = get_ultralytics_version()
    selection_stage1 = select_training_names(
        labeled_names=[path.name for path in labeled_stage1_all],
        scope=resolved.dataset_scope,
        core_size=resolved.core_size,
        train_state=train_state,
    )
    stage1_name_to_path = {path.name: path for path in labeled_stage1_all}
    labeled_stage1 = [stage1_name_to_path[name] for name in selection_stage1.selected_names if name in stage1_name_to_path]
    if not labeled_stage1:
        raise RuntimeError(
            f"No selected stage1 training images for SKU {params.sku} "
            f"(mode={resolved.mode}, scope={resolved.dataset_scope})."
        )
    models_root = Path("data/models/yolo")
    datasets_root = Path("data/datasets/yolo")
    is_first_enrollment = not _has_model_versions(params.sku, models_root)
    version_num = params.version if params.version is not None else _next_model_version(params.sku, models_root)
    model_tag = f"{params.sku}_v{version_num}"

    new_count_stage1 = len(selection_stage1.new_names)
    effective_epochs_stage1 = resolved.epochs_stage1
    if resolved.mode == "quick" and params.epochs_stage1 is None:
        effective_epochs_stage1 = max(6, dynamic_quick_epochs(new_count_stage1) - 2)
    if is_first_enrollment:
        effective_epochs_stage1 = max(effective_epochs_stage1, 20)
    effective_freeze_stage1 = resolved.freeze
    if resolved.mode == "quick" and params.freeze is None and new_count_stage1 >= 12:
        effective_freeze_stage1 = 0
    dataset_hash_stage1 = compute_dataset_hash(selected_images=labeled_stage1, labels_dir=params.labels_dir)
    print(
        "[onboarding stage1] "
        f"selected={len(labeled_stage1)}/{len(labeled_stage1_all)} "
        f"core={len(selection_stage1.core_names)} new={new_count_stage1} "
        f"epochs={effective_epochs_stage1} freeze={effective_freeze_stage1}"
    )

    if is_first_enrollment:
        print("[onboarding] first SKU enrollment detected; using boosted quick epoch minimums.")

    # Train a stage-1 onboarding seed model.
    if params.retrain_after_approve:
        with tempfile.TemporaryDirectory(prefix=f"{params.sku}_onboarding_stage1_", dir=str(datasets_root)) as tmp_ds:
            stage1_dataset_dir = Path(tmp_ds)
            data_yaml_stage1 = build_yolo_dataset(
                labeled_images=labeled_stage1,
                labels_dir=params.labels_dir,
                out_dataset_dir=stage1_dataset_dir,
                class_name="product",
            )

            with tempfile.TemporaryDirectory(prefix=f"{params.sku}_onboarding_stage1_", dir=str(models_root)) as tmp_model:
                stage1_model_dir = Path(tmp_model)
                best_stage1 = train_yolo_model(
                    data_yaml=data_yaml_stage1,
                    model_out_dir=stage1_model_dir,
                    config=TrainConfig(
                        base_model=base_model,
                        imgsz=resolved.imgsz,
                        epochs=effective_epochs_stage1,
                        conf=params.conf_propose,
                        patience=resolved.patience,
                        freeze=effective_freeze_stage1,
                        workers=resolved.workers,
                    ),
                )

                proposer = YOLO(str(best_stage1))

                processable_stage2 = {
                    image.name
                    for image in images
                    if should_process(image.name, params.overwrite, set(manifest.records.keys()))
                }
                selectable_stage2 = _select_processable_image_names(
                    images=images,
                    processable_names=processable_stage2,
                    frames_dir=params.frames_dir,
                    overwrite=params.overwrite,
                )
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
                        imgsz=resolved.imgsz,
                        pick_largest=params.pick_largest,
                        allow_negatives=params.allow_negatives,
                    )

                    if result == "quit":
                        break

                    if result in {"positive", "negative", "skipped"}:
                        upsert_record(manifest, image.name, result, str(label_path))
                        save_manifest(manifest_path, manifest)

                labeled_stage2_all = collect_labeled_images(params.frames_dir, params.labels_dir)
                selection_stage2 = select_training_names(
                    labeled_names=[path.name for path in labeled_stage2_all],
                    scope=resolved.dataset_scope,
                    core_size=resolved.core_size,
                    train_state=train_state,
                )
                stage2_name_to_path = {path.name: path for path in labeled_stage2_all}
                labeled_stage2 = [stage2_name_to_path[name] for name in selection_stage2.selected_names if name in stage2_name_to_path]
                if not labeled_stage2:
                    raise RuntimeError(
                        f"No selected stage2 training images for SKU {params.sku} "
                        f"(mode={resolved.mode}, scope={resolved.dataset_scope})."
                    )
                new_count_stage2 = len(selection_stage2.new_names)
                effective_epochs_stage2 = resolved.epochs_stage2
                if resolved.mode == "quick" and params.epochs_stage2 is None:
                    effective_epochs_stage2 = dynamic_quick_epochs(new_count_stage2)
                    if is_first_enrollment:
                        effective_epochs_stage2 = max(effective_epochs_stage2, 20)
                effective_freeze_stage2 = resolved.freeze
                if resolved.mode == "quick" and params.freeze is None and new_count_stage2 >= 12:
                    effective_freeze_stage2 = 0
                dataset_hash_stage2 = compute_dataset_hash(selected_images=labeled_stage2, labels_dir=params.labels_dir)
                print(
                    "[onboarding stage2] "
                    f"selected={len(labeled_stage2)}/{len(labeled_stage2_all)} "
                    f"core={len(selection_stage2.core_names)} new={new_count_stage2} "
                    f"epochs={effective_epochs_stage2} freeze={effective_freeze_stage2}"
                )

                final_dataset_dir = Path(f"data/datasets/yolo/{model_tag}")
                data_yaml_stage2 = build_yolo_dataset(
                    labeled_images=labeled_stage2,
                    labels_dir=params.labels_dir,
                    out_dataset_dir=final_dataset_dir,
                    class_name="product",
                )

                final_model_dir = Path(f"data/models/yolo/{model_tag}")
                resume_checkpoint = final_model_dir / "train" / "weights" / "last.pt"
                should_resume = False
                if params.resume:
                    decision = can_resume_from_state(
                        train_state=train_state,
                        requested_dataset_hash=dataset_hash_stage2,
                        requested_mode=resolved.mode,
                        requested_imgsz=resolved.imgsz,
                        requested_base_model_path=base_model,
                        requested_ultralytics_version=ultralytics_version,
                        requested_version_tag=f"v{version_num}",
                    )
                    if decision.allowed and resume_checkpoint.exists():
                        should_resume = True
                        print(f"[onboarding stage2] resume enabled from {resume_checkpoint}")
                    else:
                        reason = decision.reason if not decision.allowed else "missing_resume_checkpoint"
                        print(f"[onboarding stage2] resume skipped ({reason}); starting fresh.")
                best_stage2 = train_yolo_model(
                    data_yaml=data_yaml_stage2,
                    model_out_dir=final_model_dir,
                    config=TrainConfig(
                        base_model=str(best_stage1),
                        imgsz=resolved.imgsz,
                        epochs=effective_epochs_stage2,
                        conf=params.conf_propose,
                        patience=resolved.patience,
                        freeze=effective_freeze_stage2,
                        workers=resolved.workers,
                        resume=should_resume,
                        resume_checkpoint=str(resume_checkpoint) if should_resume else None,
                    ),
                )

                manifest.stage1_model = None
                manifest.stage2_model = str(best_stage2)
                manifest.model_version = f"v{version_num}"
                manifest.final_model = str(best_stage2)
                save_manifest(manifest_path, manifest)
                updated_state = apply_training_state_update(
                    train_state=train_state,
                    selection=selection_stage2,
                    trained_model_path=str(best_stage2),
                    mode=resolved.mode,
                    version_tag=f"v{version_num}",
                    dataset_hash=dataset_hash_stage2,
                    imgsz=resolved.imgsz,
                    base_model_path=base_model,
                    ultralytics_version=ultralytics_version,
                )
                save_train_state(train_state_path, updated_state)
                return

    # If retraining is disabled, stage-1 model becomes the final versioned model.
    final_dataset_dir = Path(f"data/datasets/yolo/{model_tag}")
    data_yaml_stage1 = build_yolo_dataset(
        labeled_images=labeled_stage1,
        labels_dir=params.labels_dir,
        out_dataset_dir=final_dataset_dir,
        class_name="product",
    )
    final_model_dir = Path(f"data/models/yolo/{model_tag}")
    resume_checkpoint = final_model_dir / "train" / "weights" / "last.pt"
    should_resume = False
    if params.resume:
        decision = can_resume_from_state(
            train_state=train_state,
            requested_dataset_hash=dataset_hash_stage1,
            requested_mode=resolved.mode,
            requested_imgsz=resolved.imgsz,
            requested_base_model_path=base_model,
            requested_ultralytics_version=ultralytics_version,
            requested_version_tag=f"v{version_num}",
        )
        if decision.allowed and resume_checkpoint.exists():
            should_resume = True
            print(f"[onboarding stage1] resume enabled from {resume_checkpoint}")
        else:
            reason = decision.reason if not decision.allowed else "missing_resume_checkpoint"
            print(f"[onboarding stage1] resume skipped ({reason}); starting fresh.")
    best_stage1 = train_yolo_model(
        data_yaml=data_yaml_stage1,
        model_out_dir=final_model_dir,
        config=TrainConfig(
            base_model=base_model,
            imgsz=resolved.imgsz,
            epochs=effective_epochs_stage1,
            conf=params.conf_propose,
            patience=resolved.patience,
            freeze=effective_freeze_stage1,
            workers=resolved.workers,
            resume=should_resume,
            resume_checkpoint=str(resume_checkpoint) if should_resume else None,
        ),
    )
    manifest.stage1_model = None
    manifest.stage2_model = str(best_stage1)
    manifest.model_version = f"v{version_num}"
    manifest.final_model = str(best_stage1)
    save_manifest(manifest_path, manifest)
    updated_state = apply_training_state_update(
        train_state=train_state,
        selection=selection_stage1,
        trained_model_path=str(best_stage1),
        mode=resolved.mode,
        version_tag=f"v{version_num}",
        dataset_hash=dataset_hash_stage1,
        imgsz=resolved.imgsz,
        base_model_path=base_model,
        ultralytics_version=ultralytics_version,
    )
    save_train_state(train_state_path, updated_state)


# Backward-compatible aliases while the onboarding naming propagates.
BootstrapParams = OnboardingParams
run_bootstrap_labeling = run_onboarding_labeling
