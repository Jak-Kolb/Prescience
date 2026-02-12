"""Browser-driven onboarding utilities (no OpenCV interactive windows)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

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
from prescience.pipeline.enroll import resolve_base_model_for_sku, train_detector_for_sku
from prescience.training.state import (
    apply_training_state_update,
    compute_dataset_hash,
    load_or_create_train_state,
    save_train_state,
    select_training_names,
    train_state_path_for_labels,
)
from prescience.training.strategy import dynamic_quick_epochs, resolve_onboarding_training_config


VERSION_DIR_PATTERN_TEMPLATE = r"^{sku}_v(\d+)$"


def _paths_for_sku(sku: str) -> tuple[Path, Path, Path]:
    frames_dir = Path(f"data/derived/frames/{sku}/frames")
    labels_dir = Path(f"data/derived/labels/{sku}/labels")
    manifest_path = labels_dir.parent / "manifest.json"
    return frames_dir, labels_dir, manifest_path


def _label_path_for_image(labels_dir: Path, image_path: Path) -> Path:
    return labels_dir / f"{image_path.stem}.txt"


def _write_positive_labels(label_path: Path, class_id: int, yolo_boxes: list[tuple[float, float, float, float]]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}" for (xc, yc, bw, bh) in yolo_boxes]
    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_negative_label(label_path: Path) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("", encoding="utf-8")


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


def _sync_manifest_from_existing_labels(manifest: LabelManifest, labels_dir: Path, images: list[Path]) -> None:
    for image in images:
        label_path = _label_path_for_image(labels_dir, image)
        if not label_path.exists() or image.name in manifest.records:
            continue
        text = label_path.read_text(encoding="utf-8").strip()
        status = "negative" if not text else "positive"
        upsert_record(manifest, image.name, status, str(label_path))


def _split_into_sections(paths: list[Path], sections: int) -> list[list[Path]]:
    if sections <= 0:
        raise ValueError("sections must be > 0")
    n = len(paths)
    out: list[list[Path]] = []
    start = 0
    for idx in range(sections):
        size = n // sections + (1 if idx < (n % sections) else 0)
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


def _append_history_new_files(
    frames_dir: Path,
    *,
    append_video_path: str | None = None,
) -> set[str]:
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

    def _file_set(entry: dict[str, Any]) -> set[str]:
        files = entry.get("new_files")
        if not isinstance(files, list):
            return set()
        return {value for value in files if isinstance(value, str)}

    if append_video_path:
        target = Path(append_video_path).as_posix()
        for item in reversed(history):
            if not isinstance(item, dict):
                continue
            video_path = item.get("video_path")
            if isinstance(video_path, str) and Path(video_path).as_posix() == target:
                return _file_set(item)

    # Fallback: focus on most recent append batch only.
    for item in reversed(history):
        if not isinstance(item, dict):
            continue
        selected = _file_set(item)
        if selected:
            return selected
    return set()


def _select_processable_image_names(
    images: list[Path],
    processable_names: set[str],
    frames_dir: Path,
    overwrite: bool,
    append_video_path: str | None = None,
) -> set[str]:
    if overwrite:
        return processable_names

    image_names = {image.name for image in images}
    appended_names = _append_history_new_files(frames_dir, append_video_path=append_video_path)
    if not appended_names:
        return processable_names
    return processable_names.intersection(image_names).intersection(appended_names)


def _next_model_version(sku: str, models_root: Path) -> int:
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
    pattern = re.compile(VERSION_DIR_PATTERN_TEMPLATE.format(sku=re.escape(sku)))
    if not models_root.exists():
        return False
    for child in models_root.iterdir():
        if child.is_dir() and pattern.match(child.name) and (child / "best.pt").exists():
            return True
    return False


def next_model_version_for_sku(sku: str) -> int:
    """Next stable version number for a SKU."""
    return _next_model_version(sku=sku, models_root=Path("data/models/yolo"))


def _manifest_and_images(sku: str) -> tuple[Path, Path, Path, LabelManifest, list[Path]]:
    frames_dir, labels_dir, manifest_path = _paths_for_sku(sku)
    labels_dir.mkdir(parents=True, exist_ok=True)
    images = list_images(frames_dir)
    if not images:
        raise RuntimeError(f"No extracted frames found for SKU {sku}")
    manifest = load_or_create_manifest(
        manifest_path=manifest_path,
        sku=sku,
        frames_dir=frames_dir,
        labels_dir=labels_dir,
    )
    _sync_manifest_from_existing_labels(manifest=manifest, labels_dir=labels_dir, images=images)
    save_manifest(manifest_path, manifest)
    return frames_dir, labels_dir, manifest_path, manifest, images


def prepare_seed_candidates(
    sku: str,
    *,
    manual_per_section: int = 4,
    sections: int = 6,
    overwrite: bool = False,
    append_video_path: str | None = None,
) -> list[dict[str, Any]]:
    """Prepare seed labeling candidates for browser onboarding."""
    frames_dir, _labels_dir, _manifest_path, manifest, images = _manifest_and_images(sku)
    sectioned = _split_into_sections(images, sections)
    processable_names = {
        image.name
        for image in images
        if should_process(image.name, overwrite, set(manifest.records.keys()))
    }
    selectable = _select_processable_image_names(
        images=images,
        processable_names=processable_names,
        frames_dir=frames_dir,
        overwrite=overwrite,
        append_video_path=append_video_path,
    )
    section_source = [image for image in images if image.name in selectable] or images
    sectioned = _split_into_sections(section_source, sections)
    candidates: list[dict[str, Any]] = []
    for section in sectioned:
        for image in _pick_evenly(section, manual_per_section, selectable):
            candidates.append({"frame_name": image.name, "status": "pending", "boxes": []})
    return candidates


def save_browser_label(
    sku: str,
    *,
    frame_name: str,
    status: str,
    boxes: list[dict[str, int]],
    class_id: int = 0,
    allow_negatives: bool = True,
) -> dict[str, Any]:
    """Persist one browser-submitted label and update manifest."""
    if status not in {"positive", "negative", "skipped"}:
        raise ValueError("status must be one of: positive, negative, skipped")

    frames_dir, labels_dir, manifest_path, manifest, _images = _manifest_and_images(sku)
    image_path = frames_dir / frame_name
    if not image_path.exists():
        raise FileNotFoundError(f"Frame not found: {image_path}")

    label_path = labels_dir / f"{image_path.stem}.txt"
    if status == "positive":
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not load frame for labeling: {image_path}")
        h, w = image.shape[:2]
        if not boxes:
            raise ValueError("Positive labels require at least one box")
        yolo_boxes = [
            _xyxy_to_yolo_norm(
                int(box["x1"]),
                int(box["y1"]),
                int(box["x2"]),
                int(box["y2"]),
                w,
                h,
            )
            for box in boxes
        ]
        _write_positive_labels(label_path=label_path, class_id=class_id, yolo_boxes=yolo_boxes)
    elif status == "negative":
        if not allow_negatives:
            raise ValueError("Negative labels are disabled for this workflow")
        _write_negative_label(label_path)
    else:
        # Skipped frames keep existing labels untouched.
        pass

    upsert_record(manifest, image_path.name, status, str(label_path))
    save_manifest(manifest_path, manifest)

    return {"frame_name": frame_name, "status": status}


def _collect_proposed_boxes_xyxy(
    boxes_xyxy: np.ndarray,
    confidences: np.ndarray,
    pick_largest: bool,
    max_boxes: int | None = None,
) -> list[dict[str, int]]:
    if boxes_xyxy.size == 0:
        return []

    if pick_largest:
        areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
        order = np.argsort(areas)[::-1]
    else:
        order = np.argsort(confidences)[::-1]

    out: list[dict[str, int]] = []
    selected = order.tolist()
    if max_boxes is not None and max_boxes > 0:
        selected = selected[:max_boxes]

    for idx in selected:
        x1, y1, x2, y2 = boxes_xyxy[int(idx)]
        out.append({"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)})
    return out


def prepare_approval_candidates(
    sku: str,
    *,
    stage1_model_path: str,
    approve_per_section: int = 5,
    sections: int = 6,
    overwrite: bool = False,
    conf: float = 0.03,
    imgsz: int = 640,
    pick_largest: bool = False,
    max_proposals_per_frame: int = 8,
    append_video_path: str | None = None,
) -> list[dict[str, Any]]:
    """Prepare approval candidates with model proposals for browser workflow."""
    frames_dir, labels_dir, manifest_path, manifest, images = _manifest_and_images(sku)
    _ = labels_dir
    _ = manifest_path
    processable_names = {
        image.name
        for image in images
        if should_process(image.name, overwrite, set(manifest.records.keys()))
    }
    selectable = _select_processable_image_names(
        images=images,
        processable_names=processable_names,
        frames_dir=frames_dir,
        overwrite=overwrite,
        append_video_path=append_video_path,
    )
    section_source = [image for image in images if image.name in selectable] or images
    sectioned = _split_into_sections(section_source, sections)

    model = YOLO(stage1_model_path)
    candidates: list[dict[str, Any]] = []
    for section in sectioned:
        for image_path in _pick_evenly(section, approve_per_section, selectable):
            image = cv2.imread(str(image_path))
            if image is None:
                proposals: list[dict[str, int]] = []
            else:
                results = model.predict(
                    source=image,
                    conf=conf,
                    imgsz=imgsz,
                    max_det=max_proposals_per_frame,
                    verbose=False,
                )
                result = results[0]
                proposals = []
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    proposals = _collect_proposed_boxes_xyxy(
                        boxes_xyxy=boxes_xyxy,
                        confidences=confs,
                        pick_largest=pick_largest,
                        max_boxes=max_proposals_per_frame,
                    )
            candidates.append(
                {
                    "frame_name": image_path.name,
                    "status": "pending",
                    "boxes": proposals.copy(),
                    "proposals": proposals,
                }
            )

    return candidates


def train_stage1_for_session(
    sku: str,
    *,
    version_num: int,
    mode: str = "quick",
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    """Train stage1 helper model for browser onboarding."""
    frames_dir, labels_dir, _manifest_path = _paths_for_sku(sku)
    labeled_stage1_all = collect_labeled_images(frames_dir, labels_dir)
    if not labeled_stage1_all:
        raise RuntimeError("No labels available for stage1 training")

    resolved = resolve_onboarding_training_config(
        mode=mode,
        dataset_scope=None,
        core_size=None,
        imgsz=None,
        epochs_stage1=None,
        epochs_stage2=None,
        patience=None,
        freeze=None,
        workers=None,
    )
    models_root = Path("data/models/yolo")
    datasets_root = Path("data/datasets/yolo")
    is_first_enrollment = not _has_model_versions(sku, models_root)
    base_model = resolve_base_model_for_sku(
        sku=sku,
        base_model="auto",
        target_version=version_num,
    )

    train_state_path = train_state_path_for_labels(labels_dir=labels_dir)
    train_state = load_or_create_train_state(train_state_path)
    selection = select_training_names(
        labeled_names=[path.name for path in labeled_stage1_all],
        scope=resolved.dataset_scope,
        core_size=resolved.core_size,
        train_state=train_state,
    )
    name_to_path = {path.name: path for path in labeled_stage1_all}
    labeled_stage1 = [name_to_path[name] for name in selection.selected_names if name in name_to_path]
    if not labeled_stage1:
        raise RuntimeError("No selected images for stage1 training")

    new_count = len(selection.new_names)
    effective_epochs = resolved.epochs_stage1
    if mode == "quick":
        effective_epochs = max(6, dynamic_quick_epochs(new_count) - 2)
    if is_first_enrollment:
        effective_epochs = max(effective_epochs, 20)

    effective_freeze = resolved.freeze
    if mode == "quick" and new_count >= 12:
        effective_freeze = 0

    stage1_dataset_dir = datasets_root / f"{sku}_onboarding_stage1_{uuid4().hex[:8]}"
    data_yaml = build_yolo_dataset(
        labeled_images=labeled_stage1,
        labels_dir=labels_dir,
        out_dataset_dir=stage1_dataset_dir,
        class_name="product",
    )
    stage1_model_dir = models_root / f"{sku}_onboarding_stage1_{uuid4().hex[:8]}"
    best_stage1 = train_yolo_model(
        data_yaml=data_yaml,
        model_out_dir=stage1_model_dir,
        config=TrainConfig(
            base_model=base_model,
            imgsz=resolved.imgsz,
            epochs=effective_epochs,
            conf=0.01,
            patience=resolved.patience,
            freeze=effective_freeze,
            workers=resolved.workers,
        ),
        progress_cb=(None if progress_cb is None else lambda payload: progress_cb({"stage": "train_stage1", **payload})),
    )
    return str(best_stage1)


def train_stage2_for_session(
    sku: str,
    *,
    version_num: int,
    stage1_model_path: str,
    mode: str = "quick",
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    """Train final stage2 model for browser onboarding."""
    frames_dir, labels_dir, manifest_path = _paths_for_sku(sku)
    labeled_stage2_all = collect_labeled_images(frames_dir, labels_dir)
    if not labeled_stage2_all:
        raise RuntimeError("No labels available for stage2 training")

    resolved = resolve_onboarding_training_config(
        mode=mode,
        dataset_scope=None,
        core_size=None,
        imgsz=None,
        epochs_stage1=None,
        epochs_stage2=None,
        patience=None,
        freeze=None,
        workers=None,
    )
    models_root = Path("data/models/yolo")
    is_first_enrollment = not _has_model_versions(sku, models_root)

    train_state_path = train_state_path_for_labels(labels_dir=labels_dir)
    train_state = load_or_create_train_state(train_state_path)
    selection = select_training_names(
        labeled_names=[path.name for path in labeled_stage2_all],
        scope=resolved.dataset_scope,
        core_size=resolved.core_size,
        train_state=train_state,
    )
    name_to_path = {path.name: path for path in labeled_stage2_all}
    labeled_stage2 = [name_to_path[name] for name in selection.selected_names if name in name_to_path]
    if not labeled_stage2:
        raise RuntimeError("No selected images for stage2 training")

    new_count = len(selection.new_names)
    effective_epochs = resolved.epochs_stage2
    if mode == "quick":
        effective_epochs = dynamic_quick_epochs(new_count)
        if is_first_enrollment:
            effective_epochs = max(effective_epochs, 20)

    effective_freeze = resolved.freeze
    if mode == "quick" and new_count >= 12:
        effective_freeze = 0

    model_tag = f"{sku}_v{version_num}"
    dataset_dir = Path(f"data/datasets/yolo/{model_tag}")
    data_yaml = build_yolo_dataset(
        labeled_images=labeled_stage2,
        labels_dir=labels_dir,
        out_dataset_dir=dataset_dir,
        class_name="product",
    )
    model_dir = Path(f"data/models/yolo/{model_tag}")
    best = train_yolo_model(
        data_yaml=data_yaml,
        model_out_dir=model_dir,
        config=TrainConfig(
            base_model=stage1_model_path,
            imgsz=resolved.imgsz,
            epochs=effective_epochs,
            conf=0.01,
            patience=resolved.patience,
            freeze=effective_freeze,
            workers=resolved.workers,
        ),
        progress_cb=(None if progress_cb is None else lambda payload: progress_cb({"stage": "train_stage2", **payload})),
    )

    manifest = load_or_create_manifest(
        manifest_path=manifest_path,
        sku=sku,
        frames_dir=frames_dir,
        labels_dir=labels_dir,
    )
    manifest.stage1_model = stage1_model_path
    manifest.stage2_model = str(best)
    manifest.model_version = f"v{version_num}"
    manifest.final_model = str(best)
    save_manifest(manifest_path, manifest)

    dataset_hash = compute_dataset_hash(selected_images=labeled_stage2, labels_dir=labels_dir)
    updated_state = apply_training_state_update(
        train_state=train_state,
        selection=selection,
        trained_model_path=str(best),
        mode=resolved.mode,
        version_tag=f"v{version_num}",
        dataset_hash=dataset_hash,
        imgsz=resolved.imgsz,
        base_model_path=resolve_base_model_for_sku(sku=sku, base_model="auto", target_version=version_num),
        ultralytics_version=get_ultralytics_version(),
    )
    save_train_state(train_state_path, updated_state)
    return str(best)


def run_full_train_for_sku(
    sku: str,
    *,
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    """Run full retraining and emit progress updates."""
    version_num = next_model_version_for_sku(sku)
    best = train_detector_for_sku(
        sku=sku,
        version=f"v{version_num}",
        mode="full",
        progress_cb=(None if progress_cb is None else lambda payload: progress_cb({"stage": "full_train", **payload})),
    )
    return str(best)
