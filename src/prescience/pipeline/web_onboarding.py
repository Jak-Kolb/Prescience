"""Browser-driven onboarding utilities (no OpenCV interactive windows)."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
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
from prescience.vision.vlm_labeler import GeminiBatchResult, GeminiBox, GeminiLabeler, GeminiLabelerError


VERSION_DIR_PATTERN_TEMPLATE = r"^{sku}_v(\d+)$"


@dataclass
class FrameProposal:
    """Detector proposal for one frame."""

    frame_name: str
    boxes: list[dict[str, int]] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Gemini validation decisions over detector proposals."""

    frame_results: list[dict[str, Any]] = field(default_factory=list)
    accepted: int = 0
    adjusted: int = 0
    rejected_negative: int = 0
    uncertain: int = 0


@dataclass
class AutoLabelSummary:
    """Outcome of fully automatic stage2 labeling set creation."""

    total: int
    accepted: int
    adjusted: int
    rejected_negative: int
    uncertain: int
    decisions: list[dict[str, Any]]


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


def _section_target_counts(total: int, sections: int) -> list[int]:
    if sections <= 0:
        raise ValueError("sections must be > 0")
    base = total // sections
    remainder = total % sections
    return [base + (1 if idx < remainder else 0) for idx in range(sections)]


def _select_evenly_binned_targets(
    sectioned_paths: list[list[Path]],
    *,
    selectable: set[str],
    target_count: int,
) -> list[Path]:
    if target_count <= 0:
        return []

    counts = _section_target_counts(target_count, max(1, len(sectioned_paths)))
    selected: list[Path] = []
    selected_names: set[str] = set()
    for idx, section in enumerate(sectioned_paths):
        for image in _pick_evenly(section, counts[idx], selectable):
            if image.name in selected_names:
                continue
            selected.append(image)
            selected_names.add(image.name)

    if len(selected) >= target_count:
        return selected[:target_count]

    remaining = target_count - len(selected)
    all_selectable = [image for section in sectioned_paths for image in section if image.name in selectable]
    leftovers = [image for image in all_selectable if image.name not in selected_names]
    if leftovers:
        idxs = np.linspace(0, len(leftovers) - 1, num=min(remaining, len(leftovers)), dtype=int).tolist()
        for idx in idxs:
            selected.append(leftovers[idx])
    return selected[:target_count]


def _gemini_api_key_from_env(api_key_env: str) -> tuple[str | None, str | None]:
    if not api_key_env.strip():
        return None, "missing_api_key_env_name"
    value = os.getenv(api_key_env, "").strip()
    if not value:
        return None, f"missing_api_key_env:{api_key_env}"
    return value, None


def _build_gemini_labeler(
    *,
    gemini_model: str,
    gemini_api_key_env: str,
) -> tuple[GeminiLabeler | None, str | None]:
    api_key, env_error = _gemini_api_key_from_env(gemini_api_key_env)
    if env_error is not None:
        return None, env_error
    try:
        return GeminiLabeler(api_key=api_key or "", model_name=gemini_model), None
    except GeminiLabelerError as exc:
        return None, str(exc)


def _boxes_to_yolo(
    *,
    boxes: list[dict[str, int]],
    width: int,
    height: int,
) -> list[tuple[float, float, float, float]]:
    return [
        _xyxy_to_yolo_norm(
            int(box["x1"]),
            int(box["y1"]),
            int(box["x2"]),
            int(box["y2"]),
            width,
            height,
        )
        for box in boxes
    ]


def _gemini_batch_proposals_for_images(
    *,
    image_paths: list[Path],
    object_description: str,
    max_boxes: int,
    enabled: bool,
    gemini_model: str,
    gemini_api_key_env: str,
    batch_size: int,
) -> tuple[dict[str, list[dict[str, int]]], dict[str, str]]:
    proposals_by_name: dict[str, list[dict[str, int]]] = {path.name: [] for path in image_paths}
    errors_by_name: dict[str, str] = {}
    if not enabled:
        for path in image_paths:
            errors_by_name[path.name] = "gemini_disabled"
        return proposals_by_name, errors_by_name
    labeler, labeler_error = _build_gemini_labeler(
        gemini_model=gemini_model,
        gemini_api_key_env=gemini_api_key_env,
    )

    if labeler is None:
        fallback_error = labeler_error or "gemini_unavailable"
        for path in image_paths:
            errors_by_name[path.name] = fallback_error
        return proposals_by_name, errors_by_name

    safe_batch_size = max(1, int(batch_size))
    for start in range(0, len(image_paths), safe_batch_size):
        chunk = image_paths[start : start + safe_batch_size]
        try:
            result: GeminiBatchResult = labeler.propose_batch(
                image_paths=chunk,
                object_description=object_description,
                max_boxes=max_boxes,
            )
        except Exception as exc:  # noqa: BLE001
            message = f"gemini_batch_failed: {exc}"
            for path in chunk:
                errors_by_name[path.name] = message
            continue

        for path in chunk:
            proposals = []
            image = cv2.imread(str(path))
            if image is None:
                errors_by_name[path.name] = "image_load_failed"
                proposals_by_name[path.name] = []
                continue
            height, width = image.shape[:2]
            for box in result.boxes_by_image.get(path.name, []):
                if not isinstance(box, GeminiBox):
                    continue
                x1, y1, x2, y2 = box.to_xyxy_pixels(width=width, height=height)
                proposals.append(
                    {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2),
                    }
                )
            proposals_by_name[path.name] = proposals
            if path.name in result.errors_by_image:
                errors_by_name[path.name] = result.errors_by_image[path.name]

    return proposals_by_name, errors_by_name


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
    if append_video_path and not selectable:
        return []
    section_source = [image for image in images if image.name in selectable] or images
    sectioned = _split_into_sections(section_source, sections)
    candidates: list[dict[str, Any]] = []
    for section in sectioned:
        for image in _pick_evenly(section, manual_per_section, selectable):
            candidates.append({"frame_name": image.name, "status": "pending", "boxes": []})
    return candidates


def prepare_seed_candidates_with_gemini(
    sku: str,
    *,
    object_description: str,
    enabled: bool = True,
    target_count: int = 24,
    sections: int = 6,
    overwrite: bool = False,
    append_video_path: str | None = None,
    gemini_model: str = "gemini-3-pro-preview",
    gemini_api_key_env: str = "GEMINI_API_KEY",
    batch_size: int = 24,
    max_proposals_per_frame: int = 4,
) -> list[dict[str, Any]]:
    """Prepare seed candidates with Gemini prelabels and persist auto-label files."""
    frames_dir, labels_dir, manifest_path, manifest, images = _manifest_and_images(sku)
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
    if append_video_path and not selectable:
        return []
    section_source = [image for image in images if image.name in selectable] or images
    sectioned = _split_into_sections(section_source, sections)
    selected_images = _select_evenly_binned_targets(
        sectioned_paths=sectioned,
        selectable=selectable if selectable else {image.name for image in section_source},
        target_count=target_count,
    )
    if not selected_images:
        return []

    proposals_by_name, errors_by_name = _gemini_batch_proposals_for_images(
        image_paths=selected_images,
        object_description=object_description,
        max_boxes=max_proposals_per_frame,
        enabled=enabled,
        gemini_model=gemini_model,
        gemini_api_key_env=gemini_api_key_env,
        batch_size=batch_size,
    )

    candidates: list[dict[str, Any]] = []
    for image_path in selected_images:
        proposals = proposals_by_name.get(image_path.name, [])
        reason = errors_by_name.get(image_path.name, "")
        image = cv2.imread(str(image_path))
        label_path = _label_path_for_image(labels_dir, image_path)

        if image is None:
            reason = reason or "image_load_failed"
            proposals = []
            _write_negative_label(label_path)
            auto_status = "negative"
        elif proposals:
            height, width = image.shape[:2]
            yolo_boxes = _boxes_to_yolo(
                boxes=proposals,
                width=width,
                height=height,
            )
            _write_positive_labels(label_path=label_path, class_id=0, yolo_boxes=yolo_boxes)
            auto_status = "positive"
        else:
            _write_negative_label(label_path)
            auto_status = "negative"

        upsert_record(manifest, image_path.name, auto_status, str(label_path))
        candidates.append(
            {
                "frame_name": image_path.name,
                "status": "pending",
                "boxes": proposals.copy(),
                "proposals": proposals,
                "source": "gemini_seed",
                "needs_review": bool(reason),
                "reason": reason,
                "auto_label_status": auto_status,
            }
        )

    save_manifest(manifest_path, manifest)
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


def _box_iou(a: dict[str, int], b: dict[str, int]) -> float:
    x_left = max(int(a["x1"]), int(b["x1"]))
    y_top = max(int(a["y1"]), int(b["y1"]))
    x_right = min(int(a["x2"]), int(b["x2"]))
    y_bottom = min(int(a["y2"]), int(b["y2"]))
    inter_w = max(0, x_right - x_left)
    inter_h = max(0, y_bottom - y_top)
    inter = float(inter_w * inter_h)
    if inter <= 0:
        return 0.0
    area_a = float(max(0, int(a["x2"]) - int(a["x1"])) * max(0, int(a["y2"]) - int(a["y1"])))
    area_b = float(max(0, int(b["x2"]) - int(b["x1"])) * max(0, int(b["y2"]) - int(b["y1"])))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def build_detector_candidates(
    sku: str,
    *,
    model_path: str,
    target_count: int,
    conf: float = 0.02,
    imgsz: int = 640,
    sections: int = 6,
    overwrite: bool = False,
    append_video_path: str | None = None,
    max_proposals_per_frame: int = 8,
    pick_largest: bool = False,
) -> list[FrameProposal]:
    """Build detector proposals for stage2 auto-labeling candidates."""
    frames_dir, _labels_dir, _manifest_path, manifest, images = _manifest_and_images(sku)
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
    if append_video_path and not selectable:
        return []
    section_source = [image for image in images if image.name in selectable] or images
    sectioned = _split_into_sections(section_source, sections)
    selected_images = _select_evenly_binned_targets(
        sectioned_paths=sectioned,
        selectable=selectable if selectable else {image.name for image in section_source},
        target_count=target_count,
    )
    if not selected_images:
        return []

    model = YOLO(model_path)
    proposals: list[FrameProposal] = []
    for image_path in selected_images:
        image = cv2.imread(str(image_path))
        if image is None:
            proposals.append(FrameProposal(frame_name=image_path.name, boxes=[]))
            continue
        results = model.predict(
            source=image,
            conf=conf,
            imgsz=imgsz,
            max_det=max_proposals_per_frame,
            verbose=False,
        )
        result = results[0]
        boxes: list[dict[str, int]] = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            boxes = _collect_proposed_boxes_xyxy(
                boxes_xyxy=boxes_xyxy,
                confidences=confs,
                pick_largest=pick_largest,
                max_boxes=max_proposals_per_frame,
            )
        proposals.append(FrameProposal(frame_name=image_path.name, boxes=boxes))
    return proposals


def validate_detector_candidates_with_gemini(
    *,
    sku: str,
    detector_candidates: list[FrameProposal],
    object_description: str,
    enabled: bool,
    gemini_model: str,
    gemini_api_key_env: str,
    batch_size: int,
    max_boxes_approval: int,
) -> ValidationResult:
    """Validate detector proposals against Gemini and produce per-frame decisions."""
    if not detector_candidates:
        return ValidationResult()

    frames_dir = Path(f"data/derived/frames/{sku}/frames")
    image_paths = [frames_dir / candidate.frame_name for candidate in detector_candidates]
    proposals_by_name, errors_by_name = _gemini_batch_proposals_for_images(
        image_paths=image_paths,
        object_description=object_description,
        max_boxes=max_boxes_approval,
        enabled=enabled,
        gemini_model=gemini_model,
        gemini_api_key_env=gemini_api_key_env,
        batch_size=batch_size,
    )

    result = ValidationResult()
    for candidate in detector_candidates:
        detector_boxes = [dict(box) for box in candidate.boxes]
        gemini_boxes = [dict(box) for box in proposals_by_name.get(candidate.frame_name, [])]
        error = errors_by_name.get(candidate.frame_name, "")

        decision = "accept"
        final_boxes = detector_boxes
        reason = "detector_confirmed"

        if error:
            decision = "uncertain"
            final_boxes = detector_boxes
            reason = error
            result.uncertain += 1
        elif not detector_boxes and not gemini_boxes:
            decision = "reject_negative"
            final_boxes = []
            reason = "both_empty"
            result.rejected_negative += 1
        elif detector_boxes and not gemini_boxes:
            decision = "reject_negative"
            final_boxes = []
            reason = "gemini_rejected_detector"
            result.rejected_negative += 1
        elif not detector_boxes and gemini_boxes:
            decision = "adjust"
            final_boxes = gemini_boxes
            reason = "gemini_added_boxes"
            result.adjusted += 1
        else:
            max_iou = 0.0
            for det_box in detector_boxes:
                for gem_box in gemini_boxes:
                    max_iou = max(max_iou, _box_iou(det_box, gem_box))
            close_count = abs(len(detector_boxes) - len(gemini_boxes)) <= 1
            if max_iou >= 0.5 and close_count:
                decision = "accept"
                final_boxes = detector_boxes
                reason = "iou_match"
                result.accepted += 1
            else:
                decision = "adjust"
                final_boxes = gemini_boxes
                reason = f"gemini_adjusted_iou_{max_iou:.2f}"
                result.adjusted += 1

        frame_result = {
            "frame_name": candidate.frame_name,
            "decision": decision,
            "final_boxes": final_boxes,
            "detector_boxes": detector_boxes,
            "gemini_boxes": gemini_boxes,
            "reason": reason,
        }
        result.frame_results.append(frame_result)

    return result


def auto_label_for_stage2(
    sku: str,
    *,
    model_path: str,
    object_description: str,
    target_count: int,
    conf: float = 0.02,
    imgsz: int = 640,
    sections: int = 6,
    overwrite: bool = False,
    append_video_path: str | None = None,
    enabled: bool = True,
    gemini_model: str = "gemini-3-pro-preview",
    gemini_api_key_env: str = "GEMINI_API_KEY",
    batch_size: int = 24,
    max_boxes_approval: int = 8,
    class_id: int = 0,
) -> AutoLabelSummary:
    """Auto-label stage2 frames via detector proposals validated by Gemini."""
    detector_candidates = build_detector_candidates(
        sku=sku,
        model_path=model_path,
        target_count=target_count,
        conf=conf,
        imgsz=imgsz,
        sections=sections,
        overwrite=overwrite,
        append_video_path=append_video_path,
        max_proposals_per_frame=max_boxes_approval,
    )
    validation = validate_detector_candidates_with_gemini(
        sku=sku,
        detector_candidates=detector_candidates,
        object_description=object_description,
        enabled=enabled,
        gemini_model=gemini_model,
        gemini_api_key_env=gemini_api_key_env,
        batch_size=batch_size,
        max_boxes_approval=max_boxes_approval,
    )

    for item in validation.frame_results:
        decision = str(item["decision"])
        if decision == "uncertain":
            continue
        status = "positive" if item["final_boxes"] else "negative"
        save_browser_label(
            sku=sku,
            frame_name=str(item["frame_name"]),
            status=status,
            boxes=[dict(box) for box in item["final_boxes"]],
            class_id=class_id,
            allow_negatives=True,
        )

    return AutoLabelSummary(
        total=len(validation.frame_results),
        accepted=validation.accepted,
        adjusted=validation.adjusted,
        rejected_negative=validation.rejected_negative,
        uncertain=validation.uncertain,
        decisions=validation.frame_results,
    )


def prepare_approval_candidates(
    sku: str,
    *,
    stage1_model_path: str,
    approve_per_section: int = 5,
    sections: int = 6,
    overwrite: bool = False,
    conf: float = 0.02,
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


def prepare_approval_candidates_with_gemini(
    sku: str,
    *,
    object_description: str,
    target_count: int,
    enabled: bool = True,
    sections: int = 6,
    overwrite: bool = False,
    gemini_model: str = "gemini-3-pro-preview",
    gemini_api_key_env: str = "GEMINI_API_KEY",
    batch_size: int = 24,
    max_proposals_per_frame: int = 8,
    append_video_path: str | None = None,
) -> list[dict[str, Any]]:
    """Prepare approval candidates from Gemini proposals (no label writes)."""
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
    selected_images = _select_evenly_binned_targets(
        sectioned_paths=sectioned,
        selectable=selectable if selectable else {image.name for image in section_source},
        target_count=target_count,
    )
    if not selected_images:
        return []

    proposals_by_name, errors_by_name = _gemini_batch_proposals_for_images(
        image_paths=selected_images,
        object_description=object_description,
        max_boxes=max_proposals_per_frame,
        enabled=enabled,
        gemini_model=gemini_model,
        gemini_api_key_env=gemini_api_key_env,
        batch_size=batch_size,
    )

    candidates: list[dict[str, Any]] = []
    for image_path in selected_images:
        proposals = proposals_by_name.get(image_path.name, [])
        reason = errors_by_name.get(image_path.name, "")
        candidates.append(
            {
                "frame_name": image_path.name,
                "status": "pending",
                "boxes": proposals.copy(),
                "proposals": proposals,
                "source": "gemini_approval",
                "needs_review": bool(reason),
                "reason": reason,
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
            conf=0.02,
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
            conf=0.02,
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
