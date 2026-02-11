"""Enrollment and training pipeline orchestration."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from prescience.datasets.bootstrap_label import OnboardingParams, run_onboarding_labeling
from prescience.datasets.yolo import TrainConfig, build_yolo_dataset, collect_labeled_images, train_yolo_model
from prescience.ingest.video_to_frames import ExtractParams, extract_frames
from prescience.profiles.io import save_profile
from prescience.profiles.schema import ProfileMetadata, ProfileModelInfo, SKUProfile
from prescience.training.state import (
    apply_training_state_update,
    load_or_create_train_state,
    save_train_state,
    select_training_names,
    train_state_path_for_labels,
)
from prescience.training.strategy import (
    resolve_detector_training_config,
    resolve_onboarding_training_config,
)
from prescience.vision.embeddings import build_embedder

SKU_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
VIDEO_NAME_PATTERN_TEMPLATE = r"^{sku}_(\d+)\.[A-Za-z0-9]+$"
MODEL_VERSION_DIR_PATTERN_TEMPLATE = r"^{sku}_v(\d+)$"
VERSION_TAG_PATTERN = re.compile(r"^v(\d+)$")


def normalize_sku_name(sku: str) -> str:
    """Normalize and validate user-provided SKU names."""
    normalized = sku.strip()
    if not normalized:
        raise ValueError("SKU name cannot be empty")
    if not SKU_NAME_PATTERN.fullmatch(normalized):
        raise ValueError("SKU name must match [A-Za-z0-9_-]+")
    return normalized


def next_enrollment_video_path(raw_videos_root: Path, sku: str, suffix: str = ".MOV") -> Path:
    """Compute next auto-numbered raw enrollment video path for SKU."""
    normalized = normalize_sku_name(sku)
    sku_dir = raw_videos_root / normalized
    sku_dir.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(VIDEO_NAME_PATTERN_TEMPLATE.format(sku=re.escape(normalized)))
    max_idx = -1
    for file_path in sku_dir.iterdir():
        if not file_path.is_file():
            continue
        match = pattern.match(file_path.name)
        if match:
            max_idx = max(max_idx, int(match.group(1)))

    next_idx = max_idx + 1
    return sku_dir / f"{normalized}_{next_idx}{suffix}"


def append_frames_to_existing_dataset(source_frames_dir: Path, target_frames_dir: Path) -> list[Path]:
    """Append extracted frames into existing SKU frame set with new sequential indices."""
    target_frames_dir.mkdir(parents=True, exist_ok=True)

    max_idx = 0
    for existing in target_frames_dir.glob("*.jpg"):
        stem = existing.stem
        if stem.isdigit():
            max_idx = max(max_idx, int(stem))

    next_idx = max_idx + 1
    moved: list[Path] = []

    for source in sorted(source_frames_dir.glob("*.jpg")):
        destination = target_frames_dir / f"{next_idx:06d}.jpg"
        shutil.move(str(source), str(destination))
        moved.append(destination)
        next_idx += 1

    return moved


def _parse_numeric_version(version: str | None) -> int | None:
    if version is None:
        return None
    match = VERSION_TAG_PATTERN.fullmatch(version.strip())
    if match is None:
        return None
    return int(match.group(1))


def _list_model_versions_for_sku(sku: str, models_root: Path) -> list[int]:
    if not models_root.exists():
        return []
    pattern = re.compile(MODEL_VERSION_DIR_PATTERN_TEMPLATE.format(sku=re.escape(sku)))
    out: list[int] = []
    for child in models_root.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if match is None:
            continue
        best = child / "best.pt"
        if best.exists():
            out.append(int(match.group(1)))
    return sorted(set(out))


def resolve_base_model_for_sku(
    *,
    sku: str,
    base_model: str,
    target_version: int | None,
    models_root: Path = Path("data/models/yolo"),
    default_base_model: str = "yolov8n.pt",
) -> str:
    """Resolve base model path; auto picks latest prior version when available."""
    if base_model != "auto":
        return base_model

    versions = _list_model_versions_for_sku(sku=sku, models_root=models_root)
    if target_version is not None:
        versions = [version for version in versions if version < target_version]

    for version in sorted(versions, reverse=True):
        candidate = models_root / f"{sku}_v{version}" / "best.pt"
        if candidate.exists():
            return str(candidate)
    return default_base_model


def extract_frames_for_sku(
    video_path: Path,
    sku: str,
    target_frames: int,
    out_root: Path,
    blur_min: float,
    dedupe_max_similarity: float,
    append: bool = False,
) -> dict:
    """Extract enrollment frames for a SKU."""
    normalized = normalize_sku_name(sku)
    out_root.mkdir(parents=True, exist_ok=True)

    params = ExtractParams(
        target_frames=target_frames,
        blur_min=blur_min,
        dedupe_max_similarity=dedupe_max_similarity,
    )

    if not append:
        meta = extract_frames(video_path=video_path, sku=normalized, out_root=out_root, params=params)
        print(f"Saved {meta['num_frames_saved']} frames to {meta['frames_dir']}")
        return meta

    sku_out_dir = out_root / normalized
    target_frames_dir = sku_out_dir / "frames"
    target_meta_path = sku_out_dir / "meta.json"

    previous_meta: dict[str, Any] = {}
    if target_meta_path.exists():
        previous_meta = json.loads(target_meta_path.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory(prefix=f"{normalized}_extract_", dir=str(out_root)) as tmp_dir:
        tmp_root = Path(tmp_dir)
        temp_meta = extract_frames(
            video_path=video_path,
            sku=normalized,
            out_root=tmp_root,
            params=params,
        )
        source_frames_dir = Path(temp_meta["frames_dir"])
        moved = append_frames_to_existing_dataset(source_frames_dir=source_frames_dir, target_frames_dir=target_frames_dir)

    all_saved = sorted(path.name for path in target_frames_dir.glob("*.jpg"))
    append_history = previous_meta.get("append_history", [])
    append_history.append(
        {
            "video_path": str(video_path),
            "appended_count": len(moved),
            "new_files": [path.name for path in moved],
        }
    )

    merged_meta: dict[str, Any] = {
        "sku": normalized,
        "video_path": str(video_path),
        "out_dir": str(sku_out_dir),
        "frames_dir": str(target_frames_dir),
        "num_frames_saved": len(all_saved),
        "saved_filenames": all_saved,
        "append_mode": True,
        "append_history": append_history,
    }
    target_meta_path.write_text(json.dumps(merged_meta, indent=2), encoding="utf-8")
    print(f"Appended {len(moved)} frames. Total frames in dataset: {len(all_saved)}")
    return merged_meta


def run_onboarding_labeling_for_sku(
    sku: str,
    manual_per_section: int = 2,
    approve_per_section: int = 5,
    overwrite: bool = False,
    allow_negatives: bool = True,
    base_model: str = "auto",
    mode: str = "quick",
    dataset_scope: str | None = None,
    core_size: int | None = None,
    imgsz: int | None = None,
    epochs_stage1: int | None = None,
    epochs_stage2: int | None = None,
    patience: int | None = None,
    freeze: int | None = None,
    workers: int | None = None,
    version: int | None = None,
) -> None:
    """Launch two-stage onboarding labeling workflow for SKU."""
    resolved = resolve_onboarding_training_config(
        mode=mode,
        dataset_scope=dataset_scope,
        core_size=core_size,
        imgsz=imgsz,
        epochs_stage1=epochs_stage1,
        epochs_stage2=epochs_stage2,
        patience=patience,
        freeze=freeze,
        workers=workers,
    )
    resolved_base_model = resolve_base_model_for_sku(
        sku=sku,
        base_model=base_model,
        target_version=version,
    )
    print(
        "[onboarding] "
        f"mode={resolved.mode} "
        f"scope={resolved.dataset_scope} "
        f"base_model={resolved_base_model} "
        f"imgsz={resolved.imgsz} "
        f"epochs_stage1={resolved.epochs_stage1} "
        f"epochs_stage2={resolved.epochs_stage2} "
        f"patience={resolved.patience} "
        f"freeze={resolved.freeze} "
        f"workers={resolved.workers} "
        f"core_size={resolved.core_size}"
    )

    run_onboarding_labeling(
        OnboardingParams(
            sku=sku,
            frames_dir=Path(f"data/derived/frames/{sku}/frames"),
            labels_dir=Path(f"data/derived/labels/{sku}/labels"),
            manual_per_section=manual_per_section,
            approve_per_section=approve_per_section,
            overwrite=overwrite,
            allow_negatives=allow_negatives,
            base_model=resolved_base_model,
            mode=resolved.mode,
            dataset_scope=resolved.dataset_scope,
            core_size=resolved.core_size,
            imgsz=resolved.imgsz,
            epochs_stage1=resolved.epochs_stage1,
            epochs_stage2=resolved.epochs_stage2,
            patience=resolved.patience,
            freeze=resolved.freeze,
            workers=resolved.workers,
            version=version,
        )
    )


# Backward-compatible alias for older imports.
run_bootstrap_labeling_for_sku = run_onboarding_labeling_for_sku


def train_detector_for_sku(
    sku: str,
    version: str,
    mode: str = "quick",
    dataset_scope: str | None = None,
    core_size: int | None = None,
    epochs: int | None = None,
    imgsz: int | None = None,
    patience: int | None = None,
    freeze: int | None = None,
    workers: int | None = None,
    conf: float = 0.35,
    base_model: str = "auto",
) -> Path:
    """Train detector from current labeled frames with mode-aware dataset scope."""
    frames_dir = Path(f"data/derived/frames/{sku}/frames")
    labels_dir = Path(f"data/derived/labels/{sku}/labels")

    resolved = resolve_detector_training_config(
        mode=mode,
        dataset_scope=dataset_scope,
        core_size=core_size,
        imgsz=imgsz,
        epochs=epochs,
        patience=patience,
        freeze=freeze,
        workers=workers,
    )
    resolved_base_model = resolve_base_model_for_sku(
        sku=sku,
        base_model=base_model,
        target_version=_parse_numeric_version(version),
    )

    labeled = collect_labeled_images(frames_dir=frames_dir, labels_dir=labels_dir)
    if not labeled:
        raise RuntimeError(f"No labeled images found for SKU {sku}")

    train_state_path = train_state_path_for_labels(labels_dir=labels_dir)
    train_state = load_or_create_train_state(train_state_path)
    selection = select_training_names(
        labeled_names=[path.name for path in labeled],
        scope=resolved.dataset_scope,
        core_size=resolved.core_size,
        train_state=train_state,
    )
    name_to_path = {path.name: path for path in labeled}
    selected = [name_to_path[name] for name in selection.selected_names if name in name_to_path]
    if not selected:
        raise RuntimeError(
            f"No selected training images for SKU {sku} (mode={resolved.mode}, scope={resolved.dataset_scope})."
        )
    print(
        "[train detector] "
        f"mode={resolved.mode} scope={resolved.dataset_scope} "
        f"base_model={resolved_base_model} imgsz={resolved.imgsz} epochs={resolved.epochs} "
        f"patience={resolved.patience} freeze={resolved.freeze} workers={resolved.workers} "
        f"selected={len(selected)}/{len(labeled)} core={len(selection.core_names)} new={len(selection.new_names)}"
    )

    dataset_dir = Path(f"data/datasets/yolo/{sku}_{version}")
    data_yaml = build_yolo_dataset(
        labeled_images=selected,
        labels_dir=labels_dir,
        out_dataset_dir=dataset_dir,
        class_name="product",
    )

    model_dir = Path(f"data/models/yolo/{sku}_{version}")
    best = train_yolo_model(
        data_yaml=data_yaml,
        model_out_dir=model_dir,
        config=TrainConfig(
            base_model=resolved_base_model,
            imgsz=resolved.imgsz,
            epochs=resolved.epochs,
            conf=conf,
            patience=resolved.patience,
            freeze=resolved.freeze,
            workers=resolved.workers,
        ),
    )
    updated_state = apply_training_state_update(
        train_state=train_state,
        selection=selection,
        trained_model_path=str(best),
        mode=resolved.mode,
    )
    save_train_state(train_state_path, updated_state)
    print(f"Trained model: {best}")
    return best


def _parse_first_positive_box(label_path: Path, image_width: int, image_height: int) -> tuple[int, int, int, int] | None:
    """Parse first YOLO box from label file; return None for negatives/empty."""
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return None

    first = text.splitlines()[0].strip()
    class_id, xc, yc, bw, bh = first.split()
    _ = class_id

    xc_f = float(xc) * image_width
    yc_f = float(yc) * image_height
    bw_f = float(bw) * image_width
    bh_f = float(bh) * image_height

    x1 = int(round(xc_f - bw_f / 2.0))
    y1 = int(round(yc_f - bh_f / 2.0))
    x2 = int(round(xc_f + bw_f / 2.0))
    y2 = int(round(yc_f + bh_f / 2.0))

    x1 = max(0, min(image_width - 1, x1))
    y1 = max(0, min(image_height - 1, y1))
    x2 = max(0, min(image_width - 1, x2))
    y2 = max(0, min(image_height - 1, y2))

    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _crop_with_padding(image: np.ndarray, box: tuple[int, int, int, int], padding: float) -> np.ndarray:
    x1, y1, x2, y2 = box
    height, width = image.shape[:2]

    pad_x = int((x2 - x1) * padding)
    pad_y = int((y2 - y1) * padding)

    x1p = max(0, x1 - pad_x)
    y1p = max(0, y1 - pad_y)
    x2p = min(width - 1, x2 + pad_x)
    y2p = min(height - 1, y2 + pad_y)

    crop = image[y1p : y2p + 1, x1p : x2p + 1]
    if crop.size == 0:
        return image[y1:y2, x1:x2]
    return crop


def build_sku_profile(
    sku: str,
    max_embeddings: int,
    threshold: float,
    padding: float,
    backbone: str,
) -> Path:
    """Build embedding profile from labeled enrollment frames."""
    frames_dir = Path(f"data/derived/frames/{sku}/frames")
    labels_dir = Path(f"data/derived/labels/{sku}/labels")
    crops_dir = Path(f"data/derived/crops/{sku}")
    profile_dir = Path(f"data/profiles/{sku}")

    crops_dir.mkdir(parents=True, exist_ok=True)

    embedder = build_embedder(backbone)

    embeddings: list[np.ndarray] = []

    labeled_images = collect_labeled_images(frames_dir=frames_dir, labels_dir=labels_dir)
    positive_images: list[Path] = []
    for image_path in labeled_images:
        label_path = labels_dir / f"{image_path.stem}.txt"
        text = label_path.read_text(encoding="utf-8").strip()
        if text:
            positive_images.append(image_path)

    if not positive_images:
        raise RuntimeError(f"No positive labels found for SKU {sku}")

    idxs = np.linspace(0, len(positive_images) - 1, num=min(max_embeddings, len(positive_images)), dtype=int)
    selected_images = [positive_images[i] for i in idxs.tolist()]

    for idx, image_path in enumerate(selected_images, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]

        label_path = labels_dir / f"{image_path.stem}.txt"
        box = _parse_first_positive_box(label_path, width, height)
        if box is None:
            continue

        crop = _crop_with_padding(image, box, padding=padding)
        crop_path = crops_dir / f"{idx:06d}.jpg"
        cv2.imwrite(str(crop_path), crop)

        emb = embedder.encode(crop)
        embeddings.append(emb)

    if not embeddings:
        raise RuntimeError("No embeddings generated; check labels/crops")

    emb_matrix = np.stack(embeddings, axis=0)

    profile = SKUProfile(
        metadata=ProfileMetadata(
            sku_id=sku,
            name=sku,
            threshold=threshold,
            model=ProfileModelInfo(
                backbone=backbone,
                preprocess_version="v1",
                embedding_dim=int(emb_matrix.shape[1]),
            ),
            num_embeddings=int(emb_matrix.shape[0]),
        )
    )

    save_profile(profile_dir=profile_dir, profile=profile, embeddings=emb_matrix)
    print(f"Saved profile to {profile_dir}")
    return profile_dir
