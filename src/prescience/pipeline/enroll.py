"""Enrollment and training pipeline orchestration."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from prescience.datasets.bootstrap_label import BootstrapParams, run_bootstrap_labeling
from prescience.datasets.yolo import TrainConfig, build_yolo_dataset, collect_labeled_images, train_yolo_model
from prescience.ingest.video_to_frames import ExtractParams, extract_frames
from prescience.profiles.io import save_profile
from prescience.profiles.schema import ProfileMetadata, ProfileModelInfo, SKUProfile
from prescience.vision.embeddings import build_embedder


def extract_frames_for_sku(
    video_path: Path,
    sku: str,
    target_frames: int,
    out_root: Path,
    blur_min: float,
    dedupe_max_similarity: float,
) -> dict:
    """Extract enrollment frames for a SKU."""
    params = ExtractParams(
        target_frames=target_frames,
        blur_min=blur_min,
        dedupe_max_similarity=dedupe_max_similarity,
    )
    meta = extract_frames(video_path=video_path, sku=sku, out_root=out_root, params=params)
    print(f"Saved {meta['num_frames_saved']} frames to {meta['frames_dir']}")
    return meta


def run_bootstrap_labeling_for_sku(
    sku: str,
    manual_per_section: int = 2,
    approve_per_section: int = 5,
    overwrite: bool = False,
    allow_negatives: bool = True,
    base_model: str = "yolov8n.pt",
    imgsz: int = 960,
    epochs_stage1: int = 30,
    epochs_stage2: int = 60,
) -> None:
    """Launch two-stage bootstrap labeling workflow for SKU."""
    run_bootstrap_labeling(
        BootstrapParams(
            sku=sku,
            frames_dir=Path(f"data/derived/frames/{sku}/frames"),
            labels_dir=Path(f"data/derived/labels/{sku}/labels"),
            manual_per_section=manual_per_section,
            approve_per_section=approve_per_section,
            overwrite=overwrite,
            allow_negatives=allow_negatives,
            base_model=base_model,
            imgsz=imgsz,
            epochs_stage1=epochs_stage1,
            epochs_stage2=epochs_stage2,
        )
    )


def train_detector_for_sku(
    sku: str,
    version: str,
    epochs: int,
    imgsz: int,
    conf: float,
    base_model: str,
) -> Path:
    """Train detector from all current labeled frames."""
    frames_dir = Path(f"data/derived/frames/{sku}/frames")
    labels_dir = Path(f"data/derived/labels/{sku}/labels")

    labeled = collect_labeled_images(frames_dir=frames_dir, labels_dir=labels_dir)
    if not labeled:
        raise RuntimeError(f"No labeled images found for SKU {sku}")

    dataset_dir = Path(f"data/datasets/yolo/{sku}_{version}")
    data_yaml = build_yolo_dataset(
        labeled_images=labeled,
        labels_dir=labels_dir,
        out_dataset_dir=dataset_dir,
        class_name="product",
    )

    model_dir = Path(f"data/models/yolo/{sku}_{version}")
    best = train_yolo_model(
        data_yaml=data_yaml,
        model_out_dir=model_dir,
        config=TrainConfig(base_model=base_model, imgsz=imgsz, epochs=epochs, conf=conf),
    )
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
