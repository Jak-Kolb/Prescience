"""YOLO dataset creation and model training helpers."""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from ultralytics import YOLO


@dataclass(frozen=True)
class TrainConfig:
    base_model: str = "yolov8n.pt"
    imgsz: int = 960
    epochs: int = 60
    conf: float = 0.35
    patience: int | None = None
    freeze: int | None = None
    workers: int | None = None


def label_path_for_image(labels_dir: Path, image_path: Path) -> Path:
    """Resolve corresponding YOLO label path for an image."""
    return labels_dir / f"{image_path.stem}.txt"


def list_images(frames_dir: Path) -> list[Path]:
    """List frame images in deterministic order."""
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in exts)


def collect_labeled_images(frames_dir: Path, labels_dir: Path) -> list[Path]:
    """Collect images that have a label txt (including empty negative labels)."""
    images = list_images(frames_dir)
    return [img for img in images if label_path_for_image(labels_dir, img).exists()]


def build_yolo_dataset(
    labeled_images: list[Path],
    labels_dir: Path,
    out_dataset_dir: Path,
    class_name: str = "product",
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Path:
    """Build train/val YOLO directory structure and data.yaml."""
    if not labeled_images:
        raise ValueError("No labeled images found for dataset build")

    out_dataset_dir.mkdir(parents=True, exist_ok=True)
    for rel in ["images/train", "images/val", "labels/train", "labels/val"]:
        (out_dataset_dir / rel).mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    images = labeled_images.copy()
    rng.shuffle(images)

    split_idx = max(1, int(len(images) * train_ratio))
    split_idx = min(split_idx, len(images) - 1) if len(images) > 1 else 1

    train_images = images[:split_idx]
    val_images = images[split_idx:] if len(images) > 1 else images

    if not val_images:
        val_images = train_images[-1:]

    def copy_pairs(items: list[Path], image_dst: Path, label_dst: Path) -> None:
        for image in items:
            label = label_path_for_image(labels_dir, image)
            if not label.exists():
                continue
            shutil.copy2(image, image_dst / image.name)
            shutil.copy2(label, label_dst / label.name)

    copy_pairs(train_images, out_dataset_dir / "images/train", out_dataset_dir / "labels/train")
    copy_pairs(val_images, out_dataset_dir / "images/val", out_dataset_dir / "labels/val")

    data_yaml = out_dataset_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {out_dataset_dir.as_posix()}",
                "train: images/train",
                "val: images/val",
                "",
                "names:",
                f"  0: {class_name}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    manifest = {
        "dataset_dir": str(out_dataset_dir),
        "num_images": len(images),
        "num_train": len(train_images),
        "num_val": len(val_images),
        "class_name": class_name,
    }
    with (out_dataset_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return data_yaml


def choose_training_device() -> str:
    """Pick best available torch device for training."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_yolo_model(
    data_yaml: Path,
    model_out_dir: Path,
    config: TrainConfig,
) -> Path:
    """Train YOLO and return stable best.pt path."""
    model_out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(config.base_model)
    device = choose_training_device()

    train_kwargs = {
        "data": str(data_yaml),
        "epochs": config.epochs,
        "imgsz": config.imgsz,
        "conf": config.conf,
        "device": device,
        "project": str(model_out_dir),
        "name": "train",
        "exist_ok": True,
        "verbose": False,
        "plots": False,
    }
    if config.patience is not None:
        train_kwargs["patience"] = int(config.patience)
    if config.freeze is not None:
        train_kwargs["freeze"] = int(config.freeze)
    if config.workers is not None:
        train_kwargs["workers"] = int(config.workers)

    model.train(
        **train_kwargs,
    )

    save_dir = Path(model.trainer.save_dir)
    best_src = save_dir / "weights" / "best.pt"
    if not best_src.exists():
        raise RuntimeError(f"Training finished but best.pt not found at: {best_src}")

    best_dst = model_out_dir / "best.pt"
    shutil.copy2(best_src, best_dst)

    meta = {
        "data_yaml": str(data_yaml),
        "base_model": config.base_model,
        "imgsz": config.imgsz,
        "epochs": config.epochs,
        "patience": config.patience,
        "freeze": config.freeze,
        "workers": config.workers,
        "device": device,
        "ultralytics_save_dir": str(save_dir),
        "best_path": str(best_dst),
    }
    with (model_out_dir / "train_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return best_dst
