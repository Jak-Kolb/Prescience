"""YOLO dataset creation and model training helpers."""

from __future__ import annotations

import cv2
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
import ultralytics
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
    resume: bool = False
    resume_checkpoint: str | None = None


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


def get_ultralytics_version() -> str:
    """Return installed ultralytics version string."""
    return getattr(ultralytics, "__version__", "unknown")


def _coerce_metric_value(value: object) -> float | int | str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            scalar = item()
            if isinstance(scalar, (int, float)):
                return scalar
        except Exception:
            pass
    return str(value)


def _write_quick_eval_artifacts(
    *,
    model: YOLO,
    data_yaml: Path,
    model_out_dir: Path,
    imgsz: int,
    conf: float,
    device: str,
) -> None:
    """Run lightweight validation + save summary artifacts."""
    eval_dir = model_out_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload: dict[str, object] = {
        "status": "ok",
        "data_yaml": str(data_yaml),
        "imgsz": imgsz,
        "conf": conf,
        "device": device,
        "ultralytics_version": get_ultralytics_version(),
    }

    try:
        metrics = model.val(data=str(data_yaml), imgsz=imgsz, device=device, verbose=False)
        results_dict = getattr(metrics, "results_dict", {}) or {}
        metrics_payload["metrics"] = {str(key): _coerce_metric_value(value) for key, value in results_dict.items()}
    except Exception as exc:
        metrics_payload["status"] = "error"
        metrics_payload["error"] = str(exc)

    # Save one sample prediction image from val split for quick visual regression checks.
    try:
        val_dir = data_yaml.parent / "images" / "val"
        sample = None
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            matches = sorted(val_dir.glob(ext))
            if matches:
                sample = matches[0]
                break
        if sample is not None:
            preds = model.predict(source=str(sample), conf=conf, imgsz=imgsz, verbose=False)
            if preds:
                annotated = preds[0].plot()
                cv2.imwrite(str(eval_dir / "example_pred.jpg"), annotated)
                metrics_payload["example_pred_source"] = str(sample)
    except Exception as exc:
        metrics_payload["example_pred_error"] = str(exc)

    (eval_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    summary_lines = [
        "# Quick Eval Summary",
        "",
        f"- status: `{metrics_payload.get('status')}`",
        f"- data: `{metrics_payload.get('data_yaml')}`",
        f"- imgsz: `{metrics_payload.get('imgsz')}`",
        f"- conf: `{metrics_payload.get('conf')}`",
        f"- device: `{metrics_payload.get('device')}`",
        f"- ultralytics: `{metrics_payload.get('ultralytics_version')}`",
    ]
    metrics = metrics_payload.get("metrics")
    if isinstance(metrics, dict) and metrics:
        summary_lines.extend(["", "## Metrics"])
        for key in sorted(metrics.keys()):
            summary_lines.append(f"- `{key}`: `{metrics[key]}`")
    if "error" in metrics_payload:
        summary_lines.extend(["", "## Eval Error", f"- `{metrics_payload['error']}`"])
    if "example_pred_source" in metrics_payload:
        summary_lines.extend(["", "## Example Prediction", f"- source: `{metrics_payload['example_pred_source']}`"])
        summary_lines.append(f"- output: `{(eval_dir / 'example_pred.jpg')}`")

    (eval_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def train_yolo_model(
    data_yaml: Path,
    model_out_dir: Path,
    config: TrainConfig,
) -> Path:
    """Train YOLO and return stable best.pt path."""
    model_out_dir.mkdir(parents=True, exist_ok=True)

    device = choose_training_device()
    resume_checkpoint = Path(config.resume_checkpoint) if config.resume_checkpoint else None
    can_resume = bool(config.resume and resume_checkpoint and resume_checkpoint.exists())
    if can_resume:
        model = YOLO(str(resume_checkpoint))
        model.train(resume=True)
    else:
        model = YOLO(config.base_model)
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
        "resume": config.resume,
        "resume_checkpoint": config.resume_checkpoint,
        "device": device,
        "ultralytics_version": get_ultralytics_version(),
        "ultralytics_save_dir": str(save_dir),
        "best_path": str(best_dst),
    }
    with (model_out_dir / "train_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    _write_quick_eval_artifacts(
        model=model,
        data_yaml=data_yaml,
        model_out_dir=model_out_dir,
        imgsz=config.imgsz,
        conf=config.conf,
        device=device,
    )

    return best_dst
