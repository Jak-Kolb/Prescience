"""YOLO dataset creation and model training helpers."""

from __future__ import annotations

import cv2
import json
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

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


def _extract_key_metric(metrics: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = metrics.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _evaluate_model_summary(
    *,
    model_path: str,
    data_yaml: Path,
    imgsz: int,
    conf: float,
    device: str,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "status": "ok",
        "model_path": model_path,
        "imgsz": imgsz,
        "conf": conf,
        "device": device,
    }
    try:
        model = YOLO(model_path)
        metrics = model.val(data=str(data_yaml), imgsz=imgsz, device=device, verbose=False)
        results_dict = getattr(metrics, "results_dict", {}) or {}
        summary["metrics"] = {str(key): _coerce_metric_value(value) for key, value in results_dict.items()}
    except Exception as exc:  # noqa: BLE001
        summary["status"] = "error"
        summary["error"] = str(exc)
        summary["metrics"] = {}
    return summary


def _model_score(summary: dict[str, Any]) -> float | None:
    metrics = summary.get("metrics", {})
    if not isinstance(metrics, dict):
        return None
    score = _extract_key_metric(
        metrics,
        keys=(
            "metrics/mAP50-95(B)",
            "metrics/mAP50-95(M)",
            "metrics/mAP50-95(P)",
            "metrics/mAP50(B)",
            "metrics/mAP50(M)",
            "metrics/mAP50(P)",
        ),
    )
    return score


def _stable_model_tag(model_out_dir: Path) -> tuple[str, int] | None:
    match = re.match(r"^(.+)_v(\d+)$", model_out_dir.name)
    if match is None:
        return None
    return match.group(1), int(match.group(2))


def _prune_older_model_versions(model_out_dir: Path) -> list[str]:
    tag = _stable_model_tag(model_out_dir)
    if tag is None:
        return []
    sku, keep_version = tag
    model_root = model_out_dir.parent
    pattern = re.compile(rf"^{re.escape(sku)}_v(\d+)$")
    deleted: list[str] = []
    for child in sorted(model_root.iterdir()):
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if match is None:
            continue
        version = int(match.group(1))
        if version >= keep_version:
            continue
        if child.resolve() == model_out_dir.resolve():
            continue
        shutil.rmtree(child, ignore_errors=True)
        deleted.append(str(child))
    return deleted


def _write_quick_eval_artifacts(
    *,
    best_model_path: Path,
    old_model_path: str | None,
    data_yaml: Path,
    model_out_dir: Path,
    imgsz: int,
    conf: float,
    device: str,
) -> None:
    """Run comparison eval (old vs new) and persist summary artifacts."""
    eval_dir = model_out_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload: dict[str, object] = {
        "status": "ok",
        "data_yaml": str(data_yaml),
        "imgsz": imgsz,
        "conf": conf,
        "device": device,
        "ultralytics_version": get_ultralytics_version(),
        "old_model": None,
        "new_model": None,
    }

    old_summary: dict[str, Any] | None = None
    if old_model_path and Path(old_model_path).exists():
        print(f"[eval] Old model summary ({old_model_path})")
        old_summary = _evaluate_model_summary(
            model_path=old_model_path,
            data_yaml=data_yaml,
            imgsz=imgsz,
            conf=conf,
            device=device,
        )
        metrics_payload["old_model"] = old_summary

    print(f"[eval] New model summary ({best_model_path})")
    new_summary = _evaluate_model_summary(
        model_path=str(best_model_path),
        data_yaml=data_yaml,
        imgsz=imgsz,
        conf=conf,
        device=device,
    )
    metrics_payload["new_model"] = new_summary

    old_score = _model_score(old_summary) if old_summary is not None else None
    new_score = _model_score(new_summary)
    comparison: dict[str, Any] = {
        "metric": "mAP50-95 (fallback mAP50)",
        "old_score": old_score,
        "new_score": new_score,
        "improved": False,
        "deleted_old_models": [],
    }
    if old_score is not None and new_score is not None and new_score > old_score:
        deleted = _prune_older_model_versions(model_out_dir)
        comparison["improved"] = True
        comparison["deleted_old_models"] = deleted
        if deleted:
            print(f"[eval] New model is better. Deleted old model dirs: {deleted}")
    metrics_payload["comparison"] = comparison

    # Save one sample prediction image and confidence stats from val split.
    try:
        model = YOLO(str(best_model_path))
        val_dir = data_yaml.parent / "images" / "val"
        val_images: list[Path] = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            val_images.extend(sorted(val_dir.glob(ext)))
        max_confidence: float | None = None
        top_conf_sum = 0.0
        top_conf_count = 0
        for sample_image in val_images:
            preds = model.predict(source=str(sample_image), conf=0.001, imgsz=imgsz, verbose=False)
            if not preds:
                continue
            boxes = preds[0].boxes
            if boxes is None or len(boxes) == 0:
                continue
            confs = boxes.conf.cpu().numpy()
            if confs.size == 0:
                continue
            frame_max = float(confs.max())
            max_confidence = frame_max if max_confidence is None else max(max_confidence, frame_max)
            top_conf_sum += frame_max
            top_conf_count += 1

        metrics_payload["prediction_stats"] = {
            "val_images_scored": len(val_images),
            "detections_scored": top_conf_count,
            "max_confidence": max_confidence,
            "mean_top_confidence": (top_conf_sum / top_conf_count) if top_conf_count else None,
        }

        sample = None
        if val_images:
            sample = val_images[0]
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
        "# Model Comparison Summary",
        "",
        f"- status: `{metrics_payload.get('status')}`",
        f"- data: `{metrics_payload.get('data_yaml')}`",
        f"- imgsz: `{metrics_payload.get('imgsz')}`",
        f"- conf: `{metrics_payload.get('conf')}`",
        f"- device: `{metrics_payload.get('device')}`",
        f"- ultralytics: `{metrics_payload.get('ultralytics_version')}`",
    ]
    prediction_stats = metrics_payload.get("prediction_stats")
    if isinstance(prediction_stats, dict):
        summary_lines.extend(["", "## Prediction Stats"])
        summary_lines.append(f"- val_images_scored: `{prediction_stats.get('val_images_scored')}`")
        summary_lines.append(f"- detections_scored: `{prediction_stats.get('detections_scored')}`")
        summary_lines.append(f"- max_confidence: `{prediction_stats.get('max_confidence')}`")
        summary_lines.append(f"- mean_top_confidence: `{prediction_stats.get('mean_top_confidence')}`")

    old_model = metrics_payload.get("old_model")
    if isinstance(old_model, dict) and old_model:
        summary_lines.extend(["", "## Old Model"])
        summary_lines.append(f"- path: `{old_model.get('model_path')}`")
        old_metrics = old_model.get("metrics")
        if isinstance(old_metrics, dict):
            for key in sorted(old_metrics.keys()):
                summary_lines.append(f"- `{key}`: `{old_metrics[key]}`")

    new_model = metrics_payload.get("new_model")
    if isinstance(new_model, dict) and new_model:
        summary_lines.extend(["", "## New Model"])
        summary_lines.append(f"- path: `{new_model.get('model_path')}`")
        new_metrics = new_model.get("metrics")
        if isinstance(new_metrics, dict):
            for key in sorted(new_metrics.keys()):
                summary_lines.append(f"- `{key}`: `{new_metrics[key]}`")

    compare = metrics_payload.get("comparison")
    if isinstance(compare, dict):
        summary_lines.extend(["", "## Decision"])
        summary_lines.append(f"- improved: `{compare.get('improved')}`")
        summary_lines.append(f"- old_score: `{compare.get('old_score')}`")
        summary_lines.append(f"- new_score: `{compare.get('new_score')}`")
        deleted_dirs = compare.get("deleted_old_models", [])
        if isinstance(deleted_dirs, list) and deleted_dirs:
            summary_lines.append(f"- deleted_old_models: `{deleted_dirs}`")

    if "example_pred_source" in metrics_payload:
        summary_lines.extend(["", "## Example Prediction", f"- source: `{metrics_payload['example_pred_source']}`"])
        summary_lines.append(f"- output: `{(eval_dir / 'example_pred.jpg')}`")

    (eval_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def train_yolo_model(
    data_yaml: Path,
    model_out_dir: Path,
    config: TrainConfig,
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    """Train YOLO and return stable best.pt path."""
    model_out_dir.mkdir(parents=True, exist_ok=True)

    device = choose_training_device()
    resume_checkpoint = Path(config.resume_checkpoint) if config.resume_checkpoint else None
    can_resume = bool(config.resume and resume_checkpoint and resume_checkpoint.exists())
    if can_resume:
        model = YOLO(str(resume_checkpoint))
        if progress_cb is not None:
            progress_cb({"status": "running", "stage": "train", "message": "Resuming training"})
        model.train(resume=True)
    else:
        model = YOLO(config.base_model)
        if progress_cb is not None:
            progress_cb(
                {
                    "status": "running",
                    "stage": "train",
                    "message": "Starting training",
                    "epoch": 0,
                    "total_epochs": int(config.epochs),
                }
            )

        if progress_cb is not None:
            def _emit_epoch_progress(trainer) -> None:
                epoch_idx = int(getattr(trainer, "epoch", -1)) + 1
                total_epochs = int(getattr(getattr(trainer, "args", object()), "epochs", config.epochs))
                progress_cb(
                    {
                        "status": "running",
                        "stage": "train",
                        "epoch": epoch_idx,
                        "total_epochs": total_epochs,
                        "message": f"Epoch {epoch_idx}/{total_epochs}",
                    }
                )

            for event_name in ("on_train_epoch_end", "on_fit_epoch_end"):
                try:
                    model.add_callback(event_name, _emit_epoch_progress)
                except Exception:
                    continue

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

    if progress_cb is not None:
        progress_cb({"status": "running", "stage": "train", "message": "Finalizing model artifacts"})

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
        best_model_path=best_dst,
        old_model_path=config.base_model,
        data_yaml=data_yaml,
        model_out_dir=model_out_dir,
        imgsz=config.imgsz,
        conf=config.conf,
        device=device,
    )
    if progress_cb is not None:
        progress_cb({"status": "succeeded", "stage": "train", "message": "Training complete", "progress": 100.0})

    return best_dst
