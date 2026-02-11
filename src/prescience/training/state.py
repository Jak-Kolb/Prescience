"""Train-state persistence and dataset selection helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from prescience.training.strategy import DatasetScope, TrainingMode


class TrainState(BaseModel):
    """State for fast incremental training dataset selection."""

    model_config = ConfigDict(extra="forbid")

    core_images: list[str] = Field(default_factory=list)
    trained_snapshot_images: list[str] = Field(default_factory=list)
    last_trained_model: str | None = None
    last_mode: TrainingMode | None = None
    last_version_tag: str | None = None
    dataset_hash: str | None = None
    last_imgsz: int | None = None
    last_base_model_path: str | None = None
    last_ultralytics_version: str | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class TrainSelection:
    selected_names: list[str]
    core_names: list[str]
    new_names: list[str]
    all_names: list[str]


@dataclass(frozen=True)
class ResumeDecision:
    allowed: bool
    reason: str


def train_state_path_for_labels(labels_dir: str | Path) -> Path:
    """Resolve train-state path for SKU labels directory."""
    return Path(labels_dir).parent / "train_state.json"


def load_or_create_train_state(path: str | Path) -> TrainState:
    """Load train-state or create defaults when missing/corrupt."""
    target = Path(path)
    if target.exists():
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
            return TrainState.model_validate(raw)
        except (OSError, json.JSONDecodeError, ValueError):
            pass
    return TrainState()


def save_train_state(path: str | Path, state: TrainState) -> None:
    """Persist train-state to disk."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    state.updated_at = datetime.now(timezone.utc)
    target.write_text(json.dumps(state.model_dump(mode="json"), indent=2), encoding="utf-8")


def _evenly_sample_names(names: list[str], k: int) -> list[str]:
    if k <= 0 or not names:
        return []
    if k >= len(names):
        return names.copy()
    idxs = np.linspace(0, len(names) - 1, num=k, dtype=int).tolist()
    out: list[str] = []
    seen: set[str] = set()
    for idx in idxs:
        name = names[idx]
        if name in seen:
            continue
        out.append(name)
        seen.add(name)
    if len(out) < k:
        for name in names:
            if name in seen:
                continue
            out.append(name)
            seen.add(name)
            if len(out) >= k:
                break
    return out


def _split_names_into_bins(names: list[str], bins: int = 6) -> list[list[str]]:
    if bins <= 0:
        raise ValueError("bins must be > 0")
    n = len(names)
    out: list[list[str]] = []
    start = 0
    for idx in range(bins):
        size = n // bins + (1 if idx < (n % bins) else 0)
        out.append(names[start : start + size])
        start += size
    return out


def _stable_core_names(all_names: list[str], previous_core: list[str], core_size: int) -> list[str]:
    if core_size <= 0:
        return []
    if len(all_names) <= core_size:
        return all_names.copy()

    sections = _split_names_into_bins(all_names, bins=6)
    base = core_size // 6
    rem = core_size % 6

    out: list[str] = []
    seen: set[str] = set()
    for idx, section in enumerate(sections):
        target = base + (1 if idx < rem else 0)
        picks = _evenly_sample_names(section, target)
        for name in picks:
            if name in seen:
                continue
            out.append(name)
            seen.add(name)

    if len(out) < core_size:
        for name in all_names:
            if name in seen:
                continue
            out.append(name)
            seen.add(name)
            if len(out) >= core_size:
                break

    return sorted(out[:core_size])


def select_training_names(
    *,
    labeled_names: list[str],
    scope: DatasetScope,
    core_size: int,
    train_state: TrainState,
) -> TrainSelection:
    """Select train set names according to dataset scope and train-state."""
    all_names = sorted(set(labeled_names))
    core_names = _stable_core_names(all_names, train_state.core_images, core_size)
    snapshot_set = set(train_state.trained_snapshot_images)
    new_names = [name for name in all_names if name not in snapshot_set]

    if scope == "all":
        selected = all_names
    elif new_names:
        selected = sorted(set(core_names).union(new_names))
    else:
        selected = core_names

    return TrainSelection(
        selected_names=selected,
        core_names=core_names,
        new_names=new_names,
        all_names=all_names,
    )


def apply_training_state_update(
    *,
    train_state: TrainState,
    selection: TrainSelection,
    trained_model_path: str | None,
    mode: TrainingMode,
    version_tag: str | None,
    dataset_hash: str | None,
    imgsz: int | None,
    base_model_path: str | None,
    ultralytics_version: str | None,
) -> TrainState:
    """Update state after successful training."""
    train_state.core_images = selection.core_names
    train_state.trained_snapshot_images = selection.all_names
    train_state.last_trained_model = trained_model_path
    train_state.last_mode = mode
    train_state.last_version_tag = version_tag
    train_state.dataset_hash = dataset_hash
    train_state.last_imgsz = imgsz
    train_state.last_base_model_path = base_model_path
    train_state.last_ultralytics_version = ultralytics_version
    train_state.updated_at = datetime.now(timezone.utc)
    return train_state


def compute_dataset_hash(*, selected_images: list[Path], labels_dir: str | Path) -> str:
    """Hash selected dataset image/label signatures for safe resume checks."""
    labels_root = Path(labels_dir)
    payload_items: list[dict[str, object]] = []
    for image_path in sorted(selected_images, key=lambda path: path.name):
        image_stat = image_path.stat()
        label_path = labels_root / f"{image_path.stem}.txt"
        label_exists = label_path.exists()
        label_size = None
        label_mtime_ns = None
        if label_exists:
            label_stat = label_path.stat()
            label_size = int(label_stat.st_size)
            label_mtime_ns = int(label_stat.st_mtime_ns)
        payload_items.append(
            {
                "image_name": image_path.name,
                "image_size": int(image_stat.st_size),
                "image_mtime_ns": int(image_stat.st_mtime_ns),
                "label_exists": label_exists,
                "label_size": label_size,
                "label_mtime_ns": label_mtime_ns,
            }
        )

    payload = json.dumps(payload_items, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def can_resume_from_state(
    *,
    train_state: TrainState,
    requested_dataset_hash: str,
    requested_mode: TrainingMode,
    requested_imgsz: int,
    requested_base_model_path: str,
    requested_ultralytics_version: str,
    requested_version_tag: str | None,
) -> ResumeDecision:
    """Check whether resume is safe for requested training context."""
    if train_state.dataset_hash != requested_dataset_hash:
        return ResumeDecision(allowed=False, reason="dataset_hash_mismatch")
    if train_state.last_mode != requested_mode:
        return ResumeDecision(allowed=False, reason="mode_mismatch")
    if train_state.last_imgsz != requested_imgsz:
        return ResumeDecision(allowed=False, reason="imgsz_mismatch")
    if train_state.last_base_model_path != requested_base_model_path:
        return ResumeDecision(allowed=False, reason="base_model_mismatch")
    if train_state.last_ultralytics_version != requested_ultralytics_version:
        return ResumeDecision(allowed=False, reason="ultralytics_version_mismatch")
    if requested_version_tag is not None and train_state.last_version_tag != requested_version_tag:
        return ResumeDecision(allowed=False, reason="version_tag_mismatch")
    return ResumeDecision(allowed=True, reason="ok")
