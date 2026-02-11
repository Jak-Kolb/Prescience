"""Train-state persistence and dataset selection helpers."""

from __future__ import annotations

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
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class TrainSelection:
    selected_names: list[str]
    core_names: list[str]
    new_names: list[str]
    all_names: list[str]


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


def _stable_core_names(all_names: list[str], previous_core: list[str], core_size: int) -> list[str]:
    if core_size <= 0:
        return []
    if len(all_names) <= core_size:
        return all_names.copy()

    all_set = set(all_names)
    kept = [name for name in previous_core if name in all_set][:core_size]
    sampled = _evenly_sample_names(all_names, core_size)
    out = kept.copy()
    seen = set(out)
    for name in sampled:
        if name in seen:
            continue
        out.append(name)
        seen.add(name)
        if len(out) >= core_size:
            break
    if len(out) < core_size:
        for name in all_names:
            if name in seen:
                continue
            out.append(name)
            seen.add(name)
            if len(out) >= core_size:
                break
    return out[:core_size]


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
) -> TrainState:
    """Update state after successful training."""
    train_state.core_images = selection.core_names
    train_state.trained_snapshot_images = selection.all_names
    train_state.last_trained_model = trained_model_path
    train_state.last_mode = mode
    train_state.updated_at = datetime.now(timezone.utc)
    return train_state

