"""Training mode presets and resolver helpers."""

from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Literal


TrainingMode = Literal["quick", "milestone", "full"]
DatasetScope = Literal["core_new", "all"]


@dataclass(frozen=True)
class ModePreset:
    imgsz: int
    epochs_stage1: int
    epochs_stage2: int
    epochs_detector: int
    patience: int
    freeze: int
    dataset_scope: DatasetScope
    core_size: int


MODE_PRESETS: dict[TrainingMode, ModePreset] = {
    "quick": ModePreset(
        imgsz=640,
        epochs_stage1=8,
        epochs_stage2=12,
        epochs_detector=12,
        patience=4,
        freeze=10,
        dataset_scope="core_new",
        core_size=48,
    ),
    "milestone": ModePreset(
        imgsz=768,
        epochs_stage1=10,
        epochs_stage2=25,
        epochs_detector=30,
        patience=6,
        freeze=0,
        dataset_scope="all",
        core_size=48,
    ),
    "full": ModePreset(
        imgsz=960,
        epochs_stage1=15,
        epochs_stage2=45,
        epochs_detector=60,
        patience=10,
        freeze=0,
        dataset_scope="all",
        core_size=48,
    ),
}


def _validate_mode(value: str) -> TrainingMode:
    if value not in MODE_PRESETS:
        raise ValueError(f"Invalid mode: {value}. Expected one of {tuple(MODE_PRESETS.keys())}")
    return value  # type: ignore[return-value]


def _validate_dataset_scope(value: str) -> DatasetScope:
    if value not in {"core_new", "all"}:
        raise ValueError(f"Invalid dataset_scope: {value}. Expected one of ('core_new', 'all')")
    return value  # type: ignore[return-value]


def default_workers_for_mode(mode: TrainingMode) -> int:
    """Return conservative worker defaults by mode/platform."""
    is_macos = sys.platform == "darwin"
    if mode == "quick":
        return 0 if is_macos else 2
    return 2 if is_macos else 4


@dataclass(frozen=True)
class OnboardingTrainingConfig:
    mode: TrainingMode
    dataset_scope: DatasetScope
    core_size: int
    imgsz: int
    epochs_stage1: int
    epochs_stage2: int
    patience: int
    freeze: int
    workers: int


@dataclass(frozen=True)
class DetectorTrainingConfig:
    mode: TrainingMode
    dataset_scope: DatasetScope
    core_size: int
    imgsz: int
    epochs: int
    patience: int
    freeze: int
    workers: int


def resolve_onboarding_training_config(
    *,
    mode: str,
    dataset_scope: str | None,
    core_size: int | None,
    imgsz: int | None,
    epochs_stage1: int | None,
    epochs_stage2: int | None,
    patience: int | None,
    freeze: int | None,
    workers: int | None,
) -> OnboardingTrainingConfig:
    """Resolve onboarding training config from mode preset + optional overrides."""
    resolved_mode = _validate_mode(mode)
    preset = MODE_PRESETS[resolved_mode]
    resolved_scope = _validate_dataset_scope(dataset_scope) if dataset_scope is not None else preset.dataset_scope

    return OnboardingTrainingConfig(
        mode=resolved_mode,
        dataset_scope=resolved_scope,
        core_size=int(core_size if core_size is not None else preset.core_size),
        imgsz=int(imgsz if imgsz is not None else preset.imgsz),
        epochs_stage1=int(epochs_stage1 if epochs_stage1 is not None else preset.epochs_stage1),
        epochs_stage2=int(epochs_stage2 if epochs_stage2 is not None else preset.epochs_stage2),
        patience=int(patience if patience is not None else preset.patience),
        freeze=int(freeze if freeze is not None else preset.freeze),
        workers=int(workers if workers is not None else default_workers_for_mode(resolved_mode)),
    )


def resolve_detector_training_config(
    *,
    mode: str,
    dataset_scope: str | None,
    core_size: int | None,
    imgsz: int | None,
    epochs: int | None,
    patience: int | None,
    freeze: int | None,
    workers: int | None,
) -> DetectorTrainingConfig:
    """Resolve detector training config from mode preset + optional overrides."""
    resolved_mode = _validate_mode(mode)
    preset = MODE_PRESETS[resolved_mode]
    resolved_scope = _validate_dataset_scope(dataset_scope) if dataset_scope is not None else preset.dataset_scope

    return DetectorTrainingConfig(
        mode=resolved_mode,
        dataset_scope=resolved_scope,
        core_size=int(core_size if core_size is not None else preset.core_size),
        imgsz=int(imgsz if imgsz is not None else preset.imgsz),
        epochs=int(epochs if epochs is not None else preset.epochs_detector),
        patience=int(patience if patience is not None else preset.patience),
        freeze=int(freeze if freeze is not None else preset.freeze),
        workers=int(workers if workers is not None else default_workers_for_mode(resolved_mode)),
    )

