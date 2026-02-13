"""Project configuration models and YAML loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class CameraSettings(BaseModel):
    source: str | int = 0


class LineSettings(BaseModel):
    line_id: str = "line-1"


class ZoneSettings(BaseModel):
    polygon: list[tuple[int, int]] = Field(
        default_factory=lambda: [(120, 120), (520, 120), (520, 420), (120, 420)]
    )
    direction: str = "left_to_right"


class CountingSettings(BaseModel):
    min_separation_seconds: float = 0.4
    track_cooldown_seconds: float = 1.0
    max_track_idle_frames: int = 30


class TrackerSettings(BaseModel):
    conf: float = 0.01
    classes: list[int] = Field(default_factory=list)
    tracker_cfg: str = "configs/bytetrack_low_conf.yaml"


class QualitySettings(BaseModel):
    min_brightness: float = 30.0
    min_blur: float = 10.0
    alert_cooldown_seconds: float = 20.0


class EventSettings(BaseModel):
    endpoint: str = "http://127.0.0.1:8000/events"
    local_jsonl_path: str = "data/runs/events.jsonl"
    heartbeat_interval_seconds: float = 5.0
    request_timeout_seconds: float = 2.0


class ModelSettings(BaseModel):
    path: str = "yolov8n.pt"


class EmbeddingSettings(BaseModel):
    backbone: str = "resnet18"
    threshold: float = 0.72
    preprocess_version: str = "v1"


class ProfileSettings(BaseModel):
    root: str = "data/profiles"


class PairingSettings(BaseModel):
    required: bool = False


class CloudSettings(BaseModel):
    db_path: str = "data/cloud/prescience.db"
    pairing: PairingSettings = Field(default_factory=PairingSettings)
    heartbeat_timeout_seconds: int = 90
    sse_retry_ms: int = 1500


class OnboardingVLMSettings(BaseModel):
    enabled: bool = True
    model: str = "gemini-3-pro-preview"
    api_key_env: str = "GEMINI_API_KEY"
    batch_size: int = 24
    seed_target_count: int = 24
    seed_required_reviews: int = 3
    initial_approval_target_count: int = 24
    append_approval_target_count: int = 30
    max_boxes_seed: int = 4
    max_boxes_approval: int = 8
    trust_required_reviews: int = 3
    initial_auto_approval_count: int = 24
    append_auto_approval_count: int = 30
    gemini_retry_attempts: int = 3
    gemini_retry_backoff_seconds: float = 1.5
    failure_mode: str = "manual_prompt"
    auto_after_trust: bool = True
    detector_conf_for_auto_label: float = 0.02


class AppSettings(BaseModel):
    camera: CameraSettings = Field(default_factory=CameraSettings)
    line: LineSettings = Field(default_factory=LineSettings)
    zone: ZoneSettings = Field(default_factory=ZoneSettings)
    counting: CountingSettings = Field(default_factory=CountingSettings)
    tracker: TrackerSettings = Field(default_factory=TrackerSettings)
    quality: QualitySettings = Field(default_factory=QualitySettings)
    events: EventSettings = Field(default_factory=EventSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    profiles: ProfileSettings = Field(default_factory=ProfileSettings)
    cloud: CloudSettings = Field(default_factory=CloudSettings)
    onboarding_vlm: OnboardingVLMSettings = Field(default_factory=OnboardingVLMSettings)


DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


def parse_source(source: str | int) -> str | int:
    """Convert numeric string camera ids to int for OpenCV."""
    if isinstance(source, int):
        return source
    normalized = source.strip()
    return int(normalized) if normalized.isdigit() else normalized


def load_settings(config_path: str | Path | None = None) -> AppSettings:
    """Load YAML configuration into typed app settings."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if not path.exists():
        return AppSettings()

    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    return AppSettings.model_validate(raw)
