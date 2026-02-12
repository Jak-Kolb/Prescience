"""Cloud API request models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RunStartRequest(BaseModel):
    line_id: str
    device_id: str
    run_id: str | None = None


class PairDeviceRequest(BaseModel):
    code: str


class PairCodeCreateRequest(BaseModel):
    line_id: str
    ttl_seconds: int = Field(default=600, ge=30, le=86400)


class SKUUpsertRequest(BaseModel):
    sku_id: str
    name: str
    profile_path: str | None = None
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict | None = None


class OnboardingBoxRequest(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class OnboardingLabelRequest(BaseModel):
    frame_name: str
    status: str = Field(pattern="^(positive|negative|skipped)$")
    boxes: list[OnboardingBoxRequest] = Field(default_factory=list)
    stage: str = Field(default="seed", pattern="^(seed|approval)$")


class ZoneConfigRequest(BaseModel):
    polygon: list[tuple[int, int]]
    direction: str


class TrackingStartRequest(BaseModel):
    source: str = "0"
    line_id: str = "line-1"
    device_id: str = "device-1"
    sku_id: str
    model_path: str | None = None
    conf: float | None = Field(default=None, ge=0.0, le=1.0)
    direction: str | None = None
    run_id: str | None = None
    event_endpoint: str | None = None
