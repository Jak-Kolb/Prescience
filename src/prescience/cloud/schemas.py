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
