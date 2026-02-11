"""Event schemas shared by edge runtime and cloud ingest."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class EventBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: Literal["COUNT", "HEARTBEAT", "ALERT"]
    seq: int = Field(ge=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    frame_ts: datetime | None = None
    line_id: str
    device_id: str
    run_id: str | None = None


class CountEvent(EventBase):
    event_type: Literal["COUNT"] = "COUNT"
    sku_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    track_id: int | None = None
    count_delta: int = 1
    counts_total_overall: int = Field(ge=0)
    counts_total_by_sku: dict[str, int] = Field(default_factory=dict)


class HeartbeatEvent(EventBase):
    event_type: Literal["HEARTBEAT"] = "HEARTBEAT"
    fps: float = Field(ge=0.0)
    uptime_s: float = Field(ge=0.0)
    brightness: float
    blur_score: float
    last_count_ts: datetime | None = None


class AlertEvent(EventBase):
    event_type: Literal["ALERT"] = "ALERT"
    code: str
    severity: Literal["info", "warning", "critical"] = "warning"
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


AnyEvent = Annotated[CountEvent | HeartbeatEvent | AlertEvent, Field(discriminator="event_type")]


def utc_now() -> datetime:
    """UTC now helper for consistent timestamps."""
    return datetime.now(timezone.utc)
