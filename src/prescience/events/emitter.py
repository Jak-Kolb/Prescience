"""Event sending utilities for edge runtime."""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from prescience.events.schemas import EventBase


class SequenceCounter:
    """Monotonic sequence counter per runtime session."""

    def __init__(self, start: int = 0):
        self._value = start

    def next(self) -> int:
        self._value += 1
        return self._value


class EventEmitter:
    """POST events to cloud endpoint with local JSONL fallback on failures."""

    def __init__(
        self,
        endpoint: str | None,
        fallback_jsonl_path: str | Path,
        timeout_seconds: float = 2.0,
    ) -> None:
        self.endpoint = endpoint
        self.fallback_jsonl_path = Path(fallback_jsonl_path)
        self.timeout_seconds = timeout_seconds
        self._client = httpx.Client(timeout=self.timeout_seconds)

    def close(self) -> None:
        self._client.close()

    def emit(self, event: EventBase) -> bool:
        """Send event. Returns True on successful cloud send."""
        payload = event.model_dump(mode="json")

        if self.endpoint:
            try:
                response = self._client.post(self.endpoint, json=payload)
                response.raise_for_status()
                return True
            except Exception:
                pass

        self._write_fallback(payload)
        return False

    def _write_fallback(self, payload: dict) -> None:
        self.fallback_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.fallback_jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
