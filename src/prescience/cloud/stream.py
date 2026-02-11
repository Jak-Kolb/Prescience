"""In-memory SSE broadcaster for single-process MVP."""

from __future__ import annotations

import asyncio
import json


def encode_sse(data: dict, retry_ms: int | None = None) -> str:
    """Encode payload as SSE frame."""
    lines = []
    if retry_ms is not None:
        lines.append(f"retry: {retry_ms}")
    lines.append(f"data: {json.dumps(data)}")
    lines.append("")
    return "\n".join(lines)


class SSEBroadcaster:
    """Simple publish/subscribe queue fan-out."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[dict]] = set()
        self._lock = asyncio.Lock()

    async def subscribe(self) -> asyncio.Queue[dict]:
        queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=100)
        async with self._lock:
            self._subscribers.add(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue[dict]) -> None:
        async with self._lock:
            self._subscribers.discard(queue)

    async def publish(self, message: dict) -> None:
        async with self._lock:
            subscribers = list(self._subscribers)

        for queue in subscribers:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                continue
