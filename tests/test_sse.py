from __future__ import annotations

import asyncio

from prescience.cloud.stream import SSEBroadcaster, encode_sse


def test_encode_sse_contains_retry_and_data() -> None:
    message = encode_sse({"hello": "world"}, retry_ms=1500)
    assert "retry: 1500" in message
    assert 'data: {"hello": "world"}' in message


def test_sse_broadcaster_publish_subscribe() -> None:
    async def scenario() -> None:
        bus = SSEBroadcaster()
        queue = await bus.subscribe()
        await bus.publish({"x": 1})
        item = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert item == {"x": 1}
        await bus.unsubscribe(queue)

    asyncio.run(scenario())
