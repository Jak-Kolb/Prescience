from __future__ import annotations

from pydantic import TypeAdapter, ValidationError

from prescience.events.schemas import AnyEvent, CountEvent, HeartbeatEvent, utc_now


def test_count_event_schema_valid() -> None:
    event = CountEvent(
        seq=1,
        timestamp=utc_now(),
        frame_ts=utc_now(),
        line_id="line-1",
        device_id="dev-1",
        run_id="run-1",
        sku_id="sku-a",
        confidence=0.92,
        track_id=12,
        count_delta=1,
        counts_total_overall=10,
        counts_total_by_sku={"sku-a": 10},
    )

    payload = event.model_dump(mode="json")
    assert payload["event_type"] == "COUNT"
    assert payload["seq"] == 1
    assert payload["counts_total_overall"] == 10


def test_invalid_sequence_rejected() -> None:
    try:
        CountEvent(
            seq=0,
            timestamp=utc_now(),
            frame_ts=utc_now(),
            line_id="line-1",
            device_id="dev-1",
            run_id="run-1",
            sku_id="sku-a",
            confidence=0.5,
            track_id=1,
            count_delta=1,
            counts_total_overall=1,
            counts_total_by_sku={"sku-a": 1},
        )
    except ValidationError:
        return

    raise AssertionError("Expected ValidationError for seq=0")


def test_any_event_union_parsing() -> None:
    adapter = TypeAdapter(AnyEvent)
    event = adapter.validate_python(
        {
            "event_type": "HEARTBEAT",
            "seq": 3,
            "timestamp": utc_now().isoformat(),
            "frame_ts": utc_now().isoformat(),
            "line_id": "line-1",
            "device_id": "dev-1",
            "run_id": "run-1",
            "fps": 27.5,
            "uptime_s": 21.0,
            "brightness": 45.0,
            "blur_score": 12.0,
            "last_count_ts": utc_now().isoformat(),
        }
    )

    assert isinstance(event, HeartbeatEvent)
    assert event.seq == 3
