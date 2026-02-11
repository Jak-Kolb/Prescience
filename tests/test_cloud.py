from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from prescience.cloud.store import CloudStore
from prescience.events.schemas import CountEvent, HeartbeatEvent, utc_now


def _count_event(seq: int, device_id: str = "dev-1", line_id: str = "line-1", run_id: str | None = None, total: int = 1):
    now = utc_now()
    return CountEvent(
        seq=seq,
        timestamp=now,
        frame_ts=now,
        line_id=line_id,
        device_id=device_id,
        run_id=run_id,
        sku_id="sku-a",
        confidence=0.9,
        track_id=seq,
        count_delta=1,
        counts_total_overall=total,
        counts_total_by_sku={"sku-a": total},
    )


def test_auto_run_creation_and_live_aggregation(tmp_path: Path) -> None:
    store = CloudStore(db_path=tmp_path / "cloud.db", heartbeat_timeout_seconds=90, pairing_required=False)

    r1 = store.ingest_event(_count_event(seq=1, total=1))
    assert r1.status == "inserted"
    assert r1.run_id.startswith("auto-dev-1-")

    r2 = store.ingest_event(_count_event(seq=2, total=2))
    assert r2.status == "inserted"
    assert r2.run_id == r1.run_id

    live = store.get_line_live("line-1")
    assert live["totals"]["overall"] == 2
    assert live["totals"]["by_sku"]["sku-a"] == 2


def test_sequence_gap_and_out_of_order_handling(tmp_path: Path) -> None:
    store = CloudStore(db_path=tmp_path / "cloud.db", heartbeat_timeout_seconds=90, pairing_required=False)

    first = store.ingest_event(_count_event(seq=1, total=1))
    run_id = first.run_id

    third = _count_event(seq=3, run_id=run_id, total=2)
    r3 = store.ingest_event(third)
    assert r3.status == "inserted"

    second = _count_event(seq=2, run_id=run_id, total=2)
    r2 = store.ingest_event(second)
    assert r2.status == "ignored"
    assert r2.ignored_reason == "duplicate_or_out_of_order_seq"

    summary = store.get_run_summary(run_id)
    assert summary["overall"] == 2
    assert summary["alerts"] >= 1


def test_auto_close_run_after_heartbeat_timeout(tmp_path: Path) -> None:
    store = CloudStore(db_path=tmp_path / "cloud.db", heartbeat_timeout_seconds=1, pairing_required=False)

    old_ts = utc_now() - timedelta(seconds=5)
    old_event = CountEvent(
        seq=1,
        timestamp=old_ts,
        frame_ts=old_ts,
        line_id="line-1",
        device_id="dev-1",
        run_id=None,
        sku_id="sku-a",
        confidence=0.95,
        track_id=1,
        count_delta=1,
        counts_total_overall=1,
        counts_total_by_sku={"sku-a": 1},
    )
    res = store.ingest_event(old_event)

    _ = store.get_line_live("line-1")
    runs = store.list_runs()
    run = next(r for r in runs if r["run_id"] == res.run_id)

    assert run["ended_at"] is not None


def test_pairing_required_blocks_unpaired_device(tmp_path: Path) -> None:
    store = CloudStore(db_path=tmp_path / "cloud.db", heartbeat_timeout_seconds=90, pairing_required=True)

    blocked = store.ingest_event(_count_event(seq=1))
    assert blocked.status == "ignored"
    assert blocked.ignored_reason == "device_not_paired"

    code = store.create_pair_code(line_id="line-1", ttl_seconds=300)["code"]
    pair = store.pair_device(device_id="dev-1", code=code)
    assert pair["paired"] is True

    accepted = store.ingest_event(_count_event(seq=1, total=1))
    assert accepted.status == "inserted"


def test_device_config_returns_skus(tmp_path: Path) -> None:
    store = CloudStore(db_path=tmp_path / "cloud.db", heartbeat_timeout_seconds=90, pairing_required=False)

    store.upsert_sku(sku_id="sku-a", name="SKU A", profile_path="data/profiles/sku-a", threshold=0.72, metadata={"v": 1})
    cfg = store.get_device_config("device-10")

    assert cfg["device_id"] == "device-10"
    assert cfg["active_skus"][0]["sku_id"] == "sku-a"
