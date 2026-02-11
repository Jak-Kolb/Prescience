"""SQLite-backed cloud storage and aggregation logic."""

from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from prescience.cloud.db import connect, init_db
from prescience.events.schemas import AlertEvent, CountEvent, EventBase, HeartbeatEvent, utc_now


@dataclass
class IngestResult:
    status: str
    run_id: str
    line_id: str
    ignored_reason: str | None = None


class CloudStore:
    """Persistence layer for event ingest and dashboard queries."""

    def __init__(self, db_path: str | Path, heartbeat_timeout_seconds: int, pairing_required: bool) -> None:
        self.db_path = Path(db_path)
        self.heartbeat_timeout_seconds = heartbeat_timeout_seconds
        self.pairing_required = pairing_required
        init_db(self.db_path)

    def _conn(self) -> sqlite3.Connection:
        return connect(self.db_path)

    @staticmethod
    def _parse_ts(raw: str | None) -> datetime | None:
        if raw is None:
            return None
        return datetime.fromisoformat(raw)

    @staticmethod
    def _iso_now() -> str:
        return utc_now().isoformat()

    def _ensure_device(self, conn: sqlite3.Connection, device_id: str, line_id: str) -> None:
        conn.execute(
            """
            INSERT INTO devices(device_id, line_id, config_json, paired)
            VALUES (?, ?, ?, 0)
            ON CONFLICT(device_id) DO UPDATE SET
              line_id=COALESCE(excluded.line_id, devices.line_id)
            """,
            (device_id, line_id, "{}"),
        )

    def _is_device_paired(self, conn: sqlite3.Connection, device_id: str) -> bool:
        row = conn.execute("SELECT paired FROM devices WHERE device_id=?", (device_id,)).fetchone()
        if row is None:
            return False
        return bool(row["paired"])

    def _close_stale_runs(self, conn: sqlite3.Connection) -> None:
        now = utc_now()
        rows = conn.execute("SELECT run_id, started_at FROM runs WHERE ended_at IS NULL").fetchall()
        for row in rows:
            run_id = row["run_id"]
            last_row = conn.execute(
                "SELECT MAX(timestamp) AS last_ts FROM events WHERE run_id=?",
                (run_id,),
            ).fetchone()
            last_ts_raw = last_row["last_ts"] if last_row else None
            if last_ts_raw is None:
                last_ts_raw = row["started_at"]

            last_ts = self._parse_ts(last_ts_raw)
            if last_ts is None:
                continue
            if (now - last_ts).total_seconds() > self.heartbeat_timeout_seconds:
                conn.execute(
                    "UPDATE runs SET ended_at=? WHERE run_id=? AND ended_at IS NULL",
                    (now.isoformat(), run_id),
                )

    def _resolve_run(self, conn: sqlite3.Connection, event: EventBase) -> str:
        if event.run_id:
            conn.execute(
                """
                INSERT INTO runs(run_id, line_id, device_id, started_at, source)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO NOTHING
                """,
                (event.run_id, event.line_id, event.device_id, event.timestamp.isoformat(), "edge"),
            )
            return event.run_id

        active = conn.execute(
            """
            SELECT run_id
            FROM runs
            WHERE line_id=? AND device_id=? AND ended_at IS NULL
            ORDER BY started_at DESC
            LIMIT 1
            """,
            (event.line_id, event.device_id),
        ).fetchone()

        if active:
            return str(active["run_id"])

        auto_run_id = f"auto-{event.device_id}-{int(event.timestamp.timestamp())}"
        conn.execute(
            "INSERT INTO runs(run_id, line_id, device_id, started_at, source) VALUES (?, ?, ?, ?, ?)",
            (auto_run_id, event.line_id, event.device_id, event.timestamp.isoformat(), "auto"),
        )
        return auto_run_id

    def _last_seq(self, conn: sqlite3.Connection, device_id: str, run_id: str) -> int | None:
        row = conn.execute(
            "SELECT MAX(seq) AS max_seq FROM events WHERE device_id=? AND run_id=?",
            (device_id, run_id),
        ).fetchone()
        if row is None or row["max_seq"] is None:
            return None
        return int(row["max_seq"])

    def _insert_event(self, conn: sqlite3.Connection, event: EventBase) -> bool:
        sku_id = None
        count_delta = None
        counts_total_overall = None

        if isinstance(event, CountEvent):
            sku_id = event.sku_id
            count_delta = event.count_delta
            counts_total_overall = event.counts_total_overall

        try:
            conn.execute(
                """
                INSERT INTO events(
                    event_id, event_type, seq, timestamp, frame_ts, line_id, device_id, run_id,
                    payload_json, sku_id, count_delta, counts_total_overall
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.event_type,
                    event.seq,
                    event.timestamp.isoformat(),
                    event.frame_ts.isoformat() if event.frame_ts else None,
                    event.line_id,
                    event.device_id,
                    event.run_id,
                    event.model_dump_json(),
                    sku_id,
                    count_delta,
                    counts_total_overall,
                ),
            )
        except sqlite3.IntegrityError:
            return False
        return True

    def ingest_event(self, event: EventBase) -> IngestResult:
        """Store event, handle sequence checks, and auto-manage run lifecycle."""
        conn = self._conn()
        try:
            with conn:
                self._close_stale_runs(conn)
                self._ensure_device(conn, event.device_id, event.line_id)

                if self.pairing_required and not self._is_device_paired(conn, event.device_id):
                    return IngestResult(
                        status="ignored",
                        run_id=event.run_id or "",
                        line_id=event.line_id,
                        ignored_reason="device_not_paired",
                    )

                run_id = self._resolve_run(conn, event)
                if event.run_id != run_id:
                    event = event.model_copy(update={"run_id": run_id})

                last_seq = self._last_seq(conn, event.device_id, run_id)
                if last_seq is not None:
                    if event.seq <= last_seq:
                        return IngestResult(
                            status="ignored",
                            run_id=run_id,
                            line_id=event.line_id,
                            ignored_reason="duplicate_or_out_of_order_seq",
                        )

                    if event.seq > (last_seq + 1):
                        gap_alert = AlertEvent(
                            seq=last_seq + 1,
                            timestamp=utc_now(),
                            frame_ts=event.frame_ts,
                            line_id=event.line_id,
                            device_id=event.device_id,
                            run_id=run_id,
                            code="seq_gap",
                            severity="warning",
                            message="Sequence gap detected",
                            details={"expected": last_seq + 1, "received": event.seq},
                        )
                        self._insert_event(conn, gap_alert)

                inserted = self._insert_event(conn, event)
                status = "inserted" if inserted else "ignored"
                reason = None if inserted else "duplicate_event_id"
                return IngestResult(status=status, run_id=run_id, line_id=event.line_id, ignored_reason=reason)
        finally:
            conn.close()

    def get_line_live(self, line_id: str, throughput_window_minutes: int = 5) -> dict[str, Any]:
        """Compute live totals, throughput, and health for a line."""
        conn = self._conn()
        try:
            with conn:
                self._close_stale_runs(conn)

                active_run = conn.execute(
                    "SELECT run_id FROM runs WHERE line_id=? AND ended_at IS NULL ORDER BY started_at DESC LIMIT 1",
                    (line_id,),
                ).fetchone()
                run_id = active_run["run_id"] if active_run else None

                filters = "line_id=?"
                params: list[Any] = [line_id]
                if run_id is not None:
                    filters += " AND run_id=?"
                    params.append(run_id)

                overall_row = conn.execute(
                    f"SELECT COALESCE(SUM(count_delta), 0) AS total FROM events WHERE event_type='COUNT' AND {filters}",
                    tuple(params),
                ).fetchone()
                overall = int(overall_row["total"]) if overall_row else 0

                by_sku_rows = conn.execute(
                    f"""
                    SELECT sku_id, COALESCE(SUM(count_delta), 0) AS total
                    FROM events
                    WHERE event_type='COUNT' AND {filters}
                    GROUP BY sku_id
                    """,
                    tuple(params),
                ).fetchall()
                by_sku = {str(row["sku_id"]): int(row["total"]) for row in by_sku_rows if row["sku_id"] is not None}

                since = (utc_now() - timedelta(minutes=throughput_window_minutes)).isoformat()
                throughput_params = params + [since]
                throughput_row = conn.execute(
                    f"""
                    SELECT COALESCE(SUM(count_delta), 0) AS total
                    FROM events
                    WHERE event_type='COUNT' AND {filters} AND timestamp >= ?
                    """,
                    tuple(throughput_params),
                ).fetchone()
                throughput_total = int(throughput_row["total"]) if throughput_row else 0
                units_per_min = throughput_total / max(float(throughput_window_minutes), 1.0)

                hb_row = conn.execute(
                    """
                    SELECT payload_json
                    FROM events
                    WHERE event_type='HEARTBEAT' AND line_id=?
                    ORDER BY timestamp DESC
                    LIMIT 1
                    """,
                    (line_id,),
                ).fetchone()
                health = {
                    "last_heartbeat": None,
                    "fps": None,
                    "uptime_s": None,
                    "brightness": None,
                    "blur_score": None,
                }
                if hb_row:
                    payload = json.loads(hb_row["payload_json"])
                    health = {
                        "last_heartbeat": payload.get("timestamp"),
                        "fps": payload.get("fps"),
                        "uptime_s": payload.get("uptime_s"),
                        "brightness": payload.get("brightness"),
                        "blur_score": payload.get("blur_score"),
                    }

                alerts_row = conn.execute(
                    """
                    SELECT COUNT(*) AS c
                    FROM events
                    WHERE event_type='ALERT' AND line_id=? AND timestamp >= ?
                    """,
                    (line_id, since),
                ).fetchone()
                alerts_recent = int(alerts_row["c"]) if alerts_row else 0

                return {
                    "line_id": line_id,
                    "run_id": run_id,
                    "generated_at": self._iso_now(),
                    "totals": {
                        "overall": overall,
                        "unknown": int(by_sku.get("UNKNOWN", 0)),
                        "by_sku": by_sku,
                    },
                    "throughput": {
                        "window_minutes": throughput_window_minutes,
                        "units_per_min": units_per_min,
                    },
                    "device_health": health,
                    "alerts_recent": alerts_recent,
                }
        finally:
            conn.close()

    def get_run_summary(self, run_id: str) -> dict[str, Any]:
        """Return aggregate summary for one run."""
        conn = self._conn()
        try:
            row = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(f"Run not found: {run_id}")

            by_sku_rows = conn.execute(
                """
                SELECT sku_id, COALESCE(SUM(count_delta), 0) AS total
                FROM events
                WHERE event_type='COUNT' AND run_id=?
                GROUP BY sku_id
                """,
                (run_id,),
            ).fetchall()
            by_sku = {str(r["sku_id"]): int(r["total"]) for r in by_sku_rows if r["sku_id"] is not None}
            overall = int(sum(by_sku.values()))

            alerts_row = conn.execute(
                "SELECT COUNT(*) AS c FROM events WHERE event_type='ALERT' AND run_id=?",
                (run_id,),
            ).fetchone()

            started = self._parse_ts(row["started_at"])
            ended = self._parse_ts(row["ended_at"]) if row["ended_at"] else None
            end_for_duration = ended or utc_now()
            elapsed_minutes = max((end_for_duration - started).total_seconds() / 60.0, 1e-6) if started else 1e-6

            return {
                "run_id": run_id,
                "line_id": row["line_id"],
                "device_id": row["device_id"],
                "started_at": row["started_at"],
                "ended_at": row["ended_at"],
                "overall": overall,
                "by_sku": by_sku,
                "unknown": int(by_sku.get("UNKNOWN", 0)),
                "alerts": int(alerts_row["c"]) if alerts_row else 0,
                "throughput_units_per_min": overall / elapsed_minutes,
            }
        finally:
            conn.close()

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        """List recent runs."""
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def start_run(self, line_id: str, device_id: str, run_id: str | None = None) -> str:
        """Manually start a run."""
        rid = run_id or f"manual-{device_id}-{int(utc_now().timestamp())}"
        conn = self._conn()
        try:
            with conn:
                conn.execute(
                    "UPDATE runs SET ended_at=? WHERE line_id=? AND device_id=? AND ended_at IS NULL",
                    (self._iso_now(), line_id, device_id),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO runs(run_id, line_id, device_id, started_at, ended_at, source) VALUES (?, ?, ?, ?, NULL, ?)",
                    (rid, line_id, device_id, self._iso_now(), "manual"),
                )
                self._ensure_device(conn, device_id, line_id)
            return rid
        finally:
            conn.close()

    def stop_run(self, run_id: str) -> bool:
        """Manually stop a run."""
        conn = self._conn()
        try:
            with conn:
                result = conn.execute(
                    "UPDATE runs SET ended_at=? WHERE run_id=? AND ended_at IS NULL",
                    (self._iso_now(), run_id),
                )
                return result.rowcount > 0
        finally:
            conn.close()

    def list_skus(self) -> list[dict[str, Any]]:
        """List known SKUs."""
        conn = self._conn()
        try:
            rows = conn.execute("SELECT * FROM skus ORDER BY sku_id").fetchall()
            out = []
            for row in rows:
                item = dict(row)
                item["metadata"] = json.loads(item["metadata_json"]) if item.get("metadata_json") else {}
                out.append(item)
            return out
        finally:
            conn.close()

    def upsert_sku(
        self,
        sku_id: str,
        name: str,
        profile_path: str | None,
        threshold: float | None,
        metadata: dict | None,
    ) -> dict[str, Any]:
        """Insert or update SKU metadata."""
        now = self._iso_now()
        meta_json = json.dumps(metadata or {})
        conn = self._conn()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO skus(sku_id, name, profile_path, threshold, metadata_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(sku_id) DO UPDATE SET
                      name=excluded.name,
                      profile_path=excluded.profile_path,
                      threshold=excluded.threshold,
                      metadata_json=excluded.metadata_json,
                      updated_at=excluded.updated_at
                    """,
                    (sku_id, name, profile_path, threshold, meta_json, now, now),
                )
            return {
                "sku_id": sku_id,
                "name": name,
                "profile_path": profile_path,
                "threshold": threshold,
                "metadata": metadata or {},
                "updated_at": now,
            }
        finally:
            conn.close()

    def get_device_config(self, device_id: str) -> dict[str, Any]:
        """Return device config + active SKU metadata."""
        conn = self._conn()
        try:
            row = conn.execute("SELECT * FROM devices WHERE device_id=?", (device_id,)).fetchone()
            if row is None:
                with conn:
                    conn.execute(
                        "INSERT INTO devices(device_id, line_id, config_json, paired) VALUES (?, ?, ?, 0)",
                        (device_id, None, "{}"),
                    )
                row = conn.execute("SELECT * FROM devices WHERE device_id=?", (device_id,)).fetchone()

            skus = self.list_skus()
            config_json = row["config_json"] or "{}"
            config = json.loads(config_json)
            return {
                "device_id": device_id,
                "line_id": row["line_id"],
                "paired": bool(row["paired"]),
                "pairing_required": self.pairing_required,
                "config": config,
                "active_skus": [
                    {
                        "sku_id": sku["sku_id"],
                        "name": sku["name"],
                        "profile_path": sku.get("profile_path"),
                        "threshold": sku.get("threshold"),
                    }
                    for sku in skus
                ],
            }
        finally:
            conn.close()

    def create_pair_code(self, line_id: str, ttl_seconds: int) -> dict[str, Any]:
        """Create one-time pairing code."""
        conn = self._conn()
        try:
            with conn:
                while True:
                    code = "".join(str(random.randint(0, 9)) for _ in range(6))
                    exists = conn.execute("SELECT 1 FROM pair_codes WHERE code=?", (code,)).fetchone()
                    if not exists:
                        break
                expires_at = (utc_now() + timedelta(seconds=ttl_seconds)).isoformat()
                conn.execute(
                    "INSERT INTO pair_codes(code, line_id, expires_at) VALUES (?, ?, ?)",
                    (code, line_id, expires_at),
                )
                return {"code": code, "line_id": line_id, "expires_at": expires_at}
        finally:
            conn.close()

    def pair_device(self, device_id: str, code: str) -> dict[str, Any]:
        """Pair device by one-time code, or no-op in dev mode."""
        conn = self._conn()
        try:
            with conn:
                if not self.pairing_required:
                    self._ensure_device(conn, device_id, "line-1")
                    conn.execute(
                        "UPDATE devices SET paired=1, paired_at=? WHERE device_id=?",
                        (self._iso_now(), device_id),
                    )
                    return {
                        "paired": True,
                        "device_id": device_id,
                        "line_id": "line-1",
                        "dev_mode": True,
                    }

                row = conn.execute(
                    "SELECT code, line_id, expires_at, used_by FROM pair_codes WHERE code=?",
                    (code,),
                ).fetchone()
                if row is None:
                    return {"paired": False, "reason": "invalid_code"}
                if row["used_by"] is not None:
                    return {"paired": False, "reason": "code_already_used"}
                if utc_now() > self._parse_ts(row["expires_at"]):
                    return {"paired": False, "reason": "code_expired"}

                line_id = row["line_id"] or "line-1"
                self._ensure_device(conn, device_id, line_id)
                conn.execute(
                    "UPDATE devices SET line_id=?, paired=1, paired_at=? WHERE device_id=?",
                    (line_id, self._iso_now(), device_id),
                )
                conn.execute(
                    "UPDATE pair_codes SET used_by=?, used_at=? WHERE code=?",
                    (device_id, self._iso_now(), code),
                )

                return {"paired": True, "device_id": device_id, "line_id": line_id}
        finally:
            conn.close()
