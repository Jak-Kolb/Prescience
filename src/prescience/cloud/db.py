"""SQLite database initialization and connection helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id TEXT NOT NULL UNIQUE,
  event_type TEXT NOT NULL,
  seq INTEGER NOT NULL,
  timestamp TEXT NOT NULL,
  frame_ts TEXT,
  line_id TEXT NOT NULL,
  device_id TEXT NOT NULL,
  run_id TEXT,
  payload_json TEXT NOT NULL,
  sku_id TEXT,
  count_delta INTEGER,
  counts_total_overall INTEGER
);

CREATE INDEX IF NOT EXISTS idx_events_line_ts ON events(line_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_events_run_seq ON events(run_id, seq);
CREATE INDEX IF NOT EXISTS idx_events_device_run_seq ON events(device_id, run_id, seq);

CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  line_id TEXT NOT NULL,
  device_id TEXT NOT NULL,
  started_at TEXT NOT NULL,
  ended_at TEXT,
  source TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_line_active ON runs(line_id, ended_at);

CREATE TABLE IF NOT EXISTS skus (
  sku_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  profile_path TEXT,
  threshold REAL,
  metadata_json TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS devices (
  device_id TEXT PRIMARY KEY,
  line_id TEXT,
  config_json TEXT,
  paired INTEGER NOT NULL DEFAULT 0,
  paired_at TEXT
);

CREATE TABLE IF NOT EXISTS pair_codes (
  code TEXT PRIMARY KEY,
  line_id TEXT,
  expires_at TEXT NOT NULL,
  used_by TEXT,
  used_at TEXT
);

CREATE TABLE IF NOT EXISTS ui_jobs (
  job_id TEXT PRIMARY KEY,
  sku_id TEXT NOT NULL,
  type TEXT NOT NULL,
  status TEXT NOT NULL,
  step TEXT,
  progress REAL,
  message TEXT,
  payload_json TEXT,
  error_json TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ui_jobs_sku_updated ON ui_jobs(sku_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_ui_jobs_status_updated ON ui_jobs(status, updated_at DESC);

CREATE TABLE IF NOT EXISTS onboarding_sessions (
  session_id TEXT PRIMARY KEY,
  sku_id TEXT NOT NULL,
  version_tag TEXT,
  mode TEXT NOT NULL,
  state TEXT NOT NULL,
  seed_candidates_json TEXT,
  approval_candidates_json TEXT,
  session_context_json TEXT,
  stage1_model_path TEXT,
  latest_job_id TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_onboarding_sessions_sku_updated ON onboarding_sessions(sku_id, updated_at DESC);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open SQLite connection with row factory configured."""
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(str(row["name"]) == column for row in rows)


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, column_sql: str) -> None:
    if _column_exists(conn, table, column):
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_sql}")


def init_db(db_path: str | Path) -> None:
    """Initialize database schema."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = connect(path)
    try:
        conn.executescript(SCHEMA_SQL)
        _ensure_column(conn, "onboarding_sessions", "session_context_json", "TEXT")
        conn.commit()
    finally:
        conn.close()
