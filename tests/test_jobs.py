from __future__ import annotations

from pathlib import Path

from prescience.cloud.app import create_app


def test_ui_job_snapshot_contains_jobs_and_sessions(tmp_path: Path) -> None:
    store_app = create_app(db_path=tmp_path / "cloud.db", config_path=Path("configs/default.yaml"))
    store = store_app.state.store
    store_app.state.job_runner.stop()

    session = store.create_onboarding_session(
        sku_id="can1_test",
        version_tag="v1",
        mode="quick",
        state="seed_labeling",
        seed_candidates=[{"frame_name": "000001.jpg", "status": "pending"}],
    )
    job = store.create_ui_job(
        sku_id="can1_test",
        job_type="train_stage1",
        status="queued",
        step="train_stage1",
        payload={"session_id": session["session_id"]},
    )
    store.update_onboarding_session(session["session_id"], latest_job_id=job["job_id"])

    snapshot = store.get_ui_snapshot(sku_id="can1_test")
    assert len(snapshot["jobs"]) == 1
    assert len(snapshot["sessions"]) == 1
    assert snapshot["jobs"][0]["type"] == "train_stage1"
    assert snapshot["sessions"][0]["state"] == "seed_labeling"
