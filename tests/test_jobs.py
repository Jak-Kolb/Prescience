from __future__ import annotations

from pathlib import Path

from prescience.cloud.app import create_app
from prescience.cloud.jobs import UIJobRunner


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


def test_extract_append_job_skips_seed_and_prepares_approval(monkeypatch, tmp_path: Path) -> None:
    app = create_app(db_path=tmp_path / "cloud.db", config_path=Path("configs/default.yaml"))
    store = app.state.store
    app.state.job_runner.stop()
    monkeypatch.chdir(tmp_path)

    model_dir = Path("data/models/yolo/can1_test_v1")
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "best.pt").write_bytes(b"pt")

    session = store.create_onboarding_session(
        sku_id="can1_test",
        version_tag="v2",
        mode="quick",
        state="extracting",
        seed_candidates=[],
        approval_candidates=[],
    )
    job = store.create_ui_job(
        sku_id="can1_test",
        job_type="extract_prepare_seed",
        status="queued",
        step="extracting",
        payload={
            "session_id": session["session_id"],
            "video_path": "data/raw/videos/can1_test/can1_test_1.MOV",
            "append": True,
            "required_approval_count": 30,
            "sections": 6,
            "approve_per_bin": 5,
        },
    )
    store.update_onboarding_session(session["session_id"], latest_job_id=job["job_id"])

    def _fake_extract(**_kwargs):
        return None

    def _fake_prepare_approval(*_args, **_kwargs):
        return [
            {"frame_name": f"{idx:06d}.jpg", "status": "pending", "boxes": [], "proposals": []}
            for idx in range(1, 31)
        ]

    def _fail_seed(*_args, **_kwargs):
        raise AssertionError("Seed path should not execute for append flow")

    monkeypatch.setattr("prescience.cloud.jobs.extract_frames_for_sku", _fake_extract)
    monkeypatch.setattr("prescience.cloud.jobs.prepare_approval_candidates", _fake_prepare_approval)
    monkeypatch.setattr("prescience.cloud.jobs.prepare_seed_candidates", _fail_seed)

    runner = UIJobRunner(store)
    runner._job_extract_prepare_seed(store.get_ui_job(job["job_id"]))

    updated = store.get_onboarding_session(session["session_id"])
    assert updated["state"] == "approval_labeling"
    assert updated["seed_candidates"] == []
    assert len(updated["approval_candidates"]) == 30
    assert str(updated["stage1_model_path"]).endswith("data/models/yolo/can1_test_v1/best.pt")

    updated_job = store.get_ui_job(job["job_id"])
    assert updated_job["status"] == "waiting_user"
    assert updated_job["step"] == "approval_labeling"
    assert updated_job["payload"]["approval_only"] is True
    assert updated_job["payload"]["required_approval_count"] == 30
