from __future__ import annotations

import json
from pathlib import Path

import pytest

from prescience.cloud.app import create_app
from prescience.cloud.jobs import UIJobRunner
from prescience.pipeline.web_onboarding import AutoLabelSummary


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
    monkeypatch.setattr("prescience.cloud.jobs.prepare_approval_candidates_with_gemini", _fake_prepare_approval)
    monkeypatch.setattr("prescience.cloud.jobs.prepare_seed_candidates_with_gemini", _fail_seed)

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


def test_extract_append_job_auto_mode_queues_stage2_for_trusted_sku(monkeypatch, tmp_path: Path) -> None:
    app = create_app(db_path=tmp_path / "cloud.db", config_path=Path("configs/default.yaml"))
    store = app.state.store
    app.state.job_runner.stop()
    monkeypatch.chdir(tmp_path)

    model_dir = Path("data/models/yolo/can1_test_v1")
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "best.pt").write_bytes(b"pt")
    store.upsert_sku(
        sku_id="can1_test",
        name="can1_test",
        profile_path="data/profiles/can1_test",
        threshold=None,
        metadata={
            "labeling_mode": "auto_trusted",
            "auto_mode_enabled": True,
            "trust_review_count": 3,
        },
    )

    session = store.create_onboarding_session(
        sku_id="can1_test",
        version_tag="v2",
        mode="quick",
        state="extracting",
        seed_candidates=[],
        approval_candidates=[],
        session_context={"flow_kind": "append", "auto_mode_expected": True},
    )
    job = store.create_ui_job(
        sku_id="can1_test",
        job_type="extract_prepare_seed",
        status="queued",
        step="extracting",
        payload={
            "session_id": session["session_id"],
            "video_path": "data/raw/videos/can1_test/can1_test_2.MOV",
            "append": True,
            "required_approval_count": 30,
            "sections": 6,
        },
    )
    store.update_onboarding_session(session["session_id"], latest_job_id=job["job_id"])

    monkeypatch.setattr("prescience.cloud.jobs.extract_frames_for_sku", lambda **_kwargs: None)
    monkeypatch.setattr(
        "prescience.cloud.jobs.auto_label_for_stage2",
        lambda **_kwargs: AutoLabelSummary(
            total=4,
            accepted=3,
            adjusted=1,
            rejected_negative=0,
            uncertain=0,
            decisions=[
                {"frame_name": "000101.jpg", "decision": "accept", "final_boxes": [{"x1": 1, "y1": 1, "x2": 5, "y2": 5}], "reason": "ok"},
                {"frame_name": "000102.jpg", "decision": "adjust", "final_boxes": [{"x1": 2, "y1": 2, "x2": 6, "y2": 6}], "reason": "ok"},
                {"frame_name": "000103.jpg", "decision": "accept", "final_boxes": [{"x1": 3, "y1": 3, "x2": 7, "y2": 7}], "reason": "ok"},
                {"frame_name": "000104.jpg", "decision": "accept", "final_boxes": [{"x1": 4, "y1": 4, "x2": 8, "y2": 8}], "reason": "ok"},
            ],
        ),
    )

    runner = UIJobRunner(store, settings=app.state.settings)
    runner._job_extract_prepare_seed(store.get_ui_job(job["job_id"]))

    updated = store.get_onboarding_session(session["session_id"])
    assert updated is not None
    assert updated["state"] == "auto_ready_stage2"
    assert updated["latest_job_id"]
    stage2_job = store.get_ui_job(updated["latest_job_id"])
    assert stage2_job is not None
    assert stage2_job["type"] == "train_stage2"


def test_append_conf_first_append_stays_baseline(monkeypatch, tmp_path: Path) -> None:
    app = create_app(db_path=tmp_path / "cloud.db", config_path=Path("configs/default.yaml"))
    store = app.state.store
    app.state.job_runner.stop()
    monkeypatch.chdir(tmp_path)

    model_dir = Path("data/models/yolo/can1_test_v1")
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "best.pt").write_bytes(b"pt")
    eval_dir = model_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "metrics.json").write_text(
        json.dumps({"prediction_stats": {"mean_top_confidence": 0.74}}),
        encoding="utf-8",
    )

    runner = UIJobRunner(store, settings=app.state.settings)
    conf = runner._resolve_append_auto_label_conf(
        sku="can1_test",
        target_version=2,
        payload={"detector_conf_for_auto_label": 0.02},
    )
    assert conf == 0.02


def test_append_conf_later_append_uses_prior_eval_mean_top(monkeypatch, tmp_path: Path) -> None:
    app = create_app(db_path=tmp_path / "cloud.db", config_path=Path("configs/default.yaml"))
    store = app.state.store
    app.state.job_runner.stop()
    monkeypatch.chdir(tmp_path)

    for version in (1, 2):
        model_dir = Path(f"data/models/yolo/can1_test_v{version}")
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "best.pt").write_bytes(b"pt")
    eval_dir = Path("data/models/yolo/can1_test_v2/eval")
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "metrics.json").write_text(
        json.dumps({"prediction_stats": {"mean_top_confidence": 0.67}}),
        encoding="utf-8",
    )

    runner = UIJobRunner(store, settings=app.state.settings)
    conf = runner._resolve_append_auto_label_conf(
        sku="can1_test",
        target_version=3,
        payload={"detector_conf_for_auto_label": 0.02},
    )
    assert conf == pytest.approx(0.65)


def test_append_conf_later_append_falls_back_to_prior_eval_max(monkeypatch, tmp_path: Path) -> None:
    app = create_app(db_path=tmp_path / "cloud.db", config_path=Path("configs/default.yaml"))
    store = app.state.store
    app.state.job_runner.stop()
    monkeypatch.chdir(tmp_path)

    for version in (1, 2):
        model_dir = Path(f"data/models/yolo/can1_test_v{version}")
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "best.pt").write_bytes(b"pt")
    eval_dir = Path("data/models/yolo/can1_test_v2/eval")
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "metrics.json").write_text(
        json.dumps({"prediction_stats": {"max_confidence": 0.58}}),
        encoding="utf-8",
    )

    runner = UIJobRunner(store, settings=app.state.settings)
    conf = runner._resolve_append_auto_label_conf(
        sku="can1_test",
        target_version=3,
        payload={"detector_conf_for_auto_label": 0.02},
    )
    assert conf == pytest.approx(0.56)
