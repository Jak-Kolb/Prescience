from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2
from fastapi.testclient import TestClient

from prescience.cloud.app import create_app


def _build_client(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = create_app(db_path=tmp_path / "cloud.db", config_path=Path("configs/default.yaml"))
    app.state.job_runner.stop()
    app.state.job_runner.enqueue = lambda _job_id: None
    return TestClient(app), app


def _write_frame(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = np.full((120, 160, 3), 200, dtype=np.uint8)
    cv2.imwrite(str(path), frame)


def test_ui_enroll_creates_session_and_job(tmp_path: Path, monkeypatch) -> None:
    client, app = _build_client(tmp_path, monkeypatch)
    with client:
        response = client.post(
            "/ui/sku/enroll",
            data={"sku": "chickfila_sauce"},
            files={"video": ("video.MOV", b"fake-mov", "video/quicktime")},
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert response.headers["location"].startswith("/?training_started=1")

        saved = tmp_path / "data" / "raw" / "videos" / "chickfila_sauce" / "chickfila_sauce_0.MOV"
        assert saved.exists()

        sessions = app.state.store.list_onboarding_sessions(sku_id="chickfila_sauce")
        assert len(sessions) == 1
        assert sessions[0]["version_tag"] == "v1"
        assert sessions[0]["state"] == "extracting"

        jobs = app.state.store.list_ui_jobs(sku_id="chickfila_sauce")
        assert len(jobs) == 1
        assert jobs[0]["type"] == "extract_prepare_seed"


def test_ui_append_train_uses_incremented_video_name(tmp_path: Path, monkeypatch) -> None:
    client, app = _build_client(tmp_path, monkeypatch)
    with client:
        base = tmp_path / "data" / "raw" / "videos" / "chickfila_sauce"
        base.mkdir(parents=True, exist_ok=True)
        (base / "chickfila_sauce_0.MOV").write_bytes(b"v0")
        model_dir = tmp_path / "data" / "models" / "yolo" / "chickfila_sauce_v1"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "best.pt").write_bytes(b"pt")

        response = client.post(
            "/ui/sku/chickfila_sauce/append-train",
            files={"video": ("video2.MOV", b"fake-2", "video/quicktime")},
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert response.headers["location"].startswith("/?training_started=1")
        assert (base / "chickfila_sauce_1.MOV").exists()

        session = app.state.store.list_onboarding_sessions(sku_id="chickfila_sauce")[0]
        assert session["version_tag"] == "v2"


def test_full_train_gate_requires_v2(tmp_path: Path, monkeypatch) -> None:
    client, app = _build_client(tmp_path, monkeypatch)
    with client:
        app.state.store.upsert_sku(
            sku_id="can1_test",
            name="can1_test",
            profile_path="data/profiles/can1_test",
            threshold=None,
            metadata={},
        )

        v1 = tmp_path / "data" / "models" / "yolo" / "can1_test_v1"
        v1.mkdir(parents=True, exist_ok=True)
        (v1 / "best.pt").write_bytes(b"pt")

        blocked = client.post("/ui/sku/can1_test/train/full")
        assert blocked.status_code == 200
        assert "requires" in blocked.text

        v2 = tmp_path / "data" / "models" / "yolo" / "can1_test_v2"
        v2.mkdir(parents=True, exist_ok=True)
        (v2 / "best.pt").write_bytes(b"pt")

        allowed = client.post("/ui/sku/can1_test/train/full")
        assert allowed.status_code == 200
        assert "Queued full training job" in allowed.text

        jobs = app.state.store.list_ui_jobs(sku_id="can1_test")
        assert jobs[0]["type"] == "full_train"


def test_full_train_gate_allows_latest_version_when_older_versions_pruned(tmp_path: Path, monkeypatch) -> None:
    client, app = _build_client(tmp_path, monkeypatch)
    with client:
        app.state.store.upsert_sku(
            sku_id="can1_test",
            name="can1_test",
            profile_path="data/profiles/can1_test",
            threshold=None,
            metadata={},
        )

        # Simulate pruning: only v3 remains.
        v3 = tmp_path / "data" / "models" / "yolo" / "can1_test_v3"
        v3.mkdir(parents=True, exist_ok=True)
        (v3 / "best.pt").write_bytes(b"pt")

        allowed = client.post("/ui/sku/can1_test/train/full")
        assert allowed.status_code == 200
        assert "Queued full training job" in allowed.text


def test_zone_config_round_trip(tmp_path: Path, monkeypatch) -> None:
    client, _app = _build_client(tmp_path, monkeypatch)
    with client:
        payload = {
            "polygon": [[10, 10], [120, 10], [120, 90], [10, 90]],
            "direction": "left_to_right",
        }
        save = client.post("/api/zone/line-7", json=payload)
        assert save.status_code == 200
        assert save.json()["ok"] is True

        loaded = client.get("/api/zone/line-7")
        assert loaded.status_code == 200
        body = loaded.json()
        assert body["zone"]["direction"] == "left_to_right"
        assert body["zone"]["polygon"] == payload["polygon"]


def test_dashboard_shows_separate_zone_and_tracking_actions(tmp_path: Path, monkeypatch) -> None:
    client, app = _build_client(tmp_path, monkeypatch)
    with client:
        app.state.store.upsert_sku(
            sku_id="can1_test",
            name="can1_test",
            profile_path="data/profiles/can1_test",
            threshold=None,
            metadata={},
        )
        v1 = tmp_path / "data" / "models" / "yolo" / "can1_test_v1"
        v1.mkdir(parents=True, exist_ok=True)
        (v1 / "best.pt").write_bytes(b"pt")

        response = client.get("/")
        assert response.status_code == 200
        assert "Define Zone" in response.text
        assert "Run Tracking" in response.text


def test_zone_frame_returns_jpeg(tmp_path: Path, monkeypatch) -> None:
    client, _app = _build_client(tmp_path, monkeypatch)

    class FakeCapture:
        def __init__(self, _source) -> None:
            self._opened = True

        def isOpened(self) -> bool:  # noqa: N802
            return self._opened

        def read(self):
            frame = np.zeros((32, 48, 3), dtype=np.uint8)
            frame[:, :, 1] = 200
            return True, frame

        def release(self) -> None:
            return None

    monkeypatch.setattr("prescience.cloud.routes.cv2.VideoCapture", FakeCapture)

    with client:
        response = client.get("/api/zone/frame", params={"source": "0 "})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("image/jpeg")
        assert response.content[:2] == b"\xff\xd8"


def test_append_approval_requires_30_decisions_before_retrain(tmp_path: Path, monkeypatch) -> None:
    client, app = _build_client(tmp_path, monkeypatch)
    with client:
        session = app.state.store.create_onboarding_session(
            sku_id="chickfila_sauce",
            version_tag="v2",
            mode="quick",
            state="approval_labeling",
            seed_candidates=[],
            approval_candidates=[
                {"frame_name": f"{idx:06d}.jpg", "status": "pending", "boxes": [], "proposals": []}
                for idx in range(1, 31)
            ],
        )
        gate_job = app.state.store.create_ui_job(
            sku_id="chickfila_sauce",
            job_type="extract_prepare_seed",
            status="waiting_user",
            step="approval_labeling",
            payload={
                "session_id": session["session_id"],
                "append": True,
                "approval_only": True,
                "required_approval_count": 30,
            },
        )
        app.state.store.update_onboarding_session(session["session_id"], latest_job_id=gate_job["job_id"])

        blocked = client.post(f"/api/onboarding/{session['session_id']}/approvals/complete")
        assert blocked.status_code == 422
        assert "at least 30" in blocked.json()["detail"]

        approved = [
            {"frame_name": f"{idx:06d}.jpg", "status": "positive", "boxes": [{"x1": 1, "y1": 1, "x2": 5, "y2": 5}]}
            for idx in range(1, 31)
        ]
        app.state.store.update_onboarding_session(session["session_id"], approval_candidates=approved)

        allowed = client.post(f"/api/onboarding/{session['session_id']}/approvals/complete")
        assert allowed.status_code == 200
        assert allowed.json()["ok"] is True


def test_seed_auto_start_enqueues_stage1_after_three_reviews(tmp_path: Path, monkeypatch) -> None:
    client, app = _build_client(tmp_path, monkeypatch)
    with client:
        sku = "chickfila_sauce"
        frames = tmp_path / "data" / "derived" / "frames" / sku / "frames"
        for idx in range(1, 5):
            _write_frame(frames / f"{idx:06d}.jpg")

        session = app.state.store.create_onboarding_session(
            sku_id=sku,
            version_tag="v1",
            mode="milestone",
            state="seed_labeling",
            seed_candidates=[
                {"frame_name": f"{idx:06d}.jpg", "status": "pending", "boxes": []}
                for idx in range(1, 5)
            ],
            approval_candidates=[],
        )
        seed_job = app.state.store.create_ui_job(
            sku_id=sku,
            job_type="extract_prepare_seed",
            status="waiting_user",
            step="seed_labeling",
            payload={
                "session_id": session["session_id"],
                "seed_required_reviews": 3,
                "seed_auto_start": True,
                "required_approval_count": 24,
                "object_description": sku,
            },
        )
        app.state.store.update_onboarding_session(session["session_id"], latest_job_id=seed_job["job_id"])

        for idx in range(1, 4):
            response = client.post(
                f"/api/onboarding/{session['session_id']}/labels",
                json={
                    "frame_name": f"{idx:06d}.jpg",
                    "status": "positive",
                    "stage": "seed",
                    "boxes": [{"x1": 10, "y1": 10, "x2": 50, "y2": 60}],
                },
            )
            assert response.status_code == 200

        session_after = app.state.store.get_onboarding_session(session["session_id"])
        assert session_after is not None
        assert session_after["state"] == "train_stage1"
        latest = app.state.store.get_ui_job(session_after["latest_job_id"])
        assert latest is not None
        assert latest["type"] == "train_stage1"

        stage1_jobs = [job for job in app.state.store.list_ui_jobs(sku_id=sku) if job["type"] == "train_stage1"]
        assert len(stage1_jobs) == 1


def test_onboarding_session_api_includes_seed_gate(tmp_path: Path, monkeypatch) -> None:
    client, app = _build_client(tmp_path, monkeypatch)
    with client:
        sku = "chickfila_sauce"
        session = app.state.store.create_onboarding_session(
            sku_id=sku,
            version_tag="v1",
            mode="milestone",
            state="seed_labeling",
            seed_candidates=[
                {"frame_name": "000001.jpg", "status": "positive", "boxes": [{"x1": 1, "y1": 1, "x2": 2, "y2": 2}]},
                {"frame_name": "000002.jpg", "status": "negative", "boxes": []},
                {"frame_name": "000003.jpg", "status": "pending", "boxes": []},
            ],
            approval_candidates=[],
        )
        seed_job = app.state.store.create_ui_job(
            sku_id=sku,
            job_type="extract_prepare_seed",
            status="waiting_user",
            step="seed_labeling",
            payload={
                "session_id": session["session_id"],
                "seed_required_reviews": 3,
                "seed_auto_start": True,
            },
        )
        app.state.store.update_onboarding_session(session["session_id"], latest_job_id=seed_job["job_id"])

        response = client.get(f"/api/onboarding/{session['session_id']}")
        assert response.status_code == 200
        body = response.json()
        assert body["seed_gate"]["required"] == 3
        assert body["seed_gate"]["reviewed"] == 2
        assert body["seed_gate"]["remaining"] == 1
        assert body["seed_gate"]["auto_start"] is True
        assert "auto_mode" in body
        assert "manual_required" in body
        assert "manual_reason" in body


def test_manual_enter_route_transitions_session_state(tmp_path: Path, monkeypatch) -> None:
    client, app = _build_client(tmp_path, monkeypatch)
    with client:
        session = app.state.store.create_onboarding_session(
            sku_id="can1_test",
            version_tag="v2",
            mode="quick",
            state="manual_required",
            seed_candidates=[],
            approval_candidates=[{"frame_name": "000001.jpg", "status": "pending", "boxes": []}],
            session_context={
                "flow_kind": "append",
                "manual_fallback_reason": "missing_api_key_env:GEMINI_API_KEY",
            },
        )
        response = client.post(f"/api/onboarding/{session['session_id']}/manual/enter")
        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is True
        updated = app.state.store.get_onboarding_session(session["session_id"])
        assert updated is not None
        assert updated["state"] == "approval_labeling"
