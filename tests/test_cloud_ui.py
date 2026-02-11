from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from prescience.cloud.app import create_app


def _build_client(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = create_app(db_path=tmp_path / "cloud.db", config_path=Path("configs/default.yaml"))
    app.state.job_runner.stop()
    app.state.job_runner.enqueue = lambda _job_id: None
    return TestClient(app), app


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
        assert "/ui/onboarding/" in response.headers["location"]

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
