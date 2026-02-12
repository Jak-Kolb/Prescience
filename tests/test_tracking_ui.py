from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from prescience.cloud.app import create_app
from prescience.pipeline.tracking_session import TrackingSession, TrackingSessionManager


def test_tracking_manager_start_stop_lifecycle(monkeypatch, tmp_path: Path) -> None:
    def fake_start(self):  # noqa: ANN001
        self.status = "running"
        self._latest_jpeg = b"\xff\xd8fake\xff\xd9"

    def fake_stop(self):  # noqa: ANN001
        self.status = "stopped"

    monkeypatch.setattr(TrackingSession, "start", fake_start)
    monkeypatch.setattr(TrackingSession, "stop", fake_stop)

    mgr = TrackingSessionManager()
    session = mgr.start_session(
        sku_id="can1_test",
        source="0",
        line_id="line-1",
        device_id="device-1",
        model_path="data/models/yolo/can1_test_v1/best.pt",
        zone_config_path=tmp_path / "line-1.yaml",
        config_path=tmp_path / "default.yaml",
        event_endpoint=None,
        run_id=None,
    )
    assert session.status == "running"
    assert mgr.get_session(session.session_id) is not None
    assert mgr.stop_session(session.session_id) is True
    assert mgr.get_session(session.session_id).status == "stopped"  # type: ignore[union-attr]


def test_tracking_start_route_and_stream(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    app = create_app(db_path=tmp_path / "cloud.db", config_path=Path("configs/default.yaml"))
    app.state.job_runner.stop()

    model_dir = tmp_path / "data" / "models" / "yolo" / "can1_test_v1"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "best.pt").write_bytes(b"pt")

    class DummySession:
        session_id = "sess-1"

        def latest_jpeg(self):
            return b"\xff\xd8fake\xff\xd9"

        def snapshot_status(self):
            return {"session_id": "sess-1", "status": "running", "total_count": 0, "fps": 0.0}

    class DummyManager:
        def __init__(self):
            self.started = False

        def start_session(self, **_kwargs):
            self.started = True
            return DummySession()

        def get_session(self, _session_id: str):
            return DummySession()

        def stop_session(self, _session_id: str):
            return True

        def stop_all(self):
            return None

    app.state.tracking_manager = DummyManager()

    with TestClient(app) as client:
        start = client.post(
            "/ui/tracking/start",
            json={"sku_id": "can1_test", "source": "0", "line_id": "line-1", "device_id": "device-1"},
        )
        assert start.status_code == 200
        body = start.json()
        assert body["ok"] is True
        assert body["session_id"] == "sess-1"

        status = client.get("/api/tracking/sess-1")
        assert status.status_code == 200
        assert status.json()["status"] == "running"

        stopped = client.post("/ui/tracking/sess-1/stop")
        assert stopped.status_code == 200
        assert stopped.json()["ok"] is True


def test_zone_and_tracking_pages_render(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    app = create_app(db_path=tmp_path / "cloud.db", config_path=Path("configs/default.yaml"))
    app.state.job_runner.stop()

    with TestClient(app) as client:
        zone = client.get("/ui/zone?sku_id=can1_test&line_id=line-1")
        assert zone.status_code == 200
        assert "Zone Setup" in zone.text
        assert "Go to Tracking" in zone.text

        tracking = client.get("/ui/tracking?sku_id=can1_test&line_id=line-1")
        assert tracking.status_code == 200
        assert "Tracking Console" in tracking.text
        assert "Run Tracking" in tracking.text
        assert "in_and_out" in tracking.text
