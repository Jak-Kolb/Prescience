"""Cloud FastAPI application factory and server entrypoint."""

from __future__ import annotations

from pathlib import Path
import time

import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from prescience.cloud.jobs import UIJobRunner
from prescience.cloud.routes import router
from prescience.cloud.store import CloudStore
from prescience.cloud.stream import SSEBroadcaster
from prescience.config import AppSettings, load_settings
from prescience.pipeline.tracking_session import TrackingSessionManager


def create_app(db_path: str | Path, config_path: str | Path) -> FastAPI:
    """Create configured FastAPI app."""
    settings: AppSettings = load_settings(config_path)
    repo_root = Path(__file__).resolve().parents[3]
    templates_dir = Path(__file__).resolve().parent / "templates"
    static_dir = repo_root / "static"

    app = FastAPI(title="Prescience Cloud", version="0.1.0")
    templates = Jinja2Templates(directory=str(templates_dir))
    app.state.asset_version = str(int(time.time()))

    app.state.settings = settings
    app.state.store = CloudStore(
        db_path=db_path,
        heartbeat_timeout_seconds=settings.cloud.heartbeat_timeout_seconds,
        pairing_required=settings.cloud.pairing.required,
    )
    app.state.store.sync_skus_from_profiles(settings.profiles.root)
    app.state.stream_bus = SSEBroadcaster()
    app.state.templates = templates
    app.state.job_runner = UIJobRunner(app.state.store)
    app.state.tracking_manager = TrackingSessionManager()
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    app.include_router(router)
    app.state.job_runner.start()

    @app.on_event("shutdown")
    def _shutdown_workers() -> None:
        app.state.job_runner.stop()
        app.state.tracking_manager.stop_all()

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request, line_id: str = Query(settings.line.line_id)):
        store: CloudStore = request.app.state.store
        live = store.get_line_live(line_id=line_id)
        runs = store.list_runs()
        skus = store.list_skus()
        sessions = store.list_onboarding_sessions(limit=20)
        for sku in skus:
            versions = store.list_model_versions_for_sku(sku["sku_id"])
            sku["model_versions"] = versions
            sku["latest_version"] = f"v{versions[-1]}" if versions else None
            sku["full_train_ready"] = store.sku_full_train_ready(sku["sku_id"])

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "line_id": line_id,
                "live": live,
                "runs": runs,
                "skus": skus,
                "sessions": sessions,
                "pairing_required": settings.cloud.pairing.required,
                "asset_version": app.state.asset_version,
            },
        )

    return app


def serve(host: str, port: int, db_path: str | Path, config_path: str | Path) -> None:
    """Run cloud app with single worker (required for in-memory SSE bus)."""
    app = create_app(db_path=db_path, config_path=config_path)
    uvicorn.run(app, host=host, port=port, workers=1)
