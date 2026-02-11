"""Cloud FastAPI application factory and server entrypoint."""

from __future__ import annotations

from pathlib import Path

import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from prescience.cloud.routes import router
from prescience.cloud.store import CloudStore
from prescience.cloud.stream import SSEBroadcaster
from prescience.config import AppSettings, load_settings


def create_app(db_path: str | Path, config_path: str | Path) -> FastAPI:
    """Create configured FastAPI app."""
    settings: AppSettings = load_settings(config_path)

    app = FastAPI(title="Prescience Cloud", version="0.1.0")
    templates = Jinja2Templates(directory="src/prescience/cloud/templates")

    app.state.settings = settings
    app.state.store = CloudStore(
        db_path=db_path,
        heartbeat_timeout_seconds=settings.cloud.heartbeat_timeout_seconds,
        pairing_required=settings.cloud.pairing.required,
    )
    app.state.stream_bus = SSEBroadcaster()
    app.state.templates = templates

    app.include_router(router)

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request, line_id: str = Query(settings.line.line_id)):
        store: CloudStore = request.app.state.store
        live = store.get_line_live(line_id=line_id)
        runs = store.list_runs()
        skus = store.list_skus()

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "line_id": line_id,
                "live": live,
                "runs": runs,
                "skus": skus,
                "pairing_required": settings.cloud.pairing.required,
            },
        )

    return app


def serve(host: str, port: int, db_path: str | Path, config_path: str | Path) -> None:
    """Run cloud app with single worker (required for in-memory SSE bus)."""
    app = create_app(db_path=db_path, config_path=config_path)
    uvicorn.run(app, host=host, port=port, workers=1)
