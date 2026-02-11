"""FastAPI routes for cloud API, SSE stream, and HTMX actions."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse

from prescience.cloud.schemas import PairCodeCreateRequest, PairDeviceRequest, RunStartRequest, SKUUpsertRequest
from prescience.cloud.store import CloudStore
from prescience.cloud.stream import SSEBroadcaster, encode_sse
from prescience.events.schemas import AnyEvent
from prescience.pipeline.enroll import next_enrollment_video_path, normalize_sku_name

router = APIRouter()


def _store(request: Request) -> CloudStore:
    return request.app.state.store


def _stream_bus(request: Request) -> SSEBroadcaster:
    return request.app.state.stream_bus


@router.post("/events")
async def post_event(event: AnyEvent, request: Request) -> dict[str, Any]:
    store = _store(request)
    stream_bus = _stream_bus(request)

    result = store.ingest_event(event)
    if result.status == "inserted":
        await stream_bus.publish(
            {
                "line_id": result.line_id,
                "run_id": result.run_id,
                "event_type": event.event_type,
            }
        )

    return {
        "status": result.status,
        "line_id": result.line_id,
        "run_id": result.run_id,
        "ignored_reason": result.ignored_reason,
    }


@router.get("/stream")
async def stream_events(
    request: Request,
    line_id: str = Query("line-1"),
) -> StreamingResponse:
    store = _store(request)
    stream_bus = _stream_bus(request)
    retry_ms = int(request.app.state.settings.cloud.sse_retry_ms)

    async def event_generator():
        queue = await stream_bus.subscribe()
        try:
            snapshot = store.get_line_live(line_id=line_id)
            yield encode_sse({"type": "snapshot", "live": snapshot}, retry_ms=retry_ms)

            while True:
                if await request.is_disconnected():
                    break
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=15.0)
                except TimeoutError:
                    yield ": keep-alive\n\n"
                    continue

                msg_line = message.get("line_id")
                if msg_line != line_id:
                    continue

                live = store.get_line_live(line_id=line_id)
                yield encode_sse({"type": "update", "live": live})
        finally:
            await stream_bus.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/lines/{line_id}/live")
def get_line_live(line_id: str, request: Request) -> dict[str, Any]:
    return _store(request).get_line_live(line_id=line_id)


@router.get("/runs")
def get_runs(request: Request) -> list[dict[str, Any]]:
    return _store(request).list_runs()


@router.get("/runs/{run_id}/summary")
def get_run_summary(run_id: str, request: Request) -> dict[str, Any]:
    store = _store(request)
    try:
        return store.get_run_summary(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/runs/start")
def start_run(payload: RunStartRequest, request: Request) -> dict[str, str]:
    run_id = _store(request).start_run(line_id=payload.line_id, device_id=payload.device_id, run_id=payload.run_id)
    return {"status": "started", "run_id": run_id}


@router.post("/runs/{run_id}/stop")
def stop_run(run_id: str, request: Request) -> dict[str, Any]:
    stopped = _store(request).stop_run(run_id)
    return {"status": "stopped" if stopped else "noop", "run_id": run_id}


@router.get("/skus")
def list_skus(request: Request) -> list[dict[str, Any]]:
    return _store(request).list_skus()


@router.post("/skus")
def create_sku(payload: SKUUpsertRequest, request: Request) -> dict[str, Any]:
    return _store(request).upsert_sku(
        sku_id=payload.sku_id,
        name=payload.name,
        profile_path=payload.profile_path,
        threshold=payload.threshold,
        metadata=payload.metadata,
    )


@router.put("/skus/{sku_id}")
def update_sku(sku_id: str, payload: SKUUpsertRequest, request: Request) -> dict[str, Any]:
    return _store(request).upsert_sku(
        sku_id=sku_id,
        name=payload.name,
        profile_path=payload.profile_path,
        threshold=payload.threshold,
        metadata=payload.metadata,
    )


@router.get("/devices/{device_id}/config")
def get_device_config(device_id: str, request: Request) -> dict[str, Any]:
    out = _store(request).get_device_config(device_id)
    settings = request.app.state.settings
    out["zone"] = {
        "polygon": settings.zone.polygon,
        "direction": settings.zone.direction,
    }
    out["model"] = {
        "path": settings.model.path,
    }
    return out


@router.post("/devices/{device_id}/pair")
def pair_device(device_id: str, payload: PairDeviceRequest, request: Request) -> dict[str, Any]:
    return _store(request).pair_device(device_id=device_id, code=payload.code)


@router.post("/pair-codes")
def create_pair_code(payload: PairCodeCreateRequest, request: Request) -> dict[str, Any]:
    return _store(request).create_pair_code(line_id=payload.line_id, ttl_seconds=payload.ttl_seconds)


@router.post("/ui/runs/start", response_class=HTMLResponse)
def ui_start_run(
    request: Request,
    line_id: str = Form(...),
    device_id: str = Form(...),
    run_id: str = Form(""),
) -> str:
    rid = _store(request).start_run(line_id=line_id, device_id=device_id, run_id=run_id or None)
    return f"<div class='ok'>Started run <code>{rid}</code>.</div>"


@router.post("/ui/runs/stop", response_class=HTMLResponse)
def ui_stop_run(request: Request, run_id: str = Form(...)) -> str:
    stopped = _store(request).stop_run(run_id)
    if stopped:
        return f"<div class='ok'>Stopped run <code>{run_id}</code>.</div>"
    return f"<div class='warn'>Run <code>{run_id}</code> was already stopped or missing.</div>"


@router.post("/ui/pair-codes", response_class=HTMLResponse)
def ui_create_pair_code(
    request: Request,
    line_id: str = Form(...),
    ttl_seconds: int = Form(600),
) -> str:
    data = _store(request).create_pair_code(line_id=line_id, ttl_seconds=ttl_seconds)
    return (
        "<div class='ok'>"
        f"Pair code: <code>{data['code']}</code> "
        f"(expires {data['expires_at']})"
        "</div>"
    )


@router.post("/ui/sku/upload-video", response_class=HTMLResponse)
async def ui_upload_sku_video(
    request: Request,
    sku: str = Form(...),
    video: UploadFile = File(...),
) -> str:
    """Upload SKU enrollment video and auto-name under data/raw/videos/{sku}/{sku}_{n}.MOV."""
    if video.filename is None:
        return "<div class='warn'>No file selected.</div>"

    try:
        normalized_sku = normalize_sku_name(sku)
    except ValueError as exc:
        return f"<div class='warn'>{exc}</div>"

    target_path = next_enrollment_video_path(
        raw_videos_root=Path("data/raw/videos"),
        sku=normalized_sku,
        suffix=".MOV",
    )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("wb") as f:
        while True:
            chunk = await video.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    await video.close()

    _store(request).upsert_sku(
        sku_id=normalized_sku,
        name=normalized_sku,
        profile_path=f"data/profiles/{normalized_sku}",
        threshold=None,
        metadata={"source": "ui_video_upload"},
    )

    return (
        "<div class='ok'>"
        f"Uploaded video for <code>{normalized_sku}</code> to "
        f"<code>{target_path}</code>"
        "</div>"
    )
