"""FastAPI routes for cloud API, UI workflows, SSE, and tracking sessions."""

from __future__ import annotations

import asyncio
import json
from html import escape
from pathlib import Path
from typing import Any

import cv2
import yaml
from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse

from prescience.cloud.jobs import UIJobRunner
from prescience.cloud.schemas import (
    OnboardingLabelRequest,
    PairCodeCreateRequest,
    PairDeviceRequest,
    RunStartRequest,
    SKUUpsertRequest,
    TrackingStartRequest,
    ZoneConfigRequest,
)
from prescience.cloud.store import CloudStore
from prescience.cloud.stream import SSEBroadcaster, encode_sse
from prescience.events.schemas import AnyEvent
from prescience.pipeline.enroll import next_enrollment_video_path, normalize_sku_name
from prescience.pipeline.tracking_session import TrackingSessionManager
from prescience.pipeline.web_onboarding import next_model_version_for_sku, save_browser_label

router = APIRouter()


def _store(request: Request) -> CloudStore:
    return request.app.state.store


def _stream_bus(request: Request) -> SSEBroadcaster:
    return request.app.state.stream_bus


def _jobs(request: Request) -> UIJobRunner:
    return request.app.state.job_runner


def _tracking(request: Request) -> TrackingSessionManager:
    return request.app.state.tracking_manager


def _zone_config_path(line_id: str) -> Path:
    clean = line_id.strip()
    if clean.startswith("line-"):
        return Path(f"configs/{clean}.yaml")
    return Path(f"configs/line-{clean}.yaml")


def _session_summary(session: dict[str, Any]) -> dict[str, Any]:
    seed = session.get("seed_candidates", [])
    approval = session.get("approval_candidates", [])
    return {
        "session_id": session["session_id"],
        "sku_id": session["sku_id"],
        "state": session["state"],
        "version_tag": session["version_tag"],
        "mode": session["mode"],
        "latest_job_id": session.get("latest_job_id"),
        "seed": {
            "total": len(seed),
            "labeled": sum(1 for item in seed if item.get("status") in {"positive", "negative", "skipped"}),
            "pending": sum(1 for item in seed if item.get("status", "pending") == "pending"),
        },
        "approval": {
            "total": len(approval),
            "labeled": sum(1 for item in approval if item.get("status") in {"positive", "negative", "skipped"}),
            "pending": sum(1 for item in approval if item.get("status", "pending") == "pending"),
        },
    }


def _update_candidate_status(
    candidates: list[dict[str, Any]],
    *,
    frame_name: str,
    status: str,
    boxes: list[dict[str, int]],
) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    for item in candidates:
        if item.get("frame_name") == frame_name:
            patched = dict(item)
            patched["status"] = status
            patched["boxes"] = boxes
            updated.append(patched)
        else:
            updated.append(item)
    return updated


@router.post("/events")
async def post_event(event: AnyEvent, request: Request) -> dict[str, Any]:
    store = _store(request)
    stream_bus = _stream_bus(request)
    result = store.ingest_event(event)
    if result.status == "inserted":
        await stream_bus.publish({"line_id": result.line_id, "run_id": result.run_id, "event_type": event.event_type})
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

                if message.get("line_id") != line_id:
                    continue
                live = store.get_line_live(line_id=line_id)
                yield encode_sse({"type": "update", "live": live})
        finally:
            await stream_bus.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.get("/stream/jobs")
async def stream_jobs(request: Request, sku_id: str | None = Query(None)) -> StreamingResponse:
    store = _store(request)
    retry_ms = int(request.app.state.settings.cloud.sse_retry_ms)

    async def event_generator():
        last_digest: str | None = None
        snapshot = store.get_ui_snapshot(sku_id=sku_id)
        last_digest = json.dumps(snapshot, sort_keys=True)
        yield encode_sse({"type": "snapshot", "snapshot": snapshot}, retry_ms=retry_ms)

        while True:
            if await request.is_disconnected():
                break
            current = store.get_ui_snapshot(sku_id=sku_id)
            digest = json.dumps(current, sort_keys=True)
            if digest != last_digest:
                last_digest = digest
                yield encode_sse({"type": "update", "snapshot": current})
            else:
                yield ": keep-alive\n\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
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


@router.delete("/skus/{sku_id}")
def delete_sku(sku_id: str, request: Request) -> dict[str, Any]:
    try:
        normalized = normalize_sku_name(sku_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _store(request).delete_sku_and_artifacts(sku_id=normalized)


@router.get("/devices/{device_id}/config")
def get_device_config(device_id: str, request: Request) -> dict[str, Any]:
    out = _store(request).get_device_config(device_id)
    settings = request.app.state.settings
    out["zone"] = {"polygon": settings.zone.polygon, "direction": settings.zone.direction}
    out["model"] = {"path": settings.model.path}
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
    return "<div class='ok'>" f"Pair code: <code>{data['code']}</code> " f"(expires {data['expires_at']})" "</div>"


async def _save_upload(video: UploadFile, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("wb") as f:
        while True:
            chunk = await video.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    await video.close()


@router.post("/ui/sku/upload-video", response_class=HTMLResponse)
async def ui_upload_sku_video(
    request: Request,
    sku: str = Form(...),
    video: UploadFile = File(...),
) -> str:
    if video.filename is None:
        return "<div class='warn'>No file selected.</div>"
    try:
        normalized_sku = normalize_sku_name(sku)
    except ValueError as exc:
        return f"<div class='warn'>{escape(str(exc))}</div>"
    target_path = next_enrollment_video_path(
        raw_videos_root=Path("data/raw/videos"),
        sku=normalized_sku,
        suffix=".MOV",
    )
    await _save_upload(video=video, target_path=target_path)
    _store(request).upsert_sku(
        sku_id=normalized_sku,
        name=normalized_sku,
        profile_path=f"data/profiles/{normalized_sku}",
        threshold=None,
        metadata={"source": "ui_video_upload"},
    )
    return "<div class='ok'>" f"Uploaded video for <code>{normalized_sku}</code> to <code>{target_path}</code>" "</div>"


@router.post("/ui/sku/enroll")
async def ui_sku_enroll(
    request: Request,
    sku: str = Form(...),
    video: UploadFile = File(...),
) -> RedirectResponse:
    if video.filename is None:
        raise HTTPException(status_code=422, detail="No file selected")
    normalized_sku = normalize_sku_name(sku)
    target_path = next_enrollment_video_path(
        raw_videos_root=Path("data/raw/videos"),
        sku=normalized_sku,
        suffix=".MOV",
    )
    await _save_upload(video=video, target_path=target_path)
    store = _store(request)
    version_num = next_model_version_for_sku(normalized_sku)
    session = store.create_onboarding_session(
        sku_id=normalized_sku,
        version_tag=f"v{version_num}",
        mode="quick",
        state="extracting",
        seed_candidates=[],
        approval_candidates=[],
    )
    job = store.create_ui_job(
        sku_id=normalized_sku,
        job_type="extract_prepare_seed",
        status="queued",
        step="extracting",
        progress=0.0,
        message="Queued extraction job",
        payload={
            "session_id": session["session_id"],
            "video_path": str(target_path),
            "append": False,
            "seed_per_bin": 4,
            "approve_per_bin": 5,
            "target_frames": 150,
            "blur_min": 4.0,
            "dedupe_sim": 0.98,
        },
    )
    store.update_onboarding_session(session["session_id"], latest_job_id=job["job_id"])
    store.upsert_sku(
        sku_id=normalized_sku,
        name=normalized_sku,
        profile_path=f"data/profiles/{normalized_sku}",
        threshold=None,
        metadata={"source": "ui_enroll"},
    )
    _jobs(request).enqueue(job["job_id"])
    return RedirectResponse(url=f"/ui/onboarding/{session['session_id']}", status_code=303)


@router.post("/ui/sku/{sku_id}/append-train")
async def ui_sku_append_train(
    sku_id: str,
    request: Request,
    video: UploadFile = File(...),
) -> RedirectResponse:
    if video.filename is None:
        raise HTTPException(status_code=422, detail="No file selected")
    normalized_sku = normalize_sku_name(sku_id)
    target_path = next_enrollment_video_path(
        raw_videos_root=Path("data/raw/videos"),
        sku=normalized_sku,
        suffix=".MOV",
    )
    await _save_upload(video=video, target_path=target_path)

    store = _store(request)
    version_num = next_model_version_for_sku(normalized_sku)
    session = store.create_onboarding_session(
        sku_id=normalized_sku,
        version_tag=f"v{version_num}",
        mode="quick",
        state="extracting",
        seed_candidates=[],
        approval_candidates=[],
    )
    job = store.create_ui_job(
        sku_id=normalized_sku,
        job_type="extract_prepare_seed",
        status="queued",
        step="extracting",
        progress=0.0,
        message="Queued append extraction job",
        payload={
            "session_id": session["session_id"],
            "video_path": str(target_path),
            "append": True,
            "seed_per_bin": 4,
            "approve_per_bin": 5,
            "target_frames": 150,
            "blur_min": 4.0,
            "dedupe_sim": 0.98,
        },
    )
    store.update_onboarding_session(session["session_id"], latest_job_id=job["job_id"])
    _jobs(request).enqueue(job["job_id"])
    return RedirectResponse(url=f"/ui/onboarding/{session['session_id']}", status_code=303)


@router.get("/ui/onboarding/{session_id}", response_class=HTMLResponse)
def onboarding_page(session_id: str, request: Request) -> HTMLResponse:
    templates = request.app.state.templates
    session = _store(request).get_onboarding_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Onboarding session not found")
    context = {
        "session": session,
        "summary": _session_summary(session),
    }
    return templates.TemplateResponse(request=request, name="onboarding.html", context=context)


@router.get("/api/onboarding/{session_id}")
def api_get_onboarding_session(session_id: str, request: Request) -> dict[str, Any]:
    session = _store(request).get_onboarding_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Onboarding session not found")
    return {
        "session": session,
        "summary": _session_summary(session),
        "full_train_ready": _store(request).sku_full_train_ready(session["sku_id"]),
    }


@router.get("/api/onboarding/{session_id}/frame/{frame_name}")
def api_get_onboarding_frame(session_id: str, frame_name: str, request: Request) -> FileResponse:
    session = _store(request).get_onboarding_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Onboarding session not found")
    safe_name = Path(frame_name).name
    frame_path = Path(f"data/derived/frames/{session['sku_id']}/frames") / safe_name
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Frame not found")
    return FileResponse(frame_path)


@router.post("/api/onboarding/{session_id}/labels")
def api_save_onboarding_label(session_id: str, payload: OnboardingLabelRequest, request: Request) -> dict[str, Any]:
    store = _store(request)
    session = store.get_onboarding_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Onboarding session not found")
    sku = session["sku_id"]
    boxes = [{"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2} for b in payload.boxes]
    save_browser_label(
        sku=sku,
        frame_name=payload.frame_name,
        status=payload.status,
        boxes=boxes,
        class_id=0,
        allow_negatives=True,
    )

    if payload.stage == "approval":
        approval = _update_candidate_status(
            session.get("approval_candidates", []),
            frame_name=payload.frame_name,
            status=payload.status,
            boxes=boxes,
        )
        session = store.update_onboarding_session(session_id, approval_candidates=approval) or session
    else:
        seed = _update_candidate_status(
            session.get("seed_candidates", []),
            frame_name=payload.frame_name,
            status=payload.status,
            boxes=boxes,
        )
        session = store.update_onboarding_session(session_id, seed_candidates=seed) or session

    return {"ok": True, "summary": _session_summary(session)}


@router.post("/api/onboarding/{session_id}/seed/complete")
def api_onboarding_seed_complete(session_id: str, request: Request) -> dict[str, Any]:
    store = _store(request)
    session = store.get_onboarding_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Onboarding session not found")
    job = store.create_ui_job(
        sku_id=session["sku_id"],
        job_type="train_stage1",
        status="queued",
        step="train_stage1",
        progress=0.0,
        message="Queued stage1 training",
        payload={"session_id": session_id, "approve_per_bin": 5, "conf_propose": 0.10},
    )
    session = store.update_onboarding_session(
        session_id,
        state="train_stage1",
        latest_job_id=job["job_id"],
    ) or session
    _jobs(request).enqueue(job["job_id"])
    return {"ok": True, "job": job, "summary": _session_summary(session)}


@router.get("/api/onboarding/{session_id}/approvals")
def api_onboarding_approvals(session_id: str, request: Request) -> dict[str, Any]:
    session = _store(request).get_onboarding_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Onboarding session not found")
    return {"approval_candidates": session.get("approval_candidates", []), "summary": _session_summary(session)}


@router.post("/api/onboarding/{session_id}/approvals/complete")
def api_onboarding_approvals_complete(session_id: str, request: Request) -> dict[str, Any]:
    store = _store(request)
    session = store.get_onboarding_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Onboarding session not found")
    job = store.create_ui_job(
        sku_id=session["sku_id"],
        job_type="train_stage2",
        status="queued",
        step="train_stage2",
        progress=0.0,
        message="Queued stage2 training",
        payload={"session_id": session_id},
    )
    session = store.update_onboarding_session(
        session_id,
        state="train_stage2",
        latest_job_id=job["job_id"],
    ) or session
    _jobs(request).enqueue(job["job_id"])
    return {"ok": True, "job": job, "summary": _session_summary(session)}


@router.post("/ui/sku/{sku_id}/train/full", response_class=HTMLResponse)
def ui_full_train(sku_id: str, request: Request) -> str:
    store = _store(request)
    normalized_sku = normalize_sku_name(sku_id)
    if not store.sku_full_train_ready(normalized_sku):
        return (
            "<div class='warn'>"
            f"Full Train requires at least <code>{normalized_sku}_v2</code> to exist."
            "</div>"
        )
    job = store.create_ui_job(
        sku_id=normalized_sku,
        job_type="full_train",
        status="queued",
        step="full_train",
        progress=0.0,
        message="Queued full train",
        payload={},
    )
    _jobs(request).enqueue(job["job_id"])
    return "<div class='ok'>" f"Queued full training job <code>{job['job_id']}</code> for <code>{normalized_sku}</code>." "</div>"


@router.post("/ui/skus/delete", response_class=HTMLResponse)
def ui_delete_sku(request: Request, sku_id: str = Form(...)) -> str:
    try:
        normalized_sku = normalize_sku_name(sku_id)
    except ValueError as exc:
        return f"<div class='warn'>{escape(str(exc))}</div>"

    result = _store(request).delete_sku_and_artifacts(sku_id=normalized_sku)
    if result["errors_count"] > 0:
        first = result["errors"][0]
        return (
            "<div class='warn'>"
            f"Deleted <code>{normalized_sku}</code> partially "
            f"({result['deleted_count']} paths), but hit errors. "
            f"First error: <code>{escape(first['path'])}</code> "
            f"({escape(first['error'])})"
            "</div>"
        )

    if not result["deleted_db_row"] and result["deleted_count"] == 0:
        return f"<div class='warn'>SKU <code>{normalized_sku}</code> was not found.</div>"

    return (
        "<div class='ok'>"
        f"Deleted SKU <code>{normalized_sku}</code> "
        f"and removed {result['deleted_count']} local artifact paths."
        "</div>"
    )


@router.get("/api/zone/{line_id}")
def api_get_zone(line_id: str, request: Request) -> dict[str, Any]:
    settings = request.app.state.settings
    path = _zone_config_path(line_id)
    if not path.exists():
        return {
            "line": {"line_id": line_id},
            "zone": {
                "polygon": settings.zone.polygon,
                "direction": settings.zone.direction,
            },
        }
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return raw


@router.post("/api/zone/{line_id}")
def api_save_zone(line_id: str, payload: ZoneConfigRequest) -> dict[str, Any]:
    path = _zone_config_path(line_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"line": {"line_id": line_id}, "zone": {"polygon": payload.polygon, "direction": payload.direction}}
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return {"ok": True, "path": str(path), "zone": data["zone"]}


@router.get("/api/zone/frame")
def api_zone_frame(source: str = Query("0")) -> StreamingResponse:
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail=f"Could not open source: {source}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(status_code=400, detail="Could not read frame from source")
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode frame")
    return StreamingResponse(iter([bytes(encoded.tobytes())]), media_type="image/jpeg")


@router.get("/ui/tracking", response_class=HTMLResponse)
def tracking_page(
    request: Request,
    sku_id: str = Query(""),
    line_id: str = Query("line-1"),
) -> HTMLResponse:
    templates = request.app.state.templates
    skus = _store(request).list_skus()
    return templates.TemplateResponse(
        request=request,
        name="tracking.html",
        context={"sku_id": sku_id, "line_id": line_id, "skus": skus},
    )


@router.post("/ui/tracking/start")
def ui_tracking_start(payload: TrackingStartRequest, request: Request) -> dict[str, Any]:
    store = _store(request)
    normalized_sku = normalize_sku_name(payload.sku_id)
    model_path = payload.model_path
    if model_path is None:
        versions = store.list_model_versions_for_sku(normalized_sku)
        if not versions:
            raise HTTPException(status_code=422, detail=f"No stable model versions found for SKU {normalized_sku}")
        model_path = f"data/models/yolo/{normalized_sku}_v{versions[-1]}/best.pt"

    zone_path = _zone_config_path(payload.line_id)
    session = _tracking(request).start_session(
        sku_id=normalized_sku,
        source=payload.source,
        line_id=payload.line_id,
        device_id=payload.device_id,
        model_path=model_path,
        zone_config_path=zone_path,
        config_path=Path("configs/default.yaml"),
        event_endpoint=payload.event_endpoint,
        run_id=payload.run_id,
    )
    return {
        "ok": True,
        "session_id": session.session_id,
        "stream_url": f"/ui/tracking/stream/{session.session_id}",
        "status_url": f"/api/tracking/{session.session_id}",
        "stop_url": f"/ui/tracking/{session.session_id}/stop",
    }


@router.get("/ui/tracking/stream/{session_id}")
async def ui_tracking_stream(session_id: str, request: Request) -> StreamingResponse:
    manager = _tracking(request)

    async def frame_generator():
        boundary = b"--frame\r\n"
        while True:
            if await request.is_disconnected():
                break
            session = manager.get_session(session_id)
            if session is None:
                break
            frame = session.latest_jpeg()
            if frame is None:
                await asyncio.sleep(0.05)
                continue
            yield boundary
            yield b"Content-Type: image/jpeg\r\n\r\n"
            yield frame
            yield b"\r\n"
            await asyncio.sleep(0.03)

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/api/tracking/{session_id}")
def api_tracking_status(session_id: str, request: Request) -> dict[str, Any]:
    session = _tracking(request).get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Tracking session not found")
    return session.snapshot_status()


@router.post("/ui/tracking/{session_id}/stop")
def ui_tracking_stop(session_id: str, request: Request) -> dict[str, Any]:
    stopped = _tracking(request).stop_session(session_id)
    return {"ok": stopped, "session_id": session_id}
