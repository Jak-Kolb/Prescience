"""Edge runtime counting loop."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from prescience.config import AppSettings, load_settings, parse_source
from prescience.events.emitter import EventEmitter, SequenceCounter
from prescience.events.schemas import AlertEvent, CountEvent, HeartbeatEvent, utc_now
from prescience.pipeline.zone_count import ZoneCrossCounter, draw_zone_overlay
from prescience.profiles.io import load_all_profiles
from prescience.vision.embeddings import Embedder, build_embedder
from prescience.vision.matcher import match_embedding
from prescience.vision.tracker import YoloTracker


def _load_zone_from_yaml(path: Path, settings: AppSettings) -> tuple[list[tuple[int, int]], str]:
    if not path.exists():
        return settings.zone.polygon, settings.zone.direction

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    polygon_raw = raw.get("zone", {}).get("polygon", settings.zone.polygon)
    direction = raw.get("zone", {}).get("direction", settings.zone.direction)
    polygon = [tuple(map(int, point)) for point in polygon_raw]
    return polygon, direction


def _quality_metrics(frame: np.ndarray) -> tuple[float, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return brightness, blur_score


def _emit_quality_alert(
    emitter: EventEmitter,
    seq: SequenceCounter,
    line_id: str,
    device_id: str,
    run_id: str | None,
    frame_ts,
    code: str,
    message: str,
    details: dict[str, Any],
) -> None:
    event = AlertEvent(
        seq=seq.next(),
        timestamp=utc_now(),
        frame_ts=frame_ts,
        line_id=line_id,
        device_id=device_id,
        run_id=run_id,
        code=code,
        severity="warning",
        message=message,
        details=details,
    )
    emitter.emit(event)


def _infer_sku(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    embedder: Embedder | None,
    profiles,
) -> tuple[str, float]:
    if embedder is None or not profiles:
        return "UNKNOWN", 0.0

    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))

    if x2 <= x1 or y2 <= y1:
        return "UNKNOWN", 0.0

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return "UNKNOWN", 0.0

    embedding = embedder.encode(crop)
    result = match_embedding(embedding, profiles)
    return result.sku_id, result.score


def run_count_agent(
    source: str,
    model_path: str,
    zone_config_path: Path,
    line_id: str,
    device_id: str,
    run_id: str | None,
    event_endpoint: str | None,
    jsonl_log_path: Path,
    profiles_root: Path,
    config_path: Path,
    conf_override: float | None = None,
) -> None:
    """Run counter runtime with detection, tracking, counting, and event emission."""
    settings = load_settings(config_path)

    polygon, direction = _load_zone_from_yaml(zone_config_path, settings)

    tracker = YoloTracker(
        model_path=model_path,
        conf=(conf_override if conf_override is not None else settings.tracker.conf),
        classes=settings.tracker.classes or None,
        tracker_cfg=settings.tracker.tracker_cfg,
    )

    counter = ZoneCrossCounter(
        polygon=polygon,
        direction=direction,
        min_separation_seconds=settings.counting.min_separation_seconds,
        track_cooldown_seconds=settings.counting.track_cooldown_seconds,
        max_track_idle_frames=settings.counting.max_track_idle_frames,
    )

    profiles = load_all_profiles(profiles_root)
    embedder = build_embedder(settings.embedding.backbone) if profiles else None

    endpoint = event_endpoint if event_endpoint is not None else settings.events.endpoint
    emitter = EventEmitter(
        endpoint=endpoint,
        fallback_jsonl_path=jsonl_log_path,
        timeout_seconds=settings.events.request_timeout_seconds,
    )
    seq = SequenceCounter()

    cap = cv2.VideoCapture(parse_source(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    totals_by_sku: dict[str, int] = {}
    total_overall = 0

    start_time = time.time()
    last_heartbeat = start_time
    last_alert_at: dict[str, float] = {}
    last_count_ts = None

    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            now_s = time.time()
            frame_ts = utc_now()

            tracked = tracker.track(frame)
            tracked_by_id = {int(t.track_id): t for t in tracked if t.track_id is not None}

            for tracked_det in tracked:
                x1, y1, x2, y2 = tracked_det.box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 170, 255), 2)
                label = f"ID {tracked_det.track_id} {tracked_det.confidence:.2f}"
                cv2.putText(frame, label, (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 170, 255), 2)

            decisions = counter.update(tracked=tracked, frame_idx=frame_idx, now_seconds=now_s)
            for decision in decisions:
                tracked_det = tracked_by_id.get(decision.track_id)
                if tracked_det is None:
                    continue

                sku_id, score = _infer_sku(frame, tracked_det.box, embedder, profiles)
                totals_by_sku[sku_id] = totals_by_sku.get(sku_id, 0) + 1
                total_overall += 1

                event = CountEvent(
                    seq=seq.next(),
                    timestamp=utc_now(),
                    frame_ts=frame_ts,
                    line_id=line_id,
                    device_id=device_id,
                    run_id=run_id,
                    sku_id=sku_id,
                    confidence=score,
                    track_id=decision.track_id,
                    count_delta=1,
                    counts_total_overall=total_overall,
                    counts_total_by_sku=dict(totals_by_sku),
                )
                emitter.emit(event)
                last_count_ts = event.timestamp

            brightness, blur_score = _quality_metrics(frame)
            alert_cooldown = settings.quality.alert_cooldown_seconds

            if brightness < settings.quality.min_brightness:
                last = last_alert_at.get("low_brightness", -1e9)
                if (now_s - last) >= alert_cooldown:
                    _emit_quality_alert(
                        emitter=emitter,
                        seq=seq,
                        line_id=line_id,
                        device_id=device_id,
                        run_id=run_id,
                        frame_ts=frame_ts,
                        code="low_brightness",
                        message="Frame brightness below configured minimum",
                        details={"brightness": brightness, "threshold": settings.quality.min_brightness},
                    )
                    last_alert_at["low_brightness"] = now_s

            if blur_score < settings.quality.min_blur:
                last = last_alert_at.get("blur", -1e9)
                if (now_s - last) >= alert_cooldown:
                    _emit_quality_alert(
                        emitter=emitter,
                        seq=seq,
                        line_id=line_id,
                        device_id=device_id,
                        run_id=run_id,
                        frame_ts=frame_ts,
                        code="blur",
                        message="Frame blur score below configured minimum",
                        details={"blur": blur_score, "threshold": settings.quality.min_blur},
                    )
                    last_alert_at["blur"] = now_s

            if (now_s - last_heartbeat) >= settings.events.heartbeat_interval_seconds:
                elapsed = max(now_s - start_time, 1e-6)
                fps = frame_idx / elapsed
                hb = HeartbeatEvent(
                    seq=seq.next(),
                    timestamp=utc_now(),
                    frame_ts=frame_ts,
                    line_id=line_id,
                    device_id=device_id,
                    run_id=run_id,
                    fps=float(fps),
                    uptime_s=elapsed,
                    brightness=brightness,
                    blur_score=blur_score,
                    last_count_ts=last_count_ts,
                )
                emitter.emit(hb)
                last_heartbeat = now_s

            draw_zone_overlay(frame, polygon=polygon, direction=direction, total_count=counter.total_count)

            cv2.imshow("Prescience Counter", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        emitter.close()


def run(source: int | str = 0, model_path: str = "yolov8n.pt", conf: float = 0.35) -> None:
    """Backward-compatible wrapper used by legacy scripts."""
    settings = load_settings(Path("configs/default.yaml"))
    run_count_agent(
        source=str(source),
        model_path=model_path,
        zone_config_path=Path("configs/default.yaml"),
        line_id=settings.line.line_id,
        device_id="device-1",
        run_id=None,
        event_endpoint=settings.events.endpoint,
        jsonl_log_path=Path(settings.events.local_jsonl_path),
        profiles_root=Path(settings.profiles.root),
        config_path=Path("configs/default.yaml"),
        conf_override=conf,
    )
