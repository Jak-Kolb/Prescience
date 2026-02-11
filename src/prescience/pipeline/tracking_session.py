"""In-process tracking sessions for browser MJPEG preview."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
import yaml

from prescience.config import AppSettings, load_settings, parse_source
from prescience.events.emitter import EventEmitter, SequenceCounter
from prescience.events.schemas import AlertEvent, CountEvent, HeartbeatEvent, utc_now
from prescience.pipeline.zone_count import ZoneCrossCounter, draw_zone_overlay
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


@dataclass
class TrackingSession:
    """One running camera tracking session with latest JPEG buffer."""

    session_id: str
    sku_id: str
    source: str
    line_id: str
    device_id: str
    model_path: str
    zone_config_path: Path
    config_path: Path
    event_endpoint: str | None
    run_id: str | None = None
    status: str = "idle"
    message: str = ""
    started_at: str | None = None
    stopped_at: str | None = None
    last_error: str | None = None
    total_count: int = 0
    counts_by_sku: dict[str, int] = field(default_factory=dict)
    frame_index: int = 0
    fps: float = 0.0
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _latest_jpeg: bytes | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self.status = "running"
        self.started_at = utc_now().isoformat()
        self._thread = threading.Thread(target=self._run_loop, name=f"tracking-{self.session_id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.status = "stopped"
        self.stopped_at = utc_now().isoformat()

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def snapshot_status(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "sku_id": self.sku_id,
            "source": self.source,
            "line_id": self.line_id,
            "device_id": self.device_id,
            "model_path": self.model_path,
            "status": self.status,
            "message": self.message,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "last_error": self.last_error,
            "total_count": self.total_count,
            "counts_by_sku": dict(self.counts_by_sku),
            "frame_index": self.frame_index,
            "fps": self.fps,
        }

    def _publish_frame(self, frame: np.ndarray) -> None:
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            return
        with self._lock:
            self._latest_jpeg = bytes(buf.tobytes())

    def _run_loop(self) -> None:
        settings = load_settings(self.config_path)
        polygon, direction = _load_zone_from_yaml(self.zone_config_path, settings)

        tracker = YoloTracker(
            model_path=self.model_path,
            conf=settings.tracker.conf,
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
        endpoint = self.event_endpoint if self.event_endpoint is not None else settings.events.endpoint
        emitter = EventEmitter(
            endpoint=endpoint,
            fallback_jsonl_path=Path(settings.events.local_jsonl_path),
            timeout_seconds=settings.events.request_timeout_seconds,
        )
        seq = SequenceCounter()

        cap = cv2.VideoCapture(parse_source(self.source))
        if not cap.isOpened():
            self.status = "failed"
            self.last_error = f"Could not open source: {self.source}"
            return

        start_time = time.time()
        last_heartbeat = start_time
        last_count_ts = None
        last_alert_at: dict[str, float] = {}
        self.message = "Tracking active"

        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                self.frame_index += 1
                now_s = time.time()
                elapsed = max(now_s - start_time, 1e-6)
                self.fps = self.frame_index / elapsed
                frame_ts = utc_now()

                tracked = tracker.track(frame)
                tracked_by_id = {int(t.track_id): t for t in tracked if t.track_id is not None}
                decisions = counter.update(tracked=tracked, frame_idx=self.frame_index, now_seconds=now_s)

                for decision in decisions:
                    tracked_det = tracked_by_id.get(decision.track_id)
                    if tracked_det is None:
                        continue
                    self.total_count += 1
                    self.counts_by_sku[self.sku_id] = self.counts_by_sku.get(self.sku_id, 0) + 1

                    event = CountEvent(
                        seq=seq.next(),
                        timestamp=utc_now(),
                        frame_ts=frame_ts,
                        line_id=self.line_id,
                        device_id=self.device_id,
                        run_id=self.run_id,
                        sku_id=self.sku_id,
                        confidence=tracked_det.confidence,
                        track_id=decision.track_id,
                        count_delta=1,
                        counts_total_overall=self.total_count,
                        counts_total_by_sku=dict(self.counts_by_sku),
                    )
                    emitter.emit(event)
                    last_count_ts = event.timestamp

                brightness, blur_score = _quality_metrics(frame)
                alert_cooldown = settings.quality.alert_cooldown_seconds
                if brightness < settings.quality.min_brightness:
                    last = last_alert_at.get("low_brightness", -1e9)
                    if (now_s - last) >= alert_cooldown:
                        emitter.emit(
                            AlertEvent(
                                seq=seq.next(),
                                timestamp=utc_now(),
                                frame_ts=frame_ts,
                                line_id=self.line_id,
                                device_id=self.device_id,
                                run_id=self.run_id,
                                code="low_brightness",
                                severity="warning",
                                message="Frame brightness below configured minimum",
                                details={"brightness": brightness, "threshold": settings.quality.min_brightness},
                            )
                        )
                        last_alert_at["low_brightness"] = now_s

                if blur_score < settings.quality.min_blur:
                    last = last_alert_at.get("blur", -1e9)
                    if (now_s - last) >= alert_cooldown:
                        emitter.emit(
                            AlertEvent(
                                seq=seq.next(),
                                timestamp=utc_now(),
                                frame_ts=frame_ts,
                                line_id=self.line_id,
                                device_id=self.device_id,
                                run_id=self.run_id,
                                code="blur",
                                severity="warning",
                                message="Frame blur score below configured minimum",
                                details={"blur": blur_score, "threshold": settings.quality.min_blur},
                            )
                        )
                        last_alert_at["blur"] = now_s

                if (now_s - last_heartbeat) >= settings.events.heartbeat_interval_seconds:
                    emitter.emit(
                        HeartbeatEvent(
                            seq=seq.next(),
                            timestamp=utc_now(),
                            frame_ts=frame_ts,
                            line_id=self.line_id,
                            device_id=self.device_id,
                            run_id=self.run_id,
                            fps=self.fps,
                            uptime_s=elapsed,
                            brightness=brightness,
                            blur_score=blur_score,
                            last_count_ts=last_count_ts,
                        )
                    )
                    last_heartbeat = now_s

                annotated = draw_zone_overlay(frame, polygon)
                for tracked_det in tracked:
                    x1, y1, x2, y2 = tracked_det.box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (80, 170, 255), 2)
                    label = f"ID {tracked_det.track_id} {tracked_det.confidence:.2f}"
                    cv2.putText(
                        annotated,
                        label,
                        (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (80, 170, 255),
                        2,
                    )
                cv2.putText(
                    annotated,
                    f"{self.sku_id}: {self.total_count}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 220, 120),
                    2,
                )
                self._publish_frame(annotated)

            self.status = "stopped" if self._stop.is_set() else "ended"
            self.message = "Tracking stopped"
        except Exception as exc:  # noqa: BLE001
            self.status = "failed"
            self.last_error = str(exc)
            self.message = "Tracking failed"
        finally:
            cap.release()
            self.stopped_at = utc_now().isoformat()


class TrackingSessionManager:
    """Registry for active UI tracking sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, TrackingSession] = {}
        self._lock = threading.Lock()

    def start_session(
        self,
        *,
        sku_id: str,
        source: str,
        line_id: str,
        device_id: str,
        model_path: str,
        zone_config_path: Path,
        config_path: Path,
        event_endpoint: str | None,
        run_id: str | None = None,
    ) -> TrackingSession:
        session_id = str(uuid4())
        session = TrackingSession(
            session_id=session_id,
            sku_id=sku_id,
            source=source,
            line_id=line_id,
            device_id=device_id,
            model_path=model_path,
            zone_config_path=zone_config_path,
            config_path=config_path,
            event_endpoint=event_endpoint,
            run_id=run_id,
        )
        with self._lock:
            self._sessions[session_id] = session
        session.start()
        return session

    def get_session(self, session_id: str) -> TrackingSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def stop_session(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            return False
        session.stop()
        return True

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            sessions = list(self._sessions.values())
        return [session.snapshot_status() for session in sessions]

    def stop_all(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            session.stop()
