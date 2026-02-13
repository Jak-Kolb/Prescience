"""Background UI job runner for onboarding and retraining workflows."""

from __future__ import annotations

import json
import queue
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from prescience.cloud.store import CloudStore
from prescience.config import AppSettings
from prescience.pipeline.enroll import extract_frames_for_sku
from prescience.pipeline.enroll import resolve_base_model_for_sku
from prescience.pipeline.web_onboarding import (
    auto_label_for_stage2,
    build_detector_candidates,
    prepare_approval_candidates_with_gemini,
    prepare_seed_candidates_with_gemini,
    run_full_train_for_sku,
    train_stage1_for_session,
    train_stage2_for_session,
)


def _version_num_from_tag(version_tag: str) -> int:
    raw = version_tag.strip().lower()
    if raw.startswith("v"):
        raw = raw[1:]
    value = int(raw)
    if value <= 0:
        raise ValueError("version number must be > 0")
    return value


class UIJobRunner:
    """Single-worker FIFO queue for browser onboarding and training jobs."""

    def __init__(self, store: CloudStore, settings: AppSettings | None = None) -> None:
        self.store = store
        self.settings = settings
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start background worker if not already running."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, name="prescience-ui-jobs", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop worker loop and join thread."""
        self._stop.set()
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def enqueue(self, job_id: str) -> None:
        """Queue a job by id."""
        self._queue.put(job_id)

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                continue
            self._run_job(item)

    def _set_failed(self, job_id: str, exc: Exception) -> None:
        self.store.update_ui_job(
            job_id,
            status="failed",
            message=str(exc),
            error={"traceback": traceback.format_exc()},
        )

    def _run_job(self, job_id: str) -> None:
        job = self.store.get_ui_job(job_id)
        if job is None:
            return
        self.store.update_ui_job(job_id, status="running", message="Job started")
        try:
            job_type = str(job["type"])
            if job_type == "extract_prepare_seed":
                self._job_extract_prepare_seed(job)
            elif job_type == "train_stage1":
                self._job_train_stage1(job)
            elif job_type == "train_stage2":
                self._job_train_stage2(job)
            elif job_type == "full_train":
                self._job_full_train(job)
            else:
                raise RuntimeError(f"Unsupported job type: {job_type}")
        except Exception as exc:  # noqa: BLE001
            self._set_failed(job_id, exc)

    @staticmethod
    def _progress_from_payload(payload: dict[str, Any]) -> float | None:
        if "progress" in payload and payload["progress"] is not None:
            try:
                return float(payload["progress"])
            except (TypeError, ValueError):
                return None
        epoch = payload.get("epoch")
        total = payload.get("total_epochs")
        if epoch is None or total in (None, 0):
            return None
        try:
            return max(0.0, min(100.0, (float(epoch) / float(total)) * 100.0))
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    def _vlm_setting(self, name: str, default: Any) -> Any:
        cfg = getattr(self.settings, "onboarding_vlm", None)
        if cfg is None:
            return default
        return getattr(cfg, name, default)

    @staticmethod
    def _clamp_conf(value: float, *, lower: float = 0.005, upper: float = 0.95) -> float:
        return max(lower, min(upper, value))

    def _latest_prior_model_mean_top_confidence(self, sku: str, target_version: int) -> float | None:
        versions = self.store.list_model_versions_for_sku(sku)
        prior = [version for version in versions if version < target_version]
        if not prior:
            return None
        latest = max(prior)
        metrics_path = Path(f"data/models/yolo/{sku}_v{latest}/eval/metrics.json")
        if not metrics_path.exists():
            return None
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            prediction_stats = payload.get("prediction_stats", {})
            value = prediction_stats.get("mean_top_confidence")
            if value is None:
                # Backward compatibility for older eval artifacts.
                value = prediction_stats.get("max_confidence")
            if value is None:
                return None
            return float(value)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None

    def _resolve_append_auto_label_conf(self, *, sku: str, target_version: int, payload: dict[str, Any]) -> float:
        base_conf = float(payload.get("detector_conf_for_auto_label", self._vlm_setting("detector_conf_for_auto_label", 0.02)))
        base_conf = self._clamp_conf(base_conf)
        # First append retrain should stay fixed at configured baseline.
        if target_version <= 2:
            return base_conf

        prior_mean_top = self._latest_prior_model_mean_top_confidence(sku=sku, target_version=target_version)
        if prior_mean_top is None:
            return base_conf
        # Slightly below prior mean top confidence, with safety clamps.
        dynamic_conf = self._clamp_conf(prior_mean_top - 0.02, lower=0.01, upper=0.9)
        return dynamic_conf

    @staticmethod
    def _session_context(session: dict[str, Any]) -> dict[str, Any]:
        ctx = session.get("session_context")
        if isinstance(ctx, dict):
            return dict(ctx)
        return {}

    def _manual_candidates_from_detector(
        self,
        *,
        sku: str,
        model_path: str,
        target_count: int,
        conf: float,
        sections: int,
        append_video_path: str | None,
        max_boxes: int,
    ) -> list[dict[str, Any]]:
        detector = build_detector_candidates(
            sku=sku,
            model_path=model_path,
            target_count=target_count,
            conf=conf,
            sections=sections,
            append_video_path=append_video_path,
            max_proposals_per_frame=max_boxes,
        )
        return [
            {
                "frame_name": item.frame_name,
                "status": "pending",
                "boxes": [dict(box) for box in item.boxes],
                "proposals": [dict(box) for box in item.boxes],
                "source": "detector_fallback",
                "needs_review": True,
                "reason": "manual_fallback_required",
            }
            for item in detector
        ]

    def _move_manual_required(
        self,
        *,
        job: dict[str, Any],
        session: dict[str, Any],
        reason: str,
        stage1_model_path: str,
        approval_candidates: list[dict[str, Any]],
        payload_updates: dict[str, Any] | None = None,
    ) -> None:
        session_context = self._session_context(session)
        session_context["manual_fallback_reason"] = reason
        session_context["auto_mode_expected"] = bool(session_context.get("auto_mode_expected", True))

        self.store.update_onboarding_session(
            session["session_id"],
            state="manual_required",
            approval_candidates=approval_candidates,
            stage1_model_path=stage1_model_path,
            latest_job_id=job["job_id"],
            session_context=session_context,
        )
        payload = dict(job.get("payload", {}))
        payload.update(payload_updates or {})
        payload["manual_required"] = True
        payload["manual_reason"] = reason
        self.store.update_ui_job(
            job["job_id"],
            status="waiting_user",
            step="manual_required",
            progress=100.0,
            message=f"Cloud training unavailable. Manual review required: {reason}",
            payload=payload,
        )

    def _auto_label_with_retries(
        self,
        *,
        sku: str,
        model_path: str,
        object_description: str,
        target_count: int,
        conf: float,
        sections: int,
        append_video_path: str | None,
        enabled: bool,
        gemini_model: str,
        gemini_api_key_env: str,
        batch_size: int,
        max_boxes: int,
        retries: int,
        backoff_seconds: float,
        job_id: str,
        step: str,
    ):
        attempts = max(1, retries)
        last_summary = None
        last_reason = "unknown"
        for attempt in range(1, attempts + 1):
            summary = auto_label_for_stage2(
                sku=sku,
                model_path=model_path,
                object_description=object_description,
                target_count=target_count,
                conf=conf,
                sections=sections,
                append_video_path=append_video_path,
                enabled=enabled,
                gemini_model=gemini_model,
                gemini_api_key_env=gemini_api_key_env,
                batch_size=batch_size,
                max_boxes_approval=max_boxes,
            )
            last_summary = summary
            if summary.total == 0:
                return summary, "no_candidates"

            if summary.uncertain < summary.total:
                return summary, None

            if summary.decisions:
                last_reason = str(summary.decisions[0].get("reason", "gemini_unavailable"))
            else:
                last_reason = "gemini_unavailable"
            self.store.update_ui_job(
                job_id,
                step=step,
                message=f"Retrying Gemini validation ({attempt}/{attempts})",
                payload={"attempt": attempt, "attempts": attempts, "last_reason": last_reason},
            )
            if attempt < attempts:
                time.sleep(max(0.0, backoff_seconds) * attempt)
        return last_summary, last_reason

    def _job_extract_prepare_seed(self, job: dict[str, Any]) -> None:
        payload = job.get("payload", {})
        sku = str(job["sku_id"])
        session_id = str(payload["session_id"])
        video_path = Path(str(payload["video_path"]))
        append = bool(payload.get("append", False))
        session = self.store.get_onboarding_session(session_id)
        if session is None:
            raise RuntimeError(f"Session not found: {session_id}")
        version_num = _version_num_from_tag(str(session["version_tag"]))
        session_context = self._session_context(session)
        trust_mode_enabled = bool(self._vlm_setting("auto_after_trust", True))
        is_trusted = self.store.sku_is_trusted(sku)
        auto_mode_expected = trust_mode_enabled and (is_trusted or not append)
        session_context.setdefault("object_description", str(payload.get("object_description") or sku))
        session_context.setdefault("flow_kind", "append" if append else "initial")
        session_context["auto_mode_expected"] = auto_mode_expected
        session_context.setdefault("manual_fallback_reason", None)

        self.store.update_ui_job(job["job_id"], step="extracting", progress=5.0, message="Extracting enrollment frames")
        extract_frames_for_sku(
            video_path=video_path,
            sku=sku,
            target_frames=int(payload.get("target_frames", 150)),
            out_root=Path("data/derived/frames"),
            blur_min=float(payload.get("blur_min", 4.0)),
            dedupe_max_similarity=float(payload.get("dedupe_sim", 0.98)),
            append=append,
        )

        if append and bool(self.store.list_model_versions_for_sku(sku)):
            target_auto = int(
                payload.get(
                    "required_approval_count",
                    self._vlm_setting("append_auto_approval_count", 30),
                )
            )
            sections = int(payload.get("sections", 6))
            auto_label_conf = self._resolve_append_auto_label_conf(
                sku=sku,
                target_version=version_num,
                payload=payload,
            )
            stage1_model_path = resolve_base_model_for_sku(
                sku=sku,
                base_model="auto",
                target_version=version_num,
            )
            if auto_mode_expected:
                self.store.update_onboarding_session(
                    session_id,
                    state="auto_labeling",
                    stage1_model_path=stage1_model_path,
                    latest_job_id=job["job_id"],
                    session_context=session_context,
                )
                self.store.update_ui_job(
                    job["job_id"],
                    step="auto_labeling",
                    progress=70.0,
                    message="Auto-labeling append frames with detector + Gemini",
                )
                summary, failure_reason = self._auto_label_with_retries(
                    sku=sku,
                    model_path=stage1_model_path,
                    object_description=str(payload.get("object_description") or sku),
                    target_count=target_auto,
                    conf=auto_label_conf,
                    sections=sections,
                    append_video_path=str(video_path),
                    enabled=bool(payload.get("vlm_enabled", self._vlm_setting("enabled", True))),
                    gemini_model=str(payload.get("gemini_model", self._vlm_setting("model", "gemini-3-pro-preview"))),
                    gemini_api_key_env=str(payload.get("gemini_api_key_env", self._vlm_setting("api_key_env", "GEMINI_API_KEY"))),
                    batch_size=int(payload.get("gemini_batch_size", self._vlm_setting("batch_size", 24))),
                    max_boxes=int(payload.get("max_proposals_per_frame", self._vlm_setting("max_boxes_approval", 8))),
                    retries=int(payload.get("gemini_retry_attempts", self._vlm_setting("gemini_retry_attempts", 3))),
                    backoff_seconds=float(
                        payload.get(
                            "gemini_retry_backoff_seconds",
                            self._vlm_setting("gemini_retry_backoff_seconds", 1.5),
                        )
                    ),
                    job_id=str(job["job_id"]),
                    step="auto_labeling",
                )
                if summary is None:
                    raise RuntimeError("Auto labeling returned no summary")
                if failure_reason not in {None, "no_candidates"}:
                    fallback_candidates = self._manual_candidates_from_detector(
                        sku=sku,
                        model_path=stage1_model_path,
                        target_count=target_auto,
                        conf=auto_label_conf,
                        sections=sections,
                        append_video_path=str(video_path),
                        max_boxes=int(payload.get("max_proposals_per_frame", self._vlm_setting("max_boxes_approval", 8))),
                    )
                    self._move_manual_required(
                        job=job,
                        session=session,
                        reason=str(failure_reason),
                        stage1_model_path=stage1_model_path,
                        approval_candidates=fallback_candidates,
                        payload_updates={"flow_kind": "append"},
                    )
                    return

                if summary.total == 0:
                    self.store.update_onboarding_session(
                        session_id,
                        state="complete",
                        seed_candidates=[],
                        approval_candidates=[],
                        stage1_model_path=stage1_model_path,
                        latest_job_id=job["job_id"],
                        session_context=session_context,
                    )
                    self.store.update_ui_job(
                        job["job_id"],
                        status="succeeded",
                        step="complete",
                        progress=100.0,
                        message="No new unlabeled append frames were found",
                    )
                    return

                stage2_job = self.store.create_ui_job(
                    sku_id=sku,
                    job_type="train_stage2",
                    status="queued",
                    step="train_stage2",
                    progress=0.0,
                    message="Queued stage2 training",
                    payload={"session_id": session_id},
                )
                self.store.update_onboarding_session(
                    session_id,
                    state="auto_ready_stage2",
                    seed_candidates=[],
                    approval_candidates=[
                        {
                            "frame_name": item.get("frame_name"),
                            "status": "positive" if item.get("final_boxes") else "negative",
                            "boxes": item.get("final_boxes", []),
                            "source": "detector_gemini_auto",
                            "reason": item.get("reason", ""),
                        }
                        for item in summary.decisions
                        if item.get("decision") != "uncertain"
                    ],
                    stage1_model_path=stage1_model_path,
                    latest_job_id=stage2_job["job_id"],
                    session_context=session_context,
                )
                self.store.update_ui_job(
                    job["job_id"],
                    status="succeeded",
                    step="auto_ready_stage2",
                    progress=100.0,
                    message=f"Auto-labeling complete (conf={auto_label_conf:.3f}). Stage2 training queued.",
                    payload={"queued_stage2_job_id": stage2_job["job_id"], "auto_label_conf": auto_label_conf},
                )
                self.enqueue(stage2_job["job_id"])
                return

            self.store.update_ui_job(
                job["job_id"],
                step="approval_labeling",
                progress=80.0,
                message="Preparing Gemini proposals for append training",
            )
            approval_candidates = prepare_approval_candidates_with_gemini(
                sku=sku,
                object_description=str(payload.get("object_description") or sku),
                target_count=target_auto,
                enabled=bool(payload.get("vlm_enabled", self._vlm_setting("enabled", True))),
                sections=sections,
                overwrite=bool(payload.get("overwrite", False)),
                gemini_model=str(payload.get("gemini_model", self._vlm_setting("model", "gemini-3-pro-preview"))),
                gemini_api_key_env=str(payload.get("gemini_api_key_env", self._vlm_setting("api_key_env", "GEMINI_API_KEY"))),
                batch_size=int(payload.get("gemini_batch_size", self._vlm_setting("batch_size", 24))),
                max_proposals_per_frame=int(
                    payload.get(
                        "max_proposals_per_frame",
                        self._vlm_setting("max_boxes_approval", 8),
                    )
                ),
                append_video_path=str(video_path),
            )
            required = min(target_auto, len(approval_candidates))
            next_payload = dict(payload)
            next_payload["required_approval_count"] = required
            next_payload["approval_only"] = True
            self.store.update_onboarding_session(
                session_id,
                state="approval_labeling",
                seed_candidates=[],
                approval_candidates=approval_candidates,
                stage1_model_path=stage1_model_path,
                latest_job_id=job["job_id"],
                session_context=session_context,
            )
            self.store.update_ui_job(
                job["job_id"],
                status="waiting_user",
                step="approval_labeling",
                progress=100.0,
                message=f"Review {required} Gemini proposals, then start retraining",
                payload=next_payload,
            )
            return

        self.store.update_ui_job(job["job_id"], step="seed_labeling", progress=80.0, message="Preparing Gemini seed labels")
        seed_candidates = prepare_seed_candidates_with_gemini(
            sku=sku,
            object_description=str(payload.get("object_description") or sku),
            enabled=bool(payload.get("vlm_enabled", self._vlm_setting("enabled", True))),
            target_count=int(payload.get("seed_target_count", self._vlm_setting("seed_target_count", 24))),
            sections=int(payload.get("sections", 6)),
            overwrite=bool(payload.get("overwrite", False)),
            gemini_model=str(payload.get("gemini_model", self._vlm_setting("model", "gemini-3-pro-preview"))),
            gemini_api_key_env=str(payload.get("gemini_api_key_env", self._vlm_setting("api_key_env", "GEMINI_API_KEY"))),
            batch_size=int(payload.get("gemini_batch_size", self._vlm_setting("batch_size", 24))),
            max_proposals_per_frame=int(
                payload.get(
                    "max_boxes_seed",
                    self._vlm_setting("max_boxes_seed", 4),
                )
            ),
        )
        self.store.update_onboarding_session(
            session_id,
            state="seed_labeling",
            seed_candidates=seed_candidates,
            approval_candidates=[],
            stage1_model_path=None,
            latest_job_id=job["job_id"],
            session_context=session_context,
        )
        self.store.update_ui_job(
            job["job_id"],
            status="waiting_user",
            step="seed_labeling",
            progress=100.0,
            message="AI seed prelabeling ready. Review 3 seed frames to auto-start Stage1.",
        )

    def _job_train_stage1(self, job: dict[str, Any]) -> None:
        payload = job.get("payload", {})
        session_id = str(payload["session_id"])
        session = self.store.get_onboarding_session(session_id)
        if session is None:
            raise RuntimeError(f"Session not found: {session_id}")
        sku = str(session["sku_id"])
        version_num = _version_num_from_tag(str(session["version_tag"]))
        session_context = self._session_context(session)
        auto_mode_expected = bool(session_context.get("auto_mode_expected", True))

        self.store.update_ui_job(job["job_id"], step="train_stage1", progress=1.0, message="Training stage1 model")
        stage1_model_path = train_stage1_for_session(
            sku=sku,
            version_num=version_num,
            mode=str(session["mode"]),
            progress_cb=lambda update: self.store.update_ui_job(
                job["job_id"],
                step="train_stage1",
                progress=self._progress_from_payload(update),
                message=str(update.get("message", "Training stage1")),
                payload=update,
            ),
        )
        if auto_mode_expected:
            self.store.update_onboarding_session(
                session_id,
                state="auto_labeling",
                stage1_model_path=stage1_model_path,
                latest_job_id=job["job_id"],
                session_context=session_context,
            )
            self.store.update_ui_job(job["job_id"], step="auto_labeling", progress=95.0, message="Auto-labeling stage2 set")
            target_auto = int(
                payload.get(
                    "required_approval_count",
                    self._vlm_setting("initial_auto_approval_count", 24),
                )
            )
            summary, failure_reason = self._auto_label_with_retries(
                sku=sku,
                model_path=stage1_model_path,
                object_description=str(payload.get("object_description") or sku),
                target_count=target_auto,
                conf=float(
                    payload.get(
                        "detector_conf_for_auto_label",
                        self._vlm_setting("detector_conf_for_auto_label", 0.02),
                    )
                ),
                sections=int(payload.get("sections", 6)),
                append_video_path=None,
                enabled=bool(payload.get("vlm_enabled", self._vlm_setting("enabled", True))),
                gemini_model=str(payload.get("gemini_model", self._vlm_setting("model", "gemini-3-pro-preview"))),
                gemini_api_key_env=str(payload.get("gemini_api_key_env", self._vlm_setting("api_key_env", "GEMINI_API_KEY"))),
                batch_size=int(payload.get("gemini_batch_size", self._vlm_setting("batch_size", 24))),
                max_boxes=int(payload.get("max_proposals_per_frame", self._vlm_setting("max_boxes_approval", 8))),
                retries=int(payload.get("gemini_retry_attempts", self._vlm_setting("gemini_retry_attempts", 3))),
                backoff_seconds=float(
                    payload.get(
                        "gemini_retry_backoff_seconds",
                        self._vlm_setting("gemini_retry_backoff_seconds", 1.5),
                    )
                ),
                job_id=str(job["job_id"]),
                step="auto_labeling",
            )
            if summary is None:
                raise RuntimeError("Auto labeling returned no summary")
            if failure_reason not in {None, "no_candidates"}:
                fallback_candidates = self._manual_candidates_from_detector(
                    sku=sku,
                    model_path=stage1_model_path,
                    target_count=target_auto,
                    conf=float(payload.get("detector_conf_for_auto_label", self._vlm_setting("detector_conf_for_auto_label", 0.02))),
                    sections=int(payload.get("sections", 6)),
                    append_video_path=None,
                    max_boxes=int(payload.get("max_proposals_per_frame", self._vlm_setting("max_boxes_approval", 8))),
                )
                self._move_manual_required(
                    job=job,
                    session=session,
                    reason=str(failure_reason),
                    stage1_model_path=stage1_model_path,
                    approval_candidates=fallback_candidates,
                    payload_updates={"flow_kind": "initial"},
                )
                return

            stage2_job = self.store.create_ui_job(
                sku_id=sku,
                job_type="train_stage2",
                status="queued",
                step="train_stage2",
                progress=0.0,
                message="Queued stage2 training",
                payload={"session_id": session_id},
            )
            self.store.update_onboarding_session(
                session_id,
                state="auto_ready_stage2",
                approval_candidates=[
                    {
                        "frame_name": item.get("frame_name"),
                        "status": "positive" if item.get("final_boxes") else "negative",
                        "boxes": item.get("final_boxes", []),
                        "source": "detector_gemini_auto",
                        "reason": item.get("reason", ""),
                    }
                    for item in summary.decisions
                    if item.get("decision") != "uncertain"
                ],
                stage1_model_path=stage1_model_path,
                latest_job_id=stage2_job["job_id"],
                session_context=session_context,
            )
            self.store.update_ui_job(
                job["job_id"],
                status="succeeded",
                step="auto_ready_stage2",
                progress=100.0,
                message="Stage1 complete. Stage2 training queued.",
                payload={"queued_stage2_job_id": stage2_job["job_id"]},
            )
            self.enqueue(stage2_job["job_id"])
            return

        self.store.update_ui_job(job["job_id"], step="approval_labeling", progress=95.0, message="Generating Gemini proposals")
        approval_candidates = prepare_approval_candidates_with_gemini(
            sku=sku,
            object_description=str(payload.get("object_description") or sku),
            target_count=int(
                payload.get(
                    "required_approval_count",
                    self._vlm_setting("initial_auto_approval_count", 24),
                )
            ),
            enabled=bool(payload.get("vlm_enabled", self._vlm_setting("enabled", True))),
            sections=int(payload.get("sections", 6)),
            overwrite=bool(payload.get("overwrite", False)),
            gemini_model=str(payload.get("gemini_model", self._vlm_setting("model", "gemini-3-pro-preview"))),
            gemini_api_key_env=str(payload.get("gemini_api_key_env", self._vlm_setting("api_key_env", "GEMINI_API_KEY"))),
            batch_size=int(payload.get("gemini_batch_size", self._vlm_setting("batch_size", 24))),
            max_proposals_per_frame=int(
                payload.get(
                    "max_proposals_per_frame",
                    self._vlm_setting("max_boxes_approval", 8),
                )
            ),
        )
        self.store.update_onboarding_session(
            session_id,
            state="approval_labeling",
            approval_candidates=approval_candidates,
            stage1_model_path=stage1_model_path,
            latest_job_id=job["job_id"],
            session_context=session_context,
        )
        self.store.update_ui_job(
            job["job_id"],
            status="waiting_user",
            step="approval_labeling",
            progress=100.0,
            message="Gemini approval labeling ready",
        )

    def _job_train_stage2(self, job: dict[str, Any]) -> None:
        payload = job.get("payload", {})
        session_id = str(payload["session_id"])
        session = self.store.get_onboarding_session(session_id)
        if session is None:
            raise RuntimeError(f"Session not found: {session_id}")
        sku = str(session["sku_id"])
        version_num = _version_num_from_tag(str(session["version_tag"]))
        stage1_model_path = session.get("stage1_model_path")
        if not stage1_model_path:
            raise RuntimeError("Stage1 model path missing from session")

        session_context = self._session_context(session)
        self.store.update_onboarding_session(
            session_id,
            state="train_stage2",
            latest_job_id=job["job_id"],
            session_context=session_context,
        )
        self.store.update_ui_job(job["job_id"], step="train_stage2", progress=1.0, message="Training stage2 model")
        best_model = train_stage2_for_session(
            sku=sku,
            version_num=version_num,
            stage1_model_path=str(stage1_model_path),
            mode=str(session["mode"]),
            progress_cb=lambda update: self.store.update_ui_job(
                job["job_id"],
                step="train_stage2",
                progress=self._progress_from_payload(update),
                message=str(update.get("message", "Training stage2")),
                payload=update,
            ),
        )
        self.store.update_sku_metadata(
            sku_id=sku,
            updates={"latest_model": best_model, "source": "ui_onboarding"},
        )
        flow_kind = str(session_context.get("flow_kind", "initial"))
        if flow_kind == "initial" and bool(self._vlm_setting("auto_after_trust", True)):
            metadata = self.store.get_sku_metadata(sku)
            configured_required = int(self._vlm_setting("trust_required_reviews", 3))
            review_count = max(int(metadata.get("trust_review_count", 0)), configured_required)
            self.store.mark_sku_trusted(sku, review_count=review_count)

        self.store.update_onboarding_session(
            session_id,
            state="complete",
            latest_job_id=job["job_id"],
            session_context=session_context,
        )
        self.store.update_ui_job(
            job["job_id"],
            status="succeeded",
            step="complete",
            progress=100.0,
            message=f"Onboarding complete: {best_model}",
            payload={"best_model": best_model},
        )

    def _job_full_train(self, job: dict[str, Any]) -> None:
        sku = str(job["sku_id"])
        self.store.update_ui_job(job["job_id"], step="full_train", progress=1.0, message="Running full train")
        best_model = run_full_train_for_sku(
            sku=sku,
            progress_cb=lambda update: self.store.update_ui_job(
                job["job_id"],
                step="full_train",
                progress=self._progress_from_payload(update),
                message=str(update.get("message", "Full training")),
                payload=update,
            ),
        )
        self.store.update_ui_job(
            job["job_id"],
            status="succeeded",
            step="complete",
            progress=100.0,
            message=f"Full training complete: {best_model}",
            payload={"best_model": best_model},
        )
