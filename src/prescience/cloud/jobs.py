"""Background UI job runner for onboarding and retraining workflows."""

from __future__ import annotations

import math
import queue
import threading
import traceback
from pathlib import Path
from typing import Any

from prescience.cloud.store import CloudStore
from prescience.pipeline.enroll import extract_frames_for_sku
from prescience.pipeline.enroll import resolve_base_model_for_sku
from prescience.pipeline.web_onboarding import (
    prepare_approval_candidates,
    prepare_seed_candidates,
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

    def __init__(self, store: CloudStore) -> None:
        self.store = store
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
            target_approvals = int(payload.get("required_approval_count", 30))
            sections = int(payload.get("sections", 6))
            approve_per_section = max(1, math.ceil(target_approvals / max(1, sections)))
            stage1_model_path = resolve_base_model_for_sku(
                sku=sku,
                base_model="auto",
                target_version=version_num,
            )
            self.store.update_ui_job(
                job["job_id"],
                step="approval_labeling",
                progress=80.0,
                message="Preparing model proposals for append training",
            )
            approval_candidates = prepare_approval_candidates(
                sku=sku,
                stage1_model_path=stage1_model_path,
                approve_per_section=approve_per_section,
                sections=sections,
                conf=float(payload.get("conf_propose", 0.03)),
                imgsz=int(payload.get("imgsz", 640)),
                overwrite=bool(payload.get("overwrite", False)),
                max_proposals_per_frame=int(payload.get("max_proposals_per_frame", 8)),
                append_video_path=str(video_path),
            )
            if target_approvals > 0 and len(approval_candidates) > target_approvals:
                approval_candidates = approval_candidates[:target_approvals]
            required = min(target_approvals, len(approval_candidates))

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
            )
            self.store.update_ui_job(
                job["job_id"],
                status="waiting_user",
                step="approval_labeling",
                progress=100.0,
                message=f"Review {required} model guesses, then start retraining",
                payload=next_payload,
            )
            return

        self.store.update_ui_job(job["job_id"], step="seed_labeling", progress=80.0, message="Preparing seed candidates")
        seed_candidates = prepare_seed_candidates(
            sku,
            manual_per_section=int(payload.get("seed_per_bin", 4)),
            sections=6,
            overwrite=bool(payload.get("overwrite", False)),
        )
        self.store.update_onboarding_session(
            session_id,
            state="seed_labeling",
            seed_candidates=seed_candidates,
            approval_candidates=[],
            stage1_model_path=None,
            latest_job_id=job["job_id"],
        )
        self.store.update_ui_job(
            job["job_id"],
            status="waiting_user",
            step="seed_labeling",
            progress=100.0,
            message="Seed labeling ready",
        )

    def _job_train_stage1(self, job: dict[str, Any]) -> None:
        payload = job.get("payload", {})
        session_id = str(payload["session_id"])
        session = self.store.get_onboarding_session(session_id)
        if session is None:
            raise RuntimeError(f"Session not found: {session_id}")
        sku = str(session["sku_id"])
        version_num = _version_num_from_tag(str(session["version_tag"]))

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
        self.store.update_ui_job(job["job_id"], step="approval_labeling", progress=95.0, message="Generating proposals")
        approval_candidates = prepare_approval_candidates(
            sku=sku,
            stage1_model_path=stage1_model_path,
            approve_per_section=int(payload.get("approve_per_bin", 5)),
            conf=float(payload.get("conf_propose", 0.03)),
            imgsz=int(payload.get("imgsz", 640)),
            overwrite=bool(payload.get("overwrite", False)),
            max_proposals_per_frame=int(payload.get("max_proposals_per_frame", 8)),
        )
        self.store.update_onboarding_session(
            session_id,
            state="approval_labeling",
            approval_candidates=approval_candidates,
            stage1_model_path=stage1_model_path,
            latest_job_id=job["job_id"],
        )
        self.store.update_ui_job(
            job["job_id"],
            status="waiting_user",
            step="approval_labeling",
            progress=100.0,
            message="Approval labeling ready",
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
        self.store.upsert_sku(
            sku_id=sku,
            name=sku,
            profile_path=f"data/profiles/{sku}",
            threshold=None,
            metadata={"latest_model": best_model, "source": "ui_onboarding"},
        )
        self.store.update_onboarding_session(
            session_id,
            state="complete",
            latest_job_id=job["job_id"],
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
