# cloud

Minimal FastAPI + SQLite backend and SSE dashboard.

## Features

- `/events` ingest for COUNT/HEARTBEAT/ALERT
- `/stream` Server-Sent Events for live dashboard updates
- `/stream/jobs` Server-Sent Events for onboarding/training job updates
- run summaries and line live views
- SKU CRUD
- device config and pairing
- web-first SKU onboarding sessions with persisted state (`onboarding_sessions`)
- background UI jobs (`ui_jobs`) for extract/train orchestration
- zone config APIs and in-browser tracking session APIs
- Gemini-first onboarding flow:
  - initial seed prelabeling (24 frames)
  - 3-review trust gate with auto-start Stage1
  - automatic detector+Gemini stage2 labeling after trust
  - append retraining runs in background for trusted SKUs
  - manual fallback route when Gemini is unavailable after retries (`/api/onboarding/{session_id}/manual/enter`)

SSE broadcaster is in-memory and intended for single-worker MVP operation.
