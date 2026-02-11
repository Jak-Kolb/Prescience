# cloud

Minimal FastAPI + SQLite backend and SSE dashboard.

## Features

- `/events` ingest for COUNT/HEARTBEAT/ALERT
- `/stream` Server-Sent Events for live dashboard updates
- run summaries and line live views
- SKU CRUD
- device config and pairing

SSE broadcaster is in-memory and intended for single-worker MVP operation.
