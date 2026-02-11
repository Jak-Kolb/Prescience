# prescience (core package)

Reusable Python package for the Prescience MVP.

## Package Layout

- `ingest/`: enrollment media ingestion and frame extraction.
- `datasets/`: labeling manifests, dataset builders, detector training helpers.
- `vision/`: detect, track, embed, and match interfaces.
- `pipeline/`: end-to-end operational workflows.
- `events/`: canonical COUNT/HEARTBEAT/ALERT schema and emitter.
- `cloud/`: FastAPI backend, SQLite store, SSE stream, dashboard templates.
- `profiles/`: SKU profile schema + profile persistence.
- `config.py`: typed app configuration loader.
- `cli.py`: Typer command-line interface.

## Guideline

Put reusable logic in `src/prescience/`; keep wrapper scripts in `scripts/`.
