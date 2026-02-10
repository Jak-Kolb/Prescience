# prescience (core package)

This folder is the reusable Python library (`import prescience`).

## Layout

- `ingest/` : turn raw videos into clean frames/assets
- `vision/` : detection, tracking, embeddings, matching
- `pipeline/`: end-to-end workflows (enroll, count video/stream)
- `profiles/`: SKU profile schema + load/save
- `types.py` : shared dataclasses used across modules

## Rule

Reusable logic goes in `src/prescience/`. Command-line entrypoints go in `scripts/`.
