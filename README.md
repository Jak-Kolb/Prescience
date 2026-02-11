# Prescience MVP

Camera-based end-of-line counting without PLC changes.

## What This MVP Includes

- SKU enrollment from short phone video (`show all sides`).
- Frame extraction with balanced 1/6 timeline coverage.
- Guided onboarding labeling (manual seeds + model-assisted approvals + negatives).
- SKU-specific YOLO detector training with stable `best.pt` output path.
- Optional embedding profile creation (`resnet18` default, pluggable interface).
- Edge runtime agent: detect + track + zone crossing count + event emission.
- Minimal cloud backend (FastAPI + SQLite) and live SSE dashboard.
- Pairing support with dev-mode bypass.
- Interactive zone calibration helper.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## CLI

```bash
prescience --help
prescience enroll --help
prescience train --help
prescience cloud --help
```

Main commands:

- `prescience enroll extract-frames`
- `prescience enroll label`
- `prescience train detector`
- `prescience enroll build-profile`
- `prescience run`
- `prescience calibrate zone`
- `prescience cloud serve`
- `prescience demo run-local`

Append additional enroll video frames to an existing SKU dataset:

```bash
prescience enroll extract-frames \
  --video data/raw/videos/<sku>/<sku>_1.MOV \
  --sku <sku> \
  --append
```

## End-to-End Demo

1. Extract frames from enrollment video:

```bash
prescience enroll extract-frames \
  --video data/raw/videos/can1_test/enroll.MOV \
  --sku can1_test \
  --target 150
```

2. Run guided labeling workflow:

```bash
prescience enroll label --sku can1_test
```

3. Train detector:

```bash
prescience train detector --sku can1_test --version v1 --epochs 60
```

4. Build embedding profile:

```bash
prescience enroll build-profile --sku can1_test --max-embeddings 40
```

5. Calibrate zone config:

```bash
prescience calibrate zone --source 0 --out configs/line-1.yaml --line-id line-1
```

6. Start cloud backend:

```bash
prescience cloud serve --host 127.0.0.1 --port 8000
```

7. Run edge agent:

```bash
prescience run \
  --source 0 \
  --model data/models/yolo/can1_test_v1/best.pt \
  --zone-config configs/line-1.yaml \
  --line-id line-1 \
  --device-id device-1 \
  --event-endpoint http://127.0.0.1:8000/events
```

8. Open dashboard: `http://127.0.0.1:8000`

## Dashboard Video Upload

From the dashboard, use **SKU Enrollment Videos** to upload local videos:

- Enter a new SKU name + choose/drag a video: file is saved as `data/raw/videos/<sku>/<sku>_0.MOV`.
- Upload another video for same SKU: auto-saved as `data/raw/videos/<sku>/<sku>_1.MOV`, then `_2`, etc.
- Existing SKUs are shown with quick “Add Video” upload forms.
- Existing SKUs also have a **Delete SKU** action that removes SKU metadata and local artifacts:
  videos, derived frames/labels/crops, profile files, and SKU-specific datasets/models.

## Event Contract

Event types:

- `COUNT`: includes `seq`, `timestamp`, `frame_ts`, `count_delta`, `counts_total_overall`, `counts_total_by_sku`.
- `HEARTBEAT`: includes fps/uptime/quality metrics.
- `ALERT`: includes warning/critical condition signals.

Cloud uses sequence checks per `(device_id, run_id)` and inserts `seq_gap` alerts for missing sequence windows.

## Cloud Notes

- SSE endpoint: `GET /stream?line_id=...`
- Ingest endpoint: `POST /events`
- Pairing can be required or disabled via `configs/default.yaml`:
  - `cloud.pairing.required: false` (dev mode)

## Directory Layout

- `src/prescience/ingest`: video-to-frames pipeline.
- `src/prescience/datasets`: label manifests + YOLO dataset/train helpers.
- `src/prescience/vision`: detector/tracker/embedder/matcher.
- `src/prescience/pipeline`: enroll/train/run/calibration orchestration.
- `src/prescience/events`: event schemas + emitter.
- `src/prescience/cloud`: FastAPI app, SQLite store, SSE stream, dashboard templates.
- `data/`: local artifacts (ignored in git except `.gitkeep` + `data/README.md`).

## One-Command Local Demo Script

```bash
bash scripts/demo_mvp.sh
```

This script starts cloud backend and then runs the edge agent against configurable source/model defaults.
