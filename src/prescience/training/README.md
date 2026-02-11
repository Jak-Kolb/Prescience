# training

Training notes for Prescience detector workflows.

## Current Workflow

Training is orchestrated through pipeline and dataset modules, not standalone scripts:

- `prescience enroll label` performs onboarding stage1/stage2 training.
- `prescience train detector` builds a versioned dataset and trains a stable SKU model.
- Default mode is `quick` with `base_model=auto` for fast incremental fine-tuning.

## Modes

- `quick`: `core_new` dataset scope, lower epochs/imgsz, early stopping, freeze enabled.
- `milestone`: all labeled images, medium epochs/imgsz, no freeze.
- `full`: all labeled images, highest epochs/imgsz, no freeze.

Train-state for incremental loops is stored per SKU at:

- `data/derived/labels/{sku}/train_state.json`

Fields include:

- `core_images`
- `trained_snapshot_images`
- `last_trained_model`
- `last_mode`
- `updated_at`

Stable output path convention:

- `data/models/yolo/{sku}_{version}/best.pt`

Ultralytics internal run paths can vary; the pipeline always copies `best.pt` to this stable location.
