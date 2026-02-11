# training

Training notes for Prescience detector workflows.

## Current Workflow

Training is orchestrated through pipeline and dataset modules, not standalone scripts:

- `prescience enroll label` performs onboarding stage1/stage2 training.
- `prescience train detector` builds a versioned dataset and trains a stable SKU model.
- Default mode is `quick` with `base_model=auto` for fast incremental fine-tuning.
- Onboarding defaults to `manual_per_section=4` (24 seed frames across 6 bins).
- First-time SKU onboarding in quick mode uses boosted epoch minimums for stage1/stage2.

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
- `last_version_tag`
- `dataset_hash`
- `last_imgsz`
- `last_base_model_path`
- `last_ultralytics_version`
- `updated_at`

Resume behavior:

- `--resume` attempts to continue from latest `last.pt` for the same SKU/version.
- Resume is only allowed when `train_state.json` signature matches current request
  (`dataset_hash`, mode, imgsz, base model path, ultralytics version, version tag).

Stable output path convention:

- `data/models/yolo/{sku}_{version}/best.pt`

Ultralytics internal run paths can vary; the pipeline always copies `best.pt` to this stable location.
