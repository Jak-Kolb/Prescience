# training

Training notes for Prescience detector workflows.

## Current Workflow

Training is orchestrated through pipeline and dataset modules, not standalone scripts:

- `prescience enroll label` performs onboarding stage1/stage2 training.
- `prescience train detector` builds a versioned dataset and trains a stable SKU model.

Stable output path convention:

- `data/models/yolo/{sku}_{version}/best.pt`

Ultralytics internal run paths can vary; the pipeline always copies `best.pt` to this stable location.
