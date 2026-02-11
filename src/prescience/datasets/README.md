# datasets

Labeling and YOLO dataset helpers.

## Responsibilities

- Manifest-driven idempotent labeling runs.
- Manual + model-assisted bootstrap labeling workflow.
- Negative label support via empty YOLO txt files.
- Build YOLO train/val layout and `data.yaml`.
- Train YOLO and copy best weights to stable output path.

## Key Files

- `manifest.py`: labeling run records, overwrite behavior, state persistence.
- `bootstrap_label.py`: guided stage1/stage2 labeling workflow.
- `yolo.py`: dataset build and train wrappers.
