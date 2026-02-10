# vision

Per-frame perception: detect, track, embed, and match.

## Responsibilities
- `detector.py`: frame -> detections
- `tracker.py` : frame -> tracked detections (stable IDs)
- `embeddings.py`: crop -> embedding vector
- `matcher.py`: embedding -> SKU match / unknown

## Contract
Return project-level types (from `prescience/types.py`), not model-specific tensors.
