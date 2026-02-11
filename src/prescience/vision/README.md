# vision

Per-frame perception and product matching components.

## Modules

- `detector.py`: YOLO detection wrapper.
- `tracker.py`: YOLO + ByteTrack wrapper with stable IDs.
- `embeddings.py`: pluggable embedder interface (`Embedder.encode`) and `resnet18` backend.
- `matcher.py`: cosine matching against SKU profile embeddings with unknown thresholding.

## Contract

Vision modules return project-level types and numpy arrays, not framework-specific tensors.
