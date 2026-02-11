# profiles

SKU identity profile schema and IO.

## Artifacts

Each SKU profile directory contains:

- `profile.json`: metadata (sku id, threshold, backbone, embedding_dim, timestamps)
- `embeddings.npy`: representative embedding matrix

## Responsibilities

- `schema.py`: typed, versioned metadata schema.
- `io.py`: save/load helpers for profile metadata and embeddings.
