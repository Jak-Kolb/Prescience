# ingest

Ingest converts user inputs (videos/images) into structured artifacts.

## Responsibilities
- Extract frames from an enrollment video (skip blurry/duplicates)
- Save outputs in a consistent folder structure + metadata

## Contract
No model training logic here. Ingest should be deterministic given the same inputs + config.
