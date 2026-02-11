# ingest

Enrollment ingest utilities.

## Responsibilities

- Extract high-quality frames from enrollment videos.
- Force temporal diversity by splitting candidate frames into 1/6 timeline bins.
- Rank by sharpness, dedupe within each bin, and save metadata (`meta.json`).

## Command

`prescience enroll extract-frames --video ... --sku ... --target ...`
