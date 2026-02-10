# pipeline

Pipelines wire together ingest + vision + profiles into runnable workflows.

## Responsibilities
- `count_stream.py`: live webcam/RTSP processing + overlay + counts
- `count_video.py` : offline file processing for repeatable experiments
- `enroll.py`      : build SKU profiles from enrollment media
- `zone_count.py`  : counting logic (line/zone, dedupe, cooldowns)

## Contract
Pipelines orchestrate; model internals belong in `vision/`.
