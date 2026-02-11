# pipeline

Orchestration workflows that combine ingest, vision, profiles, and cloud events.

## Modules

- `enroll.py`: frame extraction, bootstrap labeling, detector training, profile build.
- `count_stream.py`: edge runtime loop with tracking, zone counting, and event emit.
- `count_video.py`: convenience wrapper for file-based counting runs.
- `zone_count.py`: direction-aware polygon exit counter with anti-double-count logic.
- `calibrate_zone.py`: interactive polygon + direction calibration helper.
- `demo.py`: local backend + edge demo runner.
