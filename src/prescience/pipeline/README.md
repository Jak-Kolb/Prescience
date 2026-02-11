# pipeline

Orchestration workflows that combine ingest, vision, profiles, and cloud events.

## Modules

- `enroll.py`: frame extraction, onboarding labeling, detector training, profile build.
- `web_onboarding.py`: browser-driven onboarding candidate prep, label saves, stage1/stage2 train orchestration.
- `count_stream.py`: edge runtime loop with tracking, zone counting, and event emit.
- `tracking_session.py`: in-process tracking sessions for browser MJPEG preview and controls.
- `count_video.py`: convenience wrapper for file-based counting runs.
- `zone_count.py`: direction-aware polygon exit counter with anti-double-count logic.
- `calibrate_zone.py`: interactive polygon + direction calibration helper.
- `demo.py`: local backend + edge demo runner.
