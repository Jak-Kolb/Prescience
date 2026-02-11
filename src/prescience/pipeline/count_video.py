"""Video-file counting wrapper."""

from __future__ import annotations

from pathlib import Path

from prescience.pipeline.count_stream import run_count_agent


def run_video(
    source: str,
    model_path: str,
    zone_config_path: Path,
    line_id: str,
    device_id: str,
    run_id: str | None,
    event_endpoint: str | None,
    jsonl_log_path: Path,
    profiles_root: Path,
    config_path: Path,
) -> None:
    """Run the standard count agent against a file source."""
    run_count_agent(
        source=source,
        model_path=model_path,
        zone_config_path=zone_config_path,
        line_id=line_id,
        device_id=device_id,
        run_id=run_id,
        event_endpoint=event_endpoint,
        jsonl_log_path=jsonl_log_path,
        profiles_root=profiles_root,
        config_path=config_path,
    )
