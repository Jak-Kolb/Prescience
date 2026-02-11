"""Local demo runner utilities."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


def run_local_demo(
    source: str,
    model_path: str,
    line_id: str,
    device_id: str,
    config_path: Path,
) -> None:
    """Start cloud backend, then run edge agent in same terminal session."""
    cloud_cmd = [
        sys.executable,
        "-m",
        "prescience.cli",
        "cloud",
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--config",
        str(config_path),
    ]

    edge_cmd = [
        sys.executable,
        "-m",
        "prescience.cli",
        "run",
        "--source",
        str(source),
        "--model",
        str(model_path),
        "--line-id",
        line_id,
        "--device-id",
        device_id,
        "--config",
        str(config_path),
        "--event-endpoint",
        "http://127.0.0.1:8000/events",
    ]

    print("Starting cloud backend:", " ".join(cloud_cmd))
    cloud_proc = subprocess.Popen(cloud_cmd)

    try:
        time.sleep(1.5)
        print("Starting edge agent:", " ".join(edge_cmd))
        subprocess.run(edge_cmd, check=False)
    finally:
        cloud_proc.terminate()
        try:
            cloud_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cloud_proc.kill()
