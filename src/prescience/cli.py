"""Prescience command line interface."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(help="Prescience MVP CLI", no_args_is_help=True)
enroll_app = typer.Typer(help="Enrollment workflows", no_args_is_help=True)
train_app = typer.Typer(help="Training workflows", no_args_is_help=True)
calibrate_app = typer.Typer(help="Calibration helpers", no_args_is_help=True)
cloud_app = typer.Typer(help="Cloud backend", no_args_is_help=True)
demo_app = typer.Typer(help="Demo helpers", no_args_is_help=True)

app.add_typer(enroll_app, name="enroll")
app.add_typer(train_app, name="train")
app.add_typer(calibrate_app, name="calibrate")
app.add_typer(cloud_app, name="cloud")
app.add_typer(demo_app, name="demo")


@enroll_app.command("extract-frames")
def enroll_extract_frames(
    video: Path = typer.Option(..., exists=True, help="Path to enrollment video"),
    sku: str = typer.Option(..., help="SKU identifier"),
    target: int = typer.Option(150, help="Target number of saved frames"),
    out_root: Path = typer.Option(Path("data/derived/frames"), help="Output root"),
    blur_min: float = typer.Option(4.0, help="Minimum blur score"),
    dedupe_sim: float = typer.Option(0.98, help="Max dedupe similarity"),
    append: bool = typer.Option(False, help="Append frames to existing SKU dataset"),
) -> None:
    """Extract evenly distributed enrollment frames."""
    from prescience.pipeline.enroll import extract_frames_for_sku

    extract_frames_for_sku(
        video_path=video,
        sku=sku,
        target_frames=target,
        out_root=out_root,
        blur_min=blur_min,
        dedupe_max_similarity=dedupe_sim,
        append=append,
    )


@enroll_app.command("label")
def enroll_label(
    sku: str = typer.Option(..., help="SKU identifier"),
    seed_per_bin: int = typer.Option(2, help="Manual seed labels per section"),
    approve_per_bin: int = typer.Option(5, help="Approval labels per section"),
    overwrite: bool = typer.Option(False, help="Relabel already-labeled frames"),
    allow_negatives: bool = typer.Option(True, help="Allow negative labels (empty txt)"),
    base_model: str = typer.Option("yolov8n.pt", help="Onboarding base model"),
    imgsz: int = typer.Option(960, help="Training image size"),
    epochs_stage1: int = typer.Option(30, help="Stage1 epochs"),
    epochs_stage2: int = typer.Option(60, help="Stage2 epochs"),
    version: int | None = typer.Option(None, help="Optional final model version number"),
) -> None:
    """Run guided onboarding labeling with model-in-the-loop approvals."""
    from prescience.pipeline.enroll import run_onboarding_labeling_for_sku

    run_onboarding_labeling_for_sku(
        sku=sku,
        manual_per_section=seed_per_bin,
        approve_per_section=approve_per_bin,
        overwrite=overwrite,
        allow_negatives=allow_negatives,
        base_model=base_model,
        imgsz=imgsz,
        epochs_stage1=epochs_stage1,
        epochs_stage2=epochs_stage2,
        version=version,
    )


@enroll_app.command("build-profile")
def enroll_build_profile(
    sku: str = typer.Option(..., help="SKU identifier"),
    max_embeddings: int = typer.Option(40, help="Max embeddings to store"),
    threshold: float = typer.Option(0.72, help="Unknown threshold"),
    padding: float = typer.Option(0.10, help="Box padding ratio for crops"),
    backbone: str = typer.Option("resnet18", help="Embedding backbone"),
) -> None:
    """Build SKU embedding profile from labeled/cropped enrollment data."""
    from prescience.pipeline.enroll import build_sku_profile

    build_sku_profile(
        sku=sku,
        max_embeddings=max_embeddings,
        threshold=threshold,
        padding=padding,
        backbone=backbone,
    )


@train_app.command("detector")
def train_detector(
    sku: str = typer.Option(..., help="SKU identifier"),
    version: str = typer.Option("v1", help="Model version tag"),
    epochs: int = typer.Option(60, help="Training epochs"),
    imgsz: int = typer.Option(960, help="Training image size"),
    conf: float = typer.Option(0.35, help="Inference confidence default"),
    base_model: str = typer.Option("yolov8n.pt", help="Base model or prior checkpoint"),
) -> None:
    """Train a SKU-specific detector from verified labels."""
    from prescience.pipeline.enroll import train_detector_for_sku

    train_detector_for_sku(
        sku=sku,
        version=version,
        epochs=epochs,
        imgsz=imgsz,
        conf=conf,
        base_model=base_model,
    )


@app.command("run")
def run_counter(
    source: str = typer.Option("0", help="Camera index, video path, or RTSP URL"),
    model: str = typer.Option(..., help="Path to detector model"),
    zone_config: Path = typer.Option(Path("configs/default.yaml"), help="Zone config yaml"),
    line_id: str = typer.Option("line-1", help="Line identifier"),
    device_id: str = typer.Option("device-1", help="Device identifier"),
    run_id: str | None = typer.Option(None, help="Optional run identifier"),
    event_endpoint: str | None = typer.Option(None, help="Override event endpoint"),
    jsonl_log: Path = typer.Option(Path("data/runs/events.jsonl"), help="Local fallback JSONL log"),
    profiles_root: Path = typer.Option(Path("data/profiles"), help="Profiles root for SKU matching"),
    config: Path = typer.Option(Path("configs/default.yaml"), help="App config yaml"),
) -> None:
    """Run the edge counting agent."""
    from prescience.pipeline.count_stream import run_count_agent

    run_count_agent(
        source=source,
        model_path=model,
        zone_config_path=zone_config,
        line_id=line_id,
        device_id=device_id,
        run_id=run_id,
        event_endpoint=event_endpoint,
        jsonl_log_path=jsonl_log,
        profiles_root=profiles_root,
        config_path=config,
    )


@calibrate_app.command("zone")
def calibrate_zone(
    source: str = typer.Option("0", help="Camera index or video path"),
    out: Path = typer.Option(Path("configs/line-1.yaml"), help="Output YAML"),
    line_id: str = typer.Option("line-1", help="Line identifier"),
) -> None:
    """Interactively draw counting zone and save config."""
    from prescience.pipeline.calibrate_zone import calibrate_zone_config

    calibrate_zone_config(source=source, out_path=out, line_id=line_id)


@cloud_app.command("serve")
def cloud_serve(
    host: str = typer.Option("127.0.0.1", help="Bind host"),
    port: int = typer.Option(8000, help="Bind port"),
    db_path: Path = typer.Option(Path("data/cloud/prescience.db"), help="SQLite path"),
    config: Path = typer.Option(Path("configs/default.yaml"), help="App config yaml"),
) -> None:
    """Start cloud API and dashboard server."""
    from prescience.cloud.app import serve

    serve(host=host, port=port, db_path=db_path, config_path=config)


@demo_app.command("run-local")
def demo_run_local(
    source: str = typer.Option("0", help="Camera index or video path for edge agent"),
    model: str = typer.Option("yolov8n.pt", help="Model path"),
    line_id: str = typer.Option("line-1", help="Line ID"),
    device_id: str = typer.Option("device-1", help="Device ID"),
    config: Path = typer.Option(Path("configs/default.yaml"), help="App config yaml"),
) -> None:
    """Print and run a simple local backend + edge demo script."""
    from prescience.pipeline.demo import run_local_demo

    run_local_demo(
        source=source,
        model_path=model,
        line_id=line_id,
        device_id=device_id,
        config_path=config,
    )


if __name__ == "__main__":
    app()
