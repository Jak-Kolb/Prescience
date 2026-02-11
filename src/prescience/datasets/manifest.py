"""Manifest utilities for idempotent labeling runs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


LabelStatus = Literal["positive", "negative", "skipped"]


class LabelRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_name: str
    status: LabelStatus
    label_file: str
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class LabelManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sku: str
    frames_dir: str
    labels_dir: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    stage1_model: str | None = None
    stage2_model: str | None = None
    records: dict[str, LabelRecord] = Field(default_factory=dict)


def load_or_create_manifest(
    manifest_path: str | Path,
    sku: str,
    frames_dir: str | Path,
    labels_dir: str | Path,
) -> LabelManifest:
    """Load existing manifest or initialize a new one."""
    path = Path(manifest_path)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return LabelManifest.model_validate(raw)

    manifest = LabelManifest(
        sku=sku,
        frames_dir=str(Path(frames_dir)),
        labels_dir=str(Path(labels_dir)),
    )
    save_manifest(path, manifest)
    return manifest


def save_manifest(manifest_path: str | Path, manifest: LabelManifest) -> None:
    """Persist manifest to disk."""
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.updated_at = datetime.now(UTC)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest.model_dump(mode="json"), f, indent=2)


def upsert_record(
    manifest: LabelManifest,
    image_name: str,
    status: LabelStatus,
    label_file: str,
) -> None:
    """Upsert image labeling state in manifest."""
    manifest.records[image_name] = LabelRecord(
        image_name=image_name,
        status=status,
        label_file=label_file,
    )


def labeled_image_names(manifest: LabelManifest) -> set[str]:
    """Return names of images with positive or negative labels."""
    return {
        k
        for k, record in manifest.records.items()
        if record.status in {"positive", "negative"}
    }


def negative_image_names(manifest: LabelManifest) -> set[str]:
    """Return names of images marked as negatives."""
    return {k for k, record in manifest.records.items() if record.status == "negative"}


def should_process(image_name: str, overwrite: bool, labeled_names: set[str]) -> bool:
    """Whether image should be offered for labeling in current run."""
    if overwrite:
        return True
    return image_name not in labeled_names
