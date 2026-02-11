from __future__ import annotations

from pathlib import Path

from prescience.datasets.manifest import (
    labeled_image_names,
    load_or_create_manifest,
    save_manifest,
    should_process,
    upsert_record,
)


def test_manifest_round_trip_and_labeled_names(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest = load_or_create_manifest(
        manifest_path=manifest_path,
        sku="sku-a",
        frames_dir=tmp_path / "frames",
        labels_dir=tmp_path / "labels",
    )

    upsert_record(manifest, "000001.jpg", "positive", str(tmp_path / "labels/000001.txt"))
    upsert_record(manifest, "000002.jpg", "negative", str(tmp_path / "labels/000002.txt"))
    upsert_record(manifest, "000003.jpg", "skipped", str(tmp_path / "labels/000003.txt"))
    save_manifest(manifest_path, manifest)

    loaded = load_or_create_manifest(
        manifest_path=manifest_path,
        sku="sku-a",
        frames_dir=tmp_path / "frames",
        labels_dir=tmp_path / "labels",
    )
    names = labeled_image_names(loaded)
    assert names == {"000001.jpg", "000002.jpg"}


def test_should_process_honors_overwrite() -> None:
    labeled = {"000010.jpg"}
    assert should_process("000010.jpg", overwrite=False, labeled_names=labeled) is False
    assert should_process("000010.jpg", overwrite=True, labeled_names=labeled) is True
    assert should_process("000011.jpg", overwrite=False, labeled_names=labeled) is True
