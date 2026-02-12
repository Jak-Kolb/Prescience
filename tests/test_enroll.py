from __future__ import annotations

from pathlib import Path

from prescience.datasets.bootstrap_label import _select_processable_image_names, _write_positive_labels
from prescience.datasets.manifest import (
    labeled_image_names,
    load_or_create_manifest,
    save_manifest,
    should_process,
    upsert_record,
)
from prescience.pipeline.enroll import (
    append_frames_to_existing_dataset,
    next_enrollment_video_path,
    normalize_sku_name,
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


def test_next_enrollment_video_path_increments_index(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw" / "videos"
    first = next_enrollment_video_path(raw_videos_root=raw_root, sku="canA")
    assert first.name == "canA_0.MOV"

    first.touch()
    second = next_enrollment_video_path(raw_videos_root=raw_root, sku="canA")
    assert second.name == "canA_1.MOV"

    (raw_root / "canA" / "canA_7.mp4").touch()
    third = next_enrollment_video_path(raw_videos_root=raw_root, sku="canA")
    assert third.name == "canA_8.MOV"


def test_append_frames_to_existing_dataset_renumbers_frames(tmp_path: Path) -> None:
    target_dir = tmp_path / "frames" / "target"
    source_dir = tmp_path / "frames" / "source"
    target_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    (target_dir / "000001.jpg").write_bytes(b"a")
    (target_dir / "000002.jpg").write_bytes(b"b")
    (source_dir / "000001.jpg").write_bytes(b"c")
    (source_dir / "000002.jpg").write_bytes(b"d")

    moved = append_frames_to_existing_dataset(source_frames_dir=source_dir, target_frames_dir=target_dir)
    assert [path.name for path in moved] == ["000003.jpg", "000004.jpg"]
    assert sorted(path.name for path in target_dir.glob("*.jpg")) == [
        "000001.jpg",
        "000002.jpg",
        "000003.jpg",
        "000004.jpg",
    ]
    assert list(source_dir.glob("*.jpg")) == []


def test_normalize_sku_name_rejects_invalid_value() -> None:
    assert normalize_sku_name("  sku_1  ") == "sku_1"
    try:
        normalize_sku_name("bad sku")
    except ValueError:
        return
    raise AssertionError("Expected ValueError for invalid SKU name")


def test_select_processable_image_names_prefers_latest_append_batch(tmp_path: Path) -> None:
    frames_dir = tmp_path / "derived" / "frames" / "can1_test" / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    images = [frames_dir / "000001.jpg", frames_dir / "000002.jpg", frames_dir / "000003.jpg"]
    for image in images:
        image.write_bytes(b"x")

    meta_path = frames_dir.parent / "meta.json"
    meta_path.write_text(
        (
            '{"append_history": ['
            '{"video_path":"v1","new_files":["000002.jpg"]},'
            '{"video_path":"v2","new_files":["000003.jpg"]}'
            "]}"
        ),
        encoding="utf-8",
    )

    processable = {"000001.jpg", "000002.jpg", "000003.jpg"}
    selected = _select_processable_image_names(
        images=images,
        processable_names=processable,
        frames_dir=frames_dir,
        overwrite=False,
    )
    assert selected == {"000003.jpg"}


def test_select_processable_image_names_overwrite_uses_all(tmp_path: Path) -> None:
    frames_dir = tmp_path / "derived" / "frames" / "can1_test" / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    images = [frames_dir / "000001.jpg", frames_dir / "000002.jpg"]
    for image in images:
        image.write_bytes(b"x")

    (frames_dir.parent / "meta.json").write_text(
        '{"append_history":[{"video_path":"v1","new_files":["000002.jpg"]}]}',
        encoding="utf-8",
    )

    processable = {"000001.jpg", "000002.jpg"}
    selected = _select_processable_image_names(
        images=images,
        processable_names=processable,
        frames_dir=frames_dir,
        overwrite=True,
    )
    assert selected == processable


def test_select_processable_image_names_does_not_fallback_to_original_when_append_exists(tmp_path: Path) -> None:
    frames_dir = tmp_path / "derived" / "frames" / "can1_test" / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    images = [frames_dir / "000001.jpg", frames_dir / "000002.jpg"]
    for image in images:
        image.write_bytes(b"x")

    (frames_dir.parent / "meta.json").write_text(
        '{"append_history":[{"video_path":"v1","new_files":["000002.jpg"]}]}',
        encoding="utf-8",
    )

    # 000002.jpg already labeled; only original 000001.jpg remains processable.
    processable = {"000001.jpg"}
    selected = _select_processable_image_names(
        images=images,
        processable_names=processable,
        frames_dir=frames_dir,
        overwrite=False,
    )
    assert selected == set()


def test_write_positive_labels_writes_multiple_boxes(tmp_path: Path) -> None:
    label_path = tmp_path / "labels" / "000001.txt"
    _write_positive_labels(
        label_path=label_path,
        class_id=0,
        yolo_boxes=[
            (0.5, 0.5, 0.2, 0.2),
            (0.3, 0.3, 0.1, 0.1),
        ],
    )
    lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("0 ")
    assert lines[1].startswith("0 ")
