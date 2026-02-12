from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import cv2
import numpy as np

import prescience.pipeline.web_onboarding as web


def _write_frame(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((120, 160, 3), 200, dtype=np.uint8)
    cv2.imwrite(str(path), image)


def test_prepare_seed_candidates_default_count(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    sku = "can1_test"
    frames = Path(f"data/derived/frames/{sku}/frames")
    for idx in range(1, 61):
        _write_frame(frames / f"{idx:06d}.jpg")

    candidates = web.prepare_seed_candidates(sku)
    assert len(candidates) == 24
    assert all(item["status"] == "pending" for item in candidates)


def test_save_browser_label_supports_multibox_and_negative(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    sku = "can1_test"
    frames = Path(f"data/derived/frames/{sku}/frames")
    _write_frame(frames / "000001.jpg")
    _write_frame(frames / "000002.jpg")

    web.save_browser_label(
        sku=sku,
        frame_name="000001.jpg",
        status="positive",
        boxes=[
            {"x1": 10, "y1": 10, "x2": 40, "y2": 50},
            {"x1": 50, "y1": 20, "x2": 90, "y2": 70},
        ],
    )
    web.save_browser_label(
        sku=sku,
        frame_name="000002.jpg",
        status="negative",
        boxes=[],
    )

    pos = Path(f"data/derived/labels/{sku}/labels/000001.txt").read_text(encoding="utf-8").strip().splitlines()
    neg = Path(f"data/derived/labels/{sku}/labels/000002.txt").read_text(encoding="utf-8")
    assert len(pos) == 2
    assert neg == ""


def test_prepare_approval_candidates_returns_proposals(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    sku = "can1_test"
    frames = Path(f"data/derived/frames/{sku}/frames")
    for idx in range(1, 13):
        _write_frame(frames / f"{idx:06d}.jpg")

    class FakeBoxes:
        def __init__(self):
            self.xyxy = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.array([[5, 5, 50, 60], [60, 8, 95, 90]])))
            self.conf = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.array([0.8, 0.6])))

        def __len__(self):
            return 2

    class FakeResult:
        boxes = FakeBoxes()

    class FakeYOLO:
        def __init__(self, _path: str):
            pass

        def predict(self, **_kwargs):
            return [FakeResult()]

    monkeypatch.setattr(web, "YOLO", FakeYOLO)
    candidates = web.prepare_approval_candidates(
        sku=sku,
        stage1_model_path="fake-stage1.pt",
        approve_per_section=1,
        sections=6,
    )
    assert len(candidates) == 6
    assert all(len(item["proposals"]) == 2 for item in candidates)


def test_train_stage2_updates_manifest_and_stable_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    sku = "can1_test"
    frames = Path(f"data/derived/frames/{sku}/frames")
    labels = Path(f"data/derived/labels/{sku}/labels")
    for idx in range(1, 7):
        name = f"{idx:06d}"
        _write_frame(frames / f"{name}.jpg")
        (labels / f"{name}.txt").parent.mkdir(parents=True, exist_ok=True)
        (labels / f"{name}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    stage1 = Path("data/models/yolo/can1_test_stage1/best.pt")
    stage1.parent.mkdir(parents=True, exist_ok=True)
    stage1.write_bytes(b"pt")

    def fake_train_yolo_model(*, data_yaml: Path, model_out_dir: Path, config, progress_cb=None):  # noqa: ANN001
        _ = data_yaml
        _ = config
        model_out_dir.mkdir(parents=True, exist_ok=True)
        best = model_out_dir / "best.pt"
        best.write_bytes(b"pt")
        if progress_cb:
            progress_cb({"epoch": 1, "total_epochs": 1, "message": "done"})
        return best

    monkeypatch.setattr(web, "train_yolo_model", fake_train_yolo_model)
    best = web.train_stage2_for_session(
        sku=sku,
        version_num=1,
        stage1_model_path=str(stage1),
        mode="quick",
    )
    assert best == "data/models/yolo/can1_test_v1/best.pt"
    assert Path(best).exists()

    manifest = Path(f"data/derived/labels/{sku}/manifest.json").read_text(encoding="utf-8")
    assert "\"model_version\": \"v1\"" in manifest


def test_prepare_approval_candidates_uses_only_current_append_video_batch(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    sku = "can1_test"
    frames = Path(f"data/derived/frames/{sku}/frames")
    for idx in range(1, 7):
        _write_frame(frames / f"{idx:06d}.jpg")

    meta = {
        "sku": sku,
        "append_mode": True,
        "append_history": [
            {
                "video_path": f"data/raw/videos/{sku}/{sku}_1.MOV",
                "new_files": ["000001.jpg", "000002.jpg", "000003.jpg"],
            },
            {
                "video_path": f"data/raw/videos/{sku}/{sku}_2.MOV",
                "new_files": ["000004.jpg", "000005.jpg", "000006.jpg"],
            },
        ],
    }
    (frames.parent / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    class FakeBoxes:
        def __init__(self):
            self.xyxy = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.array([[5, 5, 50, 60]])))
            self.conf = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.array([0.8])))

        def __len__(self):
            return 1

    class FakeResult:
        boxes = FakeBoxes()

    class FakeYOLO:
        def __init__(self, _path: str):
            pass

        def predict(self, **_kwargs):
            return [FakeResult()]

    monkeypatch.setattr(web, "YOLO", FakeYOLO)
    candidates = web.prepare_approval_candidates(
        sku=sku,
        stage1_model_path="fake-stage1.pt",
        approve_per_section=10,
        sections=1,
        append_video_path=f"data/raw/videos/{sku}/{sku}_2.MOV",
    )
    names = sorted(item["frame_name"] for item in candidates)
    assert names == ["000004.jpg", "000005.jpg", "000006.jpg"]


def test_prepare_approval_candidates_from_append_batch_yields_full_target_count(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    sku = "can1_test"
    frames = Path(f"data/derived/frames/{sku}/frames")
    for idx in range(1, 181):
        _write_frame(frames / f"{idx:06d}.jpg")

    latest_video = f"data/raw/videos/{sku}/{sku}_2.MOV"
    meta = {
        "sku": sku,
        "append_mode": True,
        "append_history": [
            {
                "video_path": f"data/raw/videos/{sku}/{sku}_1.MOV",
                "new_files": [f"{idx:06d}.jpg" for idx in range(121, 151)],
            },
            {
                "video_path": latest_video,
                "new_files": [f"{idx:06d}.jpg" for idx in range(151, 181)],
            },
        ],
    }
    (frames.parent / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    class FakeBoxes:
        def __init__(self):
            self.xyxy = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.array([[5, 5, 50, 60]])))
            self.conf = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.array([0.8])))

        def __len__(self):
            return 1

    class FakeResult:
        boxes = FakeBoxes()

    class FakeYOLO:
        def __init__(self, _path: str):
            pass

        def predict(self, **_kwargs):
            return [FakeResult()]

    monkeypatch.setattr(web, "YOLO", FakeYOLO)
    candidates = web.prepare_approval_candidates(
        sku=sku,
        stage1_model_path="fake-stage1.pt",
        approve_per_section=5,
        sections=6,
        append_video_path=latest_video,
    )
    assert len(candidates) == 30
    assert all("000151.jpg" <= item["frame_name"] <= "000180.jpg" for item in candidates)
