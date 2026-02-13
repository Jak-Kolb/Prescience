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


def test_prepare_seed_candidates_with_gemini_writes_labels(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    sku = "can1_test"
    frames = Path(f"data/derived/frames/{sku}/frames")
    for idx in range(1, 31):
        _write_frame(frames / f"{idx:06d}.jpg")

    def _fake_batch(**kwargs):  # noqa: ANN003
        image_paths = kwargs["image_paths"]
        proposals: dict[str, list[dict[str, int]]] = {}
        errors: dict[str, str] = {}
        for i, path in enumerate(image_paths):
            if i % 2 == 0:
                proposals[path.name] = [{"x1": 10, "y1": 12, "x2": 60, "y2": 70}]
            else:
                proposals[path.name] = []
        return proposals, errors

    monkeypatch.setattr(web, "_gemini_batch_proposals_for_images", _fake_batch)

    candidates = web.prepare_seed_candidates_with_gemini(
        sku=sku,
        object_description=sku,
        target_count=24,
        max_proposals_per_frame=4,
    )
    assert len(candidates) == 24
    assert all(item["status"] == "pending" for item in candidates)

    labels = Path(f"data/derived/labels/{sku}/labels")
    positive = 0
    negative = 0
    for candidate in candidates:
        text = (labels / f"{Path(candidate['frame_name']).stem}.txt").read_text(encoding="utf-8")
        if text.strip():
            positive += 1
        else:
            negative += 1
    assert positive == 12
    assert negative == 12


def test_prepare_approval_candidates_with_gemini_uses_append_batch_only(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    sku = "can1_test"
    frames = Path(f"data/derived/frames/{sku}/frames")
    for idx in range(1, 61):
        _write_frame(frames / f"{idx:06d}.jpg")

    latest_video = f"data/raw/videos/{sku}/{sku}_2.MOV"
    meta = {
        "sku": sku,
        "append_mode": True,
        "append_history": [
            {
                "video_path": f"data/raw/videos/{sku}/{sku}_1.MOV",
                "new_files": [f"{idx:06d}.jpg" for idx in range(1, 31)],
            },
            {
                "video_path": latest_video,
                "new_files": [f"{idx:06d}.jpg" for idx in range(31, 61)],
            },
        ],
    }
    (frames.parent / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    def _fake_batch(**kwargs):  # noqa: ANN003
        proposals = {path.name: [{"x1": 5, "y1": 5, "x2": 25, "y2": 30}] for path in kwargs["image_paths"]}
        return proposals, {}

    monkeypatch.setattr(web, "_gemini_batch_proposals_for_images", _fake_batch)
    candidates = web.prepare_approval_candidates_with_gemini(
        sku=sku,
        object_description=sku,
        target_count=30,
        append_video_path=latest_video,
    )
    assert len(candidates) == 30
    assert all("000031.jpg" <= item["frame_name"] <= "000060.jpg" for item in candidates)


def test_validate_detector_candidates_with_gemini_decisions(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    sku = "can1_test"
    frames = Path(f"data/derived/frames/{sku}/frames")
    for idx in range(1, 5):
        _write_frame(frames / f"{idx:06d}.jpg")

    detector = [
        web.FrameProposal(frame_name="000001.jpg", boxes=[{"x1": 10, "y1": 10, "x2": 40, "y2": 40}]),
        web.FrameProposal(frame_name="000002.jpg", boxes=[{"x1": 8, "y1": 8, "x2": 18, "y2": 18}]),
        web.FrameProposal(frame_name="000003.jpg", boxes=[]),
        web.FrameProposal(frame_name="000004.jpg", boxes=[{"x1": 1, "y1": 1, "x2": 5, "y2": 5}]),
    ]

    def _fake_batch(**kwargs):  # noqa: ANN003
        names = [path.name for path in kwargs["image_paths"]]
        proposals = {name: [] for name in names}
        errors = {}
        proposals["000001.jpg"] = [{"x1": 11, "y1": 11, "x2": 39, "y2": 39}]  # accept (high IoU)
        proposals["000002.jpg"] = [{"x1": 60, "y1": 60, "x2": 80, "y2": 80}]  # adjust
        proposals["000003.jpg"] = []  # reject_negative
        errors["000004.jpg"] = "missing_api_key_env:GEMINI_API_KEY"  # uncertain
        return proposals, errors

    monkeypatch.setattr(web, "_gemini_batch_proposals_for_images", _fake_batch)
    result = web.validate_detector_candidates_with_gemini(
        sku=sku,
        detector_candidates=detector,
        object_description=sku,
        enabled=True,
        gemini_model="gemini-3-pro-preview",
        gemini_api_key_env="GEMINI_API_KEY",
        batch_size=24,
        max_boxes_approval=8,
    )
    assert result.accepted == 1
    assert result.adjusted == 1
    assert result.rejected_negative == 1
    assert result.uncertain == 1
