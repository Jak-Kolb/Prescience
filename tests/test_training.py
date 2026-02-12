from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from types import SimpleNamespace

import prescience.datasets.yolo as yolo_mod
from prescience.datasets.yolo import TrainConfig, train_yolo_model
from prescience.pipeline.enroll import resolve_base_model_for_sku
from prescience.training.state import TrainState, can_resume_from_state, compute_dataset_hash, select_training_names
from prescience.training.strategy import dynamic_quick_epochs, resolve_detector_training_config, resolve_onboarding_training_config


def test_resolve_base_model_auto_without_prior_returns_default(tmp_path: Path) -> None:
    resolved = resolve_base_model_for_sku(
        sku="can1_test",
        base_model="auto",
        target_version=1,
        models_root=tmp_path / "models" / "yolo",
    )
    assert resolved == "yolov8n.pt"


def test_resolve_base_model_auto_uses_latest_prior_version(tmp_path: Path) -> None:
    models_root = tmp_path / "models" / "yolo"
    prior = models_root / "can1_test_v1"
    prior.mkdir(parents=True, exist_ok=True)
    (prior / "best.pt").write_bytes(b"pt")

    resolved = resolve_base_model_for_sku(
        sku="can1_test",
        base_model="auto",
        target_version=2,
        models_root=models_root,
    )
    assert resolved == str(prior / "best.pt")


def test_resolve_base_model_explicit_path_overrides_auto(tmp_path: Path) -> None:
    explicit = tmp_path / "custom.pt"
    resolved = resolve_base_model_for_sku(
        sku="can1_test",
        base_model=str(explicit),
        target_version=2,
        models_root=tmp_path / "models" / "yolo",
    )
    assert resolved == str(explicit)


def test_mode_resolution_applies_defaults_and_overrides() -> None:
    quick = resolve_onboarding_training_config(
        mode="quick",
        dataset_scope=None,
        core_size=None,
        imgsz=None,
        epochs_stage1=None,
        epochs_stage2=None,
        patience=None,
        freeze=None,
        workers=None,
    )
    assert quick.mode == "quick"
    assert quick.dataset_scope == "core_new"
    assert quick.imgsz == 640
    assert quick.epochs_stage1 == 8
    assert quick.epochs_stage2 == 12
    assert quick.patience == 4
    assert quick.freeze == 5
    assert quick.core_size == 48

    overridden = resolve_detector_training_config(
        mode="milestone",
        dataset_scope="all",
        core_size=32,
        imgsz=704,
        epochs=17,
        patience=5,
        freeze=3,
        workers=1,
    )
    assert overridden.mode == "milestone"
    assert overridden.dataset_scope == "all"
    assert overridden.core_size == 32
    assert overridden.imgsz == 704
    assert overridden.epochs == 17
    assert overridden.patience == 5
    assert overridden.freeze == 3
    assert overridden.workers == 1


def test_core_new_scope_uses_core_plus_new_then_core_only() -> None:
    state = TrainState(
        core_images=["000001.jpg", "000002.jpg"],
        trained_snapshot_images=["000001.jpg", "000002.jpg"],
    )
    labeled_names = ["000001.jpg", "000002.jpg", "000003.jpg", "000004.jpg"]

    with_new = select_training_names(
        labeled_names=labeled_names,
        scope="core_new",
        core_size=2,
        train_state=state,
    )
    assert with_new.new_names == ["000003.jpg", "000004.jpg"]
    assert with_new.selected_names == ["000001.jpg", "000002.jpg", "000003.jpg", "000004.jpg"]

    no_new_state = TrainState(
        core_images=["000001.jpg", "000002.jpg"],
        trained_snapshot_images=labeled_names,
    )
    no_new = select_training_names(
        labeled_names=labeled_names,
        scope="core_new",
        core_size=2,
        train_state=no_new_state,
    )
    assert no_new.new_names == []
    assert no_new.selected_names == ["000001.jpg", "000002.jpg"]

    full = select_training_names(
        labeled_names=labeled_names,
        scope="all",
        core_size=2,
        train_state=no_new_state,
    )
    assert full.selected_names == labeled_names


def test_core_selection_is_deterministic_and_stratified_by_bins() -> None:
    labeled_names = [f"{idx:06d}.jpg" for idx in range(1, 13)]  # 12 frames
    state = TrainState()

    first = select_training_names(
        labeled_names=labeled_names,
        scope="core_new",
        core_size=6,
        train_state=state,
    )
    second = select_training_names(
        labeled_names=labeled_names,
        scope="core_new",
        core_size=6,
        train_state=state,
    )

    assert first.core_names == second.core_names
    assert len(first.core_names) == 6
    # 12 names -> 6 bins of 2 names each, core_size 6 -> one name per bin.
    bin_pairs = [set(labeled_names[i : i + 2]) for i in range(0, 12, 2)]
    for pair in bin_pairs:
        assert len(pair.intersection(first.core_names)) == 1


def test_train_yolo_model_forwards_optional_kwargs(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakePred:
        def plot(self):
            return np.zeros((10, 10, 3), dtype=np.uint8)

    class FakeYOLO:
        def __init__(self, _model: str):
            self.trainer = SimpleNamespace(save_dir="")

        def train(self, **kwargs):
            captured.update(kwargs)
            save_dir = Path(kwargs["project"]) / kwargs["name"]
            (save_dir / "weights").mkdir(parents=True, exist_ok=True)
            (save_dir / "weights" / "best.pt").write_bytes(b"pt")
            self.trainer.save_dir = str(save_dir)

        def val(self, **_kwargs):
            return SimpleNamespace(results_dict={"metrics/mAP50(B)": 0.75, "metrics/precision(B)": 0.8})

        def predict(self, **_kwargs):
            return [FakePred()]

    monkeypatch.setattr(yolo_mod, "YOLO", FakeYOLO)

    data_yaml = tmp_path / "dataset" / "data.yaml"
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    data_yaml.write_text("path: test\n", encoding="utf-8")

    best = train_yolo_model(
        data_yaml=data_yaml,
        model_out_dir=tmp_path / "models" / "out",
        config=TrainConfig(
            base_model="yolov8n.pt",
            imgsz=640,
            epochs=10,
            conf=0.25,
            patience=4,
            freeze=10,
            workers=0,
        ),
    )

    assert best.exists()
    assert captured["patience"] == 4
    assert captured["freeze"] == 10
    assert captured["workers"] == 0


def test_train_yolo_model_resume_path_uses_resume_flag(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakePred:
        def plot(self):
            return np.zeros((10, 10, 3), dtype=np.uint8)

    class FakeYOLO:
        def __init__(self, model_path: str):
            self.model_path = model_path
            self.trainer = SimpleNamespace(save_dir="")

        def train(self, **kwargs):
            captured.update(kwargs)
            if kwargs.get("resume"):
                save_dir = Path(self.model_path).parent.parent
            else:
                save_dir = tmp_path / "unexpected"
            (save_dir / "weights").mkdir(parents=True, exist_ok=True)
            (save_dir / "weights" / "best.pt").write_bytes(b"pt")
            self.trainer.save_dir = str(save_dir)

        def val(self, **_kwargs):
            return SimpleNamespace(results_dict={"metrics/mAP50(B)": 0.7})

        def predict(self, **_kwargs):
            return [FakePred()]

    monkeypatch.setattr(yolo_mod, "YOLO", FakeYOLO)

    data_yaml = tmp_path / "dataset" / "data.yaml"
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    data_yaml.write_text("path: test\n", encoding="utf-8")

    model_out = tmp_path / "models" / "can1_test_v2"
    resume_checkpoint = model_out / "train" / "weights" / "last.pt"
    resume_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    resume_checkpoint.write_bytes(b"pt")

    best = train_yolo_model(
        data_yaml=data_yaml,
        model_out_dir=model_out,
        config=TrainConfig(
            base_model="yolov8n.pt",
            imgsz=640,
            epochs=10,
            conf=0.25,
            resume=True,
            resume_checkpoint=str(resume_checkpoint),
        ),
    )

    assert best.exists()
    assert captured == {"resume": True}


def test_dynamic_quick_epochs_range() -> None:
    assert dynamic_quick_epochs(0) == 6
    assert dynamic_quick_epochs(2) == 6
    assert dynamic_quick_epochs(16) == 10
    assert dynamic_quick_epochs(1000) == 20


def test_dataset_hash_and_resume_decision(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    labels = tmp_path / "labels"
    frames.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)

    img1 = frames / "000001.jpg"
    img2 = frames / "000002.jpg"
    img1.write_bytes(b"a")
    img2.write_bytes(b"b")
    (labels / "000001.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (labels / "000002.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    dataset_hash = compute_dataset_hash(selected_images=[img1, img2], labels_dir=labels)
    state = TrainState(
        dataset_hash=dataset_hash,
        last_mode="quick",
        last_imgsz=640,
        last_base_model_path="data/models/yolo/can1_test_v1/best.pt",
        last_ultralytics_version="8.3.0",
        last_version_tag="v2",
    )

    ok = can_resume_from_state(
        train_state=state,
        requested_dataset_hash=dataset_hash,
        requested_mode="quick",
        requested_imgsz=640,
        requested_base_model_path="data/models/yolo/can1_test_v1/best.pt",
        requested_ultralytics_version="8.3.0",
        requested_version_tag="v2",
    )
    assert ok.allowed is True
    assert ok.reason == "ok"

    bad = can_resume_from_state(
        train_state=state,
        requested_dataset_hash="different",
        requested_mode="quick",
        requested_imgsz=640,
        requested_base_model_path="data/models/yolo/can1_test_v1/best.pt",
        requested_ultralytics_version="8.3.0",
        requested_version_tag="v2",
    )
    assert bad.allowed is False
    assert bad.reason == "dataset_hash_mismatch"


def test_eval_comparison_prunes_older_models_when_new_is_better(tmp_path: Path, monkeypatch) -> None:
    model_root = tmp_path / "models" / "yolo"
    v1 = model_root / "can1_test_v1"
    v2 = model_root / "can1_test_v2"
    v3 = model_root / "can1_test_v3"
    for path in (v1, v2, v3):
        path.mkdir(parents=True, exist_ok=True)
        (path / "best.pt").write_bytes(b"pt")

    dataset_dir = tmp_path / "dataset"
    (dataset_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "images" / "val" / "000001.jpg").write_bytes(b"fake")
    data_yaml = dataset_dir / "data.yaml"
    data_yaml.write_text("path: test\n", encoding="utf-8")

    def fake_eval(*, model_path: str, **_kwargs):  # noqa: ANN001
        if model_path.endswith("can1_test_v2/best.pt"):
            return {"status": "ok", "model_path": model_path, "metrics": {"metrics/mAP50-95(B)": 0.60}}
        return {"status": "ok", "model_path": model_path, "metrics": {"metrics/mAP50-95(B)": 0.71}}

    class FakePred:
        def plot(self):
            return np.zeros((10, 10, 3), dtype=np.uint8)

    class FakeYOLO:
        def __init__(self, _model_path: str):
            pass

        def predict(self, **_kwargs):
            return [FakePred()]

    monkeypatch.setattr(yolo_mod, "_evaluate_model_summary", fake_eval)
    monkeypatch.setattr(yolo_mod, "YOLO", FakeYOLO)

    yolo_mod._write_quick_eval_artifacts(
        best_model_path=v3 / "best.pt",
        old_model_path=str(v2 / "best.pt"),
        data_yaml=data_yaml,
        model_out_dir=v3,
        imgsz=640,
        conf=0.25,
        device="cpu",
    )

    assert v3.exists()
    assert not v1.exists()
    assert not v2.exists()

    metrics = json.loads((v3 / "eval" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["comparison"]["improved"] is True
