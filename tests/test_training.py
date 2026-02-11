from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import prescience.datasets.yolo as yolo_mod
from prescience.datasets.yolo import TrainConfig, train_yolo_model
from prescience.pipeline.enroll import resolve_base_model_for_sku
from prescience.training.state import TrainState, select_training_names
from prescience.training.strategy import resolve_detector_training_config, resolve_onboarding_training_config


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
    assert quick.freeze == 10
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


def test_train_yolo_model_forwards_optional_kwargs(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeYOLO:
        def __init__(self, _model: str):
            self.trainer = SimpleNamespace(save_dir="")

        def train(self, **kwargs):
            captured.update(kwargs)
            save_dir = Path(kwargs["project"]) / kwargs["name"]
            (save_dir / "weights").mkdir(parents=True, exist_ok=True)
            (save_dir / "weights" / "best.pt").write_bytes(b"pt")
            self.trainer.save_dir = str(save_dir)

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

