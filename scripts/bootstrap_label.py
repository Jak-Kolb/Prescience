import argparse
from pathlib import Path

from prescience.datasets.bootstrap_label import BootstrapParams, run_bootstrap_labeling


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sku", required=True, help="SKU name, e.g. can1_test")
    p.add_argument("--base-model", default="yolov8n.pt", help="Base YOLO model for stage1 training")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf-propose", type=float, default=0.10)
    p.add_argument("--epochs1", type=int, default=30, help="Epochs for stage1 (12 images)")
    p.add_argument("--epochs2", type=int, default=60, help="Epochs for stage2 retrain (42 images)")
    p.add_argument("--no-retrain", action="store_true", help="Skip stage2 retraining")

    args = p.parse_args()

    params = BootstrapParams(
        sku=args.sku,
        frames_dir=Path(f"data/derived/frames/{args.sku}/frames"),
        labels_dir=Path(f"data/derived/labels/{args.sku}/labels"),
        base_model=args.base_model,
        imgsz=args.imgsz,
        conf_propose=args.conf_propose,
        epochs_stage1=args.epochs1,
        epochs_stage2=args.epochs2,
        retrain_after_approve=not args.no_retrain,
    )

    run_bootstrap_labeling(params)


if __name__ == "__main__":
    main()
