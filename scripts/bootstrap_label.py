import argparse
from pathlib import Path

from prescience.datasets.bootstrap_label import OnboardingParams, run_onboarding_labeling


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sku", required=True, help="SKU name, e.g. can1_test")
    p.add_argument("--base-model", default="auto", help="Base model path or auto")
    p.add_argument("--mode", choices=["quick", "milestone", "full"], default="quick")
    p.add_argument("--dataset-scope", choices=["core_new", "all"], default=None)
    p.add_argument("--core-size", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--conf-propose", type=float, default=0.10)
    p.add_argument("--epochs1", type=int, default=None, help="Stage1 epochs override")
    p.add_argument("--epochs2", type=int, default=None, help="Stage2 epochs override")
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--freeze", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--no-retrain", action="store_true", help="Skip stage2 retraining")

    args = p.parse_args()

    params = OnboardingParams(
        sku=args.sku,
        frames_dir=Path(f"data/derived/frames/{args.sku}/frames"),
        labels_dir=Path(f"data/derived/labels/{args.sku}/labels"),
        base_model=args.base_model,
        mode=args.mode,
        dataset_scope=args.dataset_scope,
        core_size=args.core_size,
        imgsz=args.imgsz,
        conf_propose=args.conf_propose,
        epochs_stage1=args.epochs1,
        epochs_stage2=args.epochs2,
        patience=args.patience,
        freeze=args.freeze,
        workers=args.workers,
        retrain_after_approve=not args.no_retrain,
    )

    run_onboarding_labeling(params)


if __name__ == "__main__":
    main()
