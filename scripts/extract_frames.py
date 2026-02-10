# CLI wrapper for extracting frames from a product enrollment video.

import argparse
from pathlib import Path

from prescience.ingest.video_to_frames import ExtractParams, extract_frames


def main() -> None:
    p = argparse.ArgumentParser()

    # Path to your iPhone video file.
    p.add_argument("--video", required=True, help="Path to input video (mp4/mov)")

    # SKU folder name (e.g., can1_test)
    p.add_argument("--sku", required=True, help="SKU identifier (output folder name)")

    # Output root where extracted frames will go.
    p.add_argument("--out-root", default="data/derived/frames", help="Root output dir")

    # How many frames to save.
    p.add_argument("--target", type=int, default=150, help="Number of frames to save")

    # Blur threshold.
    p.add_argument("--blur-min", type=float, default=100.0, help="Min blur score (higher=stricter)")

    # Dedupe threshold (similarity).
    p.add_argument("--dedupe-sim", type=float, default=0.92, help="Max similarity allowed (higher=stricter)")

    args = p.parse_args()

    params = ExtractParams(
        target_frames=args.target,
        blur_min=args.blur_min,
        dedupe_max_similarity=args.dedupe_sim,
    )

    meta = extract_frames(
        video_path=Path(args.video),
        sku=args.sku,
        out_root=Path(args.out_root),
        params=params,
    )

    print("✅ Extraction complete")
    print(f"Saved {meta['num_frames_saved']} frames to: {meta['frames_dir']}")
    print(f"Meta written to: {Path(meta['out_dir']) / 'meta.json'}")


if __name__ == "__main__":
    main()
