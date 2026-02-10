"""
Video -> frames extraction for enrollment videos.

Simplified selection strategy (training-friendly):
- Split the video timeline into 6 equal sections.
- From each section, pick the best (sharpest) 25 frames.
- Dedupe inside each section: only keep a frame if its hash similarity to
  already-selected frames in that section is < 0.98.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class ExtractParams:
    # Total frames to save. If this doesn't match sections*frames_per_section,
    # we will derive frames_per_section from target_frames.
    target_frames: int = 150

    # How many frames to *consider* across the whole video (evenly spaced).
    # Higher = more choices per section, slower.
    candidate_frames: int = 600

    # Skip the first few frames (auto-exposure / focus settling).
    warmup_frames: int = 10

    # If > 0, discard frames with blur score below this.
    # Set to 0 to disable hard blur rejection.
    blur_min: float = 4.0

    # Dedupe rule: a frame is considered a duplicate if similarity >= this value.
    # You asked for "dedupe_sim less than 0.98", meaning:
    # keep frame only if max similarity to selected < 0.98
    dedupe_max_similarity: float = 0.98

    # Split video into this many equal time sections.
    sections: int = 6

    # Frames to take per section (default 25 -> 6*25=150)
    frames_per_section: int = 25

    # Hash resolution for dedupe (bigger = more sensitive, slower).
    hash_width: int = 32
    hash_height: int = 32


def _variance_of_laplacian(gray: np.ndarray) -> float:
    """Blur metric: higher means sharper."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def _avg_hash(gray: np.ndarray, w: int, h: int) -> np.ndarray:
    """
    Average hash (aHash):
    - resize to w x h
    - threshold pixels by mean
    """
    small = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
    mean = small.mean()
    return (small > mean)


def _hash_similarity(h1: np.ndarray, h2: np.ndarray) -> float:
    """Fraction of equal bits (1.0 = identical)."""
    return float((h1 == h2).mean())


def extract_frames(
    video_path: str | Path,
    sku: str,
    out_root: str | Path = "data/derived/frames",
    params: ExtractParams = ExtractParams(),
) -> dict[str, Any]:
    video_path = Path(video_path)
    out_root = Path(out_root)

    # Output folders
    out_dir = out_root / sku
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Video reports 0 frames. Cannot section evenly.")

    # Compute the usable range of frame indices
    start = min(params.warmup_frames, total_frames - 1)
    end = total_frames - 1
    usable_len = max(1, end - start + 1)

    # Derive frames_per_section from target_frames if needed
    sections = max(1, int(params.sections))
    if params.target_frames != sections * params.frames_per_section:
        if params.target_frames % sections != 0:
            cap.release()
            raise ValueError(
                f"target_frames={params.target_frames} must be divisible by sections={sections}"
            )
        frames_per_section = params.target_frames // sections
    else:
        frames_per_section = params.frames_per_section

    # Decide which frame indices to sample (evenly spread across the usable range)
    sample_count = min(int(params.candidate_frames), usable_len)
    sampled = np.linspace(start, end, num=sample_count, dtype=int)
    sampled_indices = sorted(set(int(x) for x in sampled.tolist()))
    sampled_set = set(sampled_indices)

    # Each bin holds tuples: (blur_score, frame_idx, hash_bits)
    bins: list[list[tuple[float, int, np.ndarray]]] = [[] for _ in range(sections)]

    # Read the video sequentially once and only process sampled indices
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if idx in sampled_set:
            # Assign to a section based on timeline position
            # rel in [0, 1)
            rel = (idx - start) / usable_len
            b = int(rel * sections)
            b = max(0, min(sections - 1, b))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_score = _variance_of_laplacian(gray)

            # Optional hard blur rejection
            if blur_score >= params.blur_min:
                hbits = _avg_hash(gray, params.hash_width, params.hash_height)
                bins[b].append((blur_score, idx, hbits))

        idx += 1

    cap.release()

    # For each section, select the best frames with dedupe constraint
    selected_per_bin: list[list[tuple[float, int, np.ndarray]]] = [[] for _ in range(sections)]
    for b in range(sections):
        # Sort candidates in this bin by sharpness (highest first)
        bins[b].sort(key=lambda t: t[0], reverse=True)

        chosen: list[tuple[float, int, np.ndarray]] = []
        chosen_hashes: list[np.ndarray] = []

        # Pass 1: enforce dedupe (similarity must be < threshold)
        for blur_score, fidx, hbits in bins[b]:
            is_dup = any(
                _hash_similarity(hbits, prev) >= params.dedupe_max_similarity
                for prev in chosen_hashes
            )
            if is_dup:
                continue

            chosen.append((blur_score, fidx, hbits))
            chosen_hashes.append(hbits)

            if len(chosen) >= frames_per_section:
                break

        # Pass 2 (fallback): if not enough, fill with next best ignoring dedupe
        if len(chosen) < frames_per_section:
            chosen_ids = {fidx for _, fidx, _ in chosen}
            for blur_score, fidx, hbits in bins[b]:
                if fidx in chosen_ids:
                    continue
                chosen.append((blur_score, fidx, hbits))
                chosen_ids.add(fidx)
                if len(chosen) >= frames_per_section:
                    break

        selected_per_bin[b] = chosen

    # Flatten selection in bin order (section 0 then 1 ...), keeping each bin's chosen order
    selected_all: list[tuple[int, float, int]] = []
    for b in range(sections):
        for blur_score, fidx, _hbits in selected_per_bin[b]:
            selected_all.append((b, blur_score, fidx))

    # Sort by (bin, frame_idx) so outputs look time-ordered within each section
    selected_all.sort(key=lambda t: (t[0], t[2]))

    # Create mapping from frame index -> output filename index
    frame_to_outnum: dict[int, int] = {}
    for outnum, (_b, _blur, fidx) in enumerate(selected_all, start=1):
        frame_to_outnum[fidx] = outnum

    # Save frames: read video again sequentially and write selected frames
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not reopen video: {video_path}")

    saved_paths: list[str] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if idx in frame_to_outnum:
            outnum = frame_to_outnum[idx]
            filename = f"{outnum:06d}.jpg"
            out_path = frames_dir / filename
            cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            saved_paths.append(str(out_path))

        idx += 1

    cap.release()

    # Metadata
    meta: dict[str, Any] = {
        "sku": sku,
        "video_path": str(video_path),
        "out_dir": str(out_dir),
        "frames_dir": str(frames_dir),
        "video_fps": fps,
        "video_total_frames": total_frames,
        "strategy": {
            "sections": sections,
            "frames_per_section": frames_per_section,
            "dedupe_keep_if_similarity_lt": params.dedupe_max_similarity,
        },
        "params": {
            "target_frames": params.target_frames,
            "candidate_frames": params.candidate_frames,
            "warmup_frames": params.warmup_frames,
            "blur_min": params.blur_min,
            "dedupe_max_similarity": params.dedupe_max_similarity,
            "hash_width": params.hash_width,
            "hash_height": params.hash_height,
        },
        "bin_stats": [
            {
                "bin": b,
                "candidates": len(bins[b]),
                "selected": len(selected_per_bin[b]),
            }
            for b in range(sections)
        ],
        "num_frames_saved": len(saved_paths),
        "saved_filenames": [Path(p).name for p in saved_paths],
    }

    meta_path = out_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta
