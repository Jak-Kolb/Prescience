"""Profile persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from prescience.profiles.schema import SKUProfile


PROFILE_FILENAME = "profile.json"


def save_profile(profile_dir: str | Path, profile: SKUProfile, embeddings: np.ndarray) -> None:
    """Save profile metadata JSON + embeddings matrix."""
    root = Path(profile_dir)
    root.mkdir(parents=True, exist_ok=True)

    emb_path = root / profile.embeddings_file
    np.save(emb_path, embeddings)

    profile_path = root / PROFILE_FILENAME
    with profile_path.open("w", encoding="utf-8") as f:
        json.dump(profile.model_dump(mode="json"), f, indent=2)


def load_profile(profile_dir: str | Path) -> tuple[SKUProfile, np.ndarray]:
    """Load profile metadata and embeddings matrix."""
    root = Path(profile_dir)
    profile_path = root / PROFILE_FILENAME

    with profile_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    profile = SKUProfile.model_validate(raw)
    embeddings = np.load(root / profile.embeddings_file)
    return profile, embeddings


def load_all_profiles(profiles_root: str | Path) -> list[tuple[SKUProfile, np.ndarray]]:
    """Load all profiles from profile root directory."""
    root = Path(profiles_root)
    if not root.exists():
        return []

    out: list[tuple[SKUProfile, np.ndarray]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        profile_path = child / PROFILE_FILENAME
        if not profile_path.exists():
            continue
        out.append(load_profile(child))
    return out
