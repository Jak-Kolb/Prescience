"""Embedding profile matching utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from prescience.profiles.schema import SKUProfile


@dataclass(frozen=True)
class MatchResult:
    sku_id: str
    score: float
    is_unknown: bool


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def match_embedding(
    embedding: np.ndarray,
    profiles: list[tuple[SKUProfile, np.ndarray]],
) -> MatchResult:
    """Return best matching SKU or UNKNOWN based on profile thresholds."""
    best_sku = "UNKNOWN"
    best_score = -1.0
    threshold = 1.0

    for profile, profile_embeddings in profiles:
        if profile_embeddings.size == 0:
            continue
        scores = profile_embeddings @ embedding
        score = float(np.max(scores))
        if score > best_score:
            best_score = score
            best_sku = profile.metadata.sku_id
            threshold = profile.metadata.threshold

    if best_score < threshold:
        return MatchResult(sku_id="UNKNOWN", score=max(best_score, 0.0), is_unknown=True)

    return MatchResult(sku_id=best_sku, score=best_score, is_unknown=False)
