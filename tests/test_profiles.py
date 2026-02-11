from __future__ import annotations

import numpy as np

from prescience.profiles.schema import ProfileMetadata, ProfileModelInfo, SKUProfile
from prescience.vision.matcher import cosine_similarity, match_embedding


def make_profile(sku_id: str, threshold: float, embeddings: np.ndarray) -> tuple[SKUProfile, np.ndarray]:
    profile = SKUProfile(
        metadata=ProfileMetadata(
            sku_id=sku_id,
            name=sku_id,
            threshold=threshold,
            model=ProfileModelInfo(backbone="resnet18", preprocess_version="v1", embedding_dim=embeddings.shape[1]),
            num_embeddings=embeddings.shape[0],
        )
    )
    return profile, embeddings.astype(np.float32)


def test_cosine_similarity_basic() -> None:
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0], dtype=np.float32)
    c = np.array([-1.0, 0.0], dtype=np.float32)

    assert cosine_similarity(a, b) == 1.0
    assert cosine_similarity(a, c) == -1.0


def test_match_embedding_returns_best_known_sku() -> None:
    emb = np.array([1.0, 0.0], dtype=np.float32)

    p1 = make_profile("sku-a", 0.70, np.array([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32))
    p2 = make_profile("sku-b", 0.70, np.array([[0.0, 1.0], [0.2, 0.8]], dtype=np.float32))

    result = match_embedding(emb, [p1, p2])
    assert result.sku_id == "sku-a"
    assert result.is_unknown is False
    assert result.score > 0.9


def test_match_embedding_returns_unknown_when_below_threshold() -> None:
    emb = np.array([-1.0, 0.0], dtype=np.float32)

    p1 = make_profile("sku-a", 0.60, np.array([[1.0, 0.0]], dtype=np.float32))
    p2 = make_profile("sku-b", 0.60, np.array([[0.0, 1.0]], dtype=np.float32))

    result = match_embedding(emb, [p1, p2])
    assert result.sku_id == "UNKNOWN"
    assert result.is_unknown is True
