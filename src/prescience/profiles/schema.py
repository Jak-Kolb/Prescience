"""SKU profile schema definitions."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field


class ProfileModelInfo(BaseModel):
    backbone: str
    preprocess_version: str = "v1"
    embedding_dim: int


class ProfileMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sku_id: str
    name: str
    threshold: float = Field(ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model: ProfileModelInfo
    num_embeddings: int = Field(ge=1)


class SKUProfile(BaseModel):
    """Serializable profile metadata + embeddings filename."""

    model_config = ConfigDict(extra="forbid")

    version: str = "1"
    metadata: ProfileMetadata
    embeddings_file: str = "embeddings.npy"
