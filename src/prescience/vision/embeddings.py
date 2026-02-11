"""Embedding extraction implementations."""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models


def choose_torch_device() -> torch.device:
    """Select MPS when available, otherwise CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Embedder(Protocol):
    """Embeddings interface for pluggable backbones."""

    def encode(self, image: np.ndarray) -> np.ndarray:
        """Encode BGR image into L2-normalized feature vector."""


class ResNet18Embedder:
    """Torchvision ResNet18 backbone feature extractor."""

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or choose_torch_device()
        weights = models.ResNet18_Weights.DEFAULT
        try:
            backbone = models.resnet18(weights=weights)
            transforms = weights.transforms()
        except Exception:
            # Offline fallback keeps MVP functional even without model download.
            backbone = models.resnet18(weights=None)
            transforms = models.ResNet18_Weights.IMAGENET1K_V1.transforms()
        self._model = nn.Sequential(*list(backbone.children())[:-1]).to(self.device)
        self._model.eval()
        self._preprocess = transforms

    def encode(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected BGR image with shape [H, W, 3]")

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        batch = self._preprocess(tensor).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self._model(batch).flatten().detach().cpu().numpy().astype(np.float32)

        norm = np.linalg.norm(emb)
        if norm > 0:
            emb /= norm
        return emb


def build_embedder(backbone: str = "resnet18") -> Embedder:
    """Factory for embedding backbones."""
    normalized = backbone.lower().strip()
    if normalized == "resnet18":
        return ResNet18Embedder()
    raise ValueError(f"Unsupported embedding backbone: {backbone}")
