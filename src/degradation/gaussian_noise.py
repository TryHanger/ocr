"""D3 — Additive Gaussian Noise degradation primitive."""

from __future__ import annotations

import numpy as np

from src.core.schemas import DegradationSpec
from src.degradation.base import BaseDegradationPrimitive


class GaussianNoiseDegradation(BaseDegradationPrimitive):
    """Additive Gaussian noise degradation with deterministic RNG seeding."""

    def _apply(self, image: np.ndarray, spec: DegradationSpec) -> np.ndarray:
        params = spec.parameters
        mean = float(params.get("mean", 0.0))
        std = float(params.get("std", 10.0))
        if std < 0:
            raise ValueError(f"std cannot be negative, got {std}")

        if std == 0.0 and mean == 0.0:
            return image

        rng = np.random.default_rng(spec.seed)
        noise = rng.normal(loc=mean, scale=std, size=image.shape).astype(np.float32)

        noisy = np.clip(image.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)
        return noisy
