"""D8 — Synthetic Shadow degradation primitive."""

from __future__ import annotations

import math
import numpy as np

from src.core.schemas import DegradationSpec
from src.degradation.base import BaseDegradationPrimitive


class ShadowDegradation(BaseDegradationPrimitive):
    """Synthetic illumination and shadow degradation with smooth boundaries.

    Simulates ambient document shadows (e.g. hand, phone, or uneven lighting).
    """

    def _apply(self, image: np.ndarray, spec: DegradationSpec) -> np.ndarray:
        params = spec.parameters
        opacity = float(params.get("opacity", 0.4))
        if not (0.0 <= opacity <= 1.0):
            raise ValueError(f"opacity must be in [0.0, 1.0], got {opacity}")

        if opacity == 0.0:
            return image

        angle = float(params.get("angle", 45.0))
        coverage = float(params.get("coverage", 0.5))
        if not (0.0 < coverage <= 1.0):
            raise ValueError(f"coverage must be in (0.0, 1.0], got {coverage}")

        smoothness = float(params.get("smoothness", 0.15))
        smoothness = max(1e-4, smoothness)

        h, w = image.shape[:2]
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)

        # Coordinate grid
        ys, xs = np.mgrid[0:h, 0:w]
        proj = xs * cos_a + ys * sin_a
        p_min, p_max = proj.min(), proj.max()
        if p_max > p_min:
            proj_norm = (proj - p_min) / (p_max - p_min)
        else:
            proj_norm = np.zeros_like(proj)

        # Seed can optionally offset the shadow boundary position
        rng = np.random.default_rng(spec.seed)
        offset = float(rng.uniform(-0.05, 0.05)) if spec.parameters.get("random_offset", False) else 0.0
        boundary = (1.0 - coverage) + offset

        # Sigmoid transition for soft, natural shadow falloff
        k = 10.0 / smoothness
        weight = 1.0 / (1.0 + np.exp(-k * (proj_norm - boundary)))

        # Mask: 1.0 (unshadowed) -> (1.0 - opacity) (shadowed)
        attenuation = 1.0 - opacity * weight
        if image.ndim == 3:
            mask = attenuation[:, :, np.newaxis]
        else:
            mask = attenuation

        shadowed = np.clip(image.astype(np.float32) * mask, 0.0, 255.0).astype(np.uint8)
        return shadowed
