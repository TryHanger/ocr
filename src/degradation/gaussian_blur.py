"""D1 — Gaussian Blur degradation primitive."""

from __future__ import annotations

import cv2
import numpy as np

from src.core.schemas import DegradationSpec
from src.degradation.base import BaseDegradationPrimitive


class GaussianBlurDegradation(BaseDegradationPrimitive):
    """Gaussian blur degradation with configurable kernel size and sigma."""

    def _apply(self, image: np.ndarray, spec: DegradationSpec) -> np.ndarray:
        params = spec.parameters
        sigma = float(params.get("sigma", 1.0))
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        kernel_size = params.get("kernel_size")
        if kernel_size is not None:
            if not isinstance(kernel_size, int) or isinstance(kernel_size, bool):
                raise TypeError(f"kernel_size must be int, got {type(kernel_size).__name__}")
            if kernel_size <= 0 or kernel_size % 2 == 0:
                raise ValueError(f"kernel_size must be positive odd integer, got {kernel_size}")
            k = kernel_size
        else:
            # Derive standard odd kernel size covering approx +/- 3 sigma
            k = int(round(sigma * 6.0))
            if k % 2 == 0:
                k += 1
            k = max(3, k)

        blurred = cv2.GaussianBlur(image, (k, k), sigmaX=sigma, sigmaY=sigma)
        return blurred
