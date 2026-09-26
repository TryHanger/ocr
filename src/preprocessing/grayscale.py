"""P1 — Grayscale Preprocessing Primitive."""

from __future__ import annotations

import cv2
import numpy as np

from src.core.schemas import PreprocessingSpec
from src.preprocessing.base import BasePreprocessingPrimitive


class GrayscalePreprocessor(BasePreprocessingPrimitive):
    """Convert input image to single-channel 2D grayscale uint8.

    Conventions:
    - Input: RGB (H, W, 3), RGBA (H, W, 4), Grayscale (H, W, 1) or (H, W), dtype uint8.
    - Output: 2D array (H, W), dtype uint8, values in [0, 255].
    - Color mapping: cv2.COLOR_RGB2GRAY (standard perceptual weights: 0.299 R + 0.587 G + 0.114 B).
    - If already 2D (H, W), returns an exact copy without modification.
    """

    def _process(self, image: np.ndarray, spec: PreprocessingSpec) -> np.ndarray:
        """Convert image to 2D grayscale uint8."""
        if image.ndim == 2:
            return image.copy()

        if image.ndim == 3:
            channels = image.shape[2]
            if channels == 1:
                return image[:, :, 0].copy()
            elif channels == 3:
                return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif channels == 4:
                return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            else:
                raise ValueError(f"Unsupported channel count for grayscale: {channels}")

        raise ValueError(f"Unsupported image shape for grayscale: {image.shape}")
