"""P4 — Adaptive Binarization Preprocessing Primitive."""

from __future__ import annotations

from typing import Any, Dict
import cv2
import numpy as np

from src.core.schemas import PreprocessingSpec
from src.preprocessing.base import BasePreprocessingPrimitive


class BinarizationPreprocessor(BasePreprocessingPrimitive):
    """Adaptive document binarization primitive.

    Conventions:
    - Input: 2D (H, W) or 3D (H, W, C) uint8.
    - Output: 2D array (H, W), dtype uint8.
    - Output values: Strictly binary values {0, 255} where 0 = black (foreground text)
      and 255 = white (background).
    - Color-space handling: Automatically converts color images to grayscale before thresholding.
    - Supported methods:
      1. 'gaussian': cv2.adaptiveThreshold with ADAPTIVE_THRESH_GAUSSIAN_C.
      2. 'mean': cv2.adaptiveThreshold with ADAPTIVE_THRESH_MEAN_C.
      3. 'otsu': global Otsu thresholding.
    """

    SUPPORTED_METHODS = ("gaussian", "mean", "otsu")

    def _get_params(self, spec: PreprocessingSpec) -> Dict[str, Any]:
        params = spec.parameters.get("binarization", spec.parameters)
        return dict(params) if isinstance(params, dict) else {}

    def _process(self, image: np.ndarray, spec: PreprocessingSpec) -> np.ndarray:
        """Apply adaptive or Otsu binarization to image."""
        params = self._get_params(spec)
        method = str(params.get("method", "gaussian")).lower()

        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported binarization method '{method}'. Expected one of {self.SUPPORTED_METHODS}"
            )

        # 1. Convert to 2D grayscale if necessary
        if image.ndim == 2:
            gray = image
        elif image.ndim == 3:
            channels = image.shape[2]
            if channels == 1:
                gray = image[:, :, 0]
            elif channels == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif channels == 4:
                gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            else:
                raise ValueError(f"Unsupported channel count for binarization: {channels}")
        else:
            raise ValueError(f"Unsupported image shape for binarization: {image.shape}")

        # 2. Apply thresholding
        if method == "otsu":
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary

        # Adaptive Gaussian or Mean
        block_size = int(params.get("block_size", 11))
        c = float(params.get("c", 2.0))

        if block_size < 3 or block_size % 2 == 0:
            raise ValueError(f"block_size must be an odd integer >= 3, got {block_size}")

        adaptive_method = (
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            if method == "gaussian"
            else cv2.ADAPTIVE_THRESH_MEAN_C
        )

        binary = cv2.adaptiveThreshold(
            gray,
            maxValue=255,
            adaptiveMethod=adaptive_method,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=block_size,
            C=c,
        )

        return binary
