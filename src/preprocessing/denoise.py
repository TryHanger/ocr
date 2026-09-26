"""P2 — Denoising Preprocessing Primitive."""

from __future__ import annotations

from typing import Any, Dict
import cv2
import numpy as np

from src.core.schemas import PreprocessingSpec
from src.preprocessing.base import BasePreprocessingPrimitive


class DenoisePreprocessor(BasePreprocessingPrimitive):
    """Deterministic noise reduction primitive supporting median, bilateral, and fast_nl_means.

    Conventions:
    - Input: 2D (H, W) or 3D (H, W, C) uint8.
    - Output: Same shape and channels as input, dtype uint8.
    - Supported methods:
      1. 'median': cv2.medianBlur(image, ksize).
      2. 'bilateral': cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace).
      3. 'fast_nl_means': cv2.fastNlMeansDenoising or cv2.fastNlMeansDenoisingColored.
    """

    SUPPORTED_METHODS = ("median", "bilateral", "fast_nl_means")

    def _get_params(self, spec: PreprocessingSpec) -> Dict[str, Any]:
        """Extract method parameters supporting nested or flat specifications."""
        params = spec.parameters.get("denoise", spec.parameters)
        return dict(params) if isinstance(params, dict) else {}

    def _process(self, image: np.ndarray, spec: PreprocessingSpec) -> np.ndarray:
        """Apply selected denoising filter."""
        params = self._get_params(spec)
        method = str(params.get("method", "median")).lower()

        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported denoise method '{method}'. Expected one of {self.SUPPORTED_METHODS}"
            )

        if method == "median":
            ksize = int(params.get("ksize", 3))
            if ksize < 1 or ksize % 2 == 0:
                raise ValueError(f"ksize for median denoise must be an odd positive integer, got {ksize}")
            if ksize == 1:
                return image.copy()
            return cv2.medianBlur(image, ksize)

        elif method == "bilateral":
            d = int(params.get("d", 9))
            sigma_color = float(params.get("sigma_color", 75.0))
            sigma_space = float(params.get("sigma_space", 75.0))
            if d < 1:
                raise ValueError(f"d for bilateral filter must be >= 1, got {d}")
            if sigma_color <= 0 or sigma_space <= 0:
                raise ValueError("sigma_color and sigma_space for bilateral filter must be positive")
            return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

        elif method == "fast_nl_means":
            h = float(params.get("h", 10.0))
            template_window = int(params.get("template_window_size", 7))
            search_window = int(params.get("search_window_size", 21))

            if h <= 0:
                raise ValueError(f"h for fast_nl_means must be positive, got {h}")
            if template_window % 2 == 0 or template_window < 3:
                raise ValueError(f"template_window_size must be an odd integer >= 3, got {template_window}")
            if search_window % 2 == 0 or search_window < 3:
                raise ValueError(f"search_window_size must be an odd integer >= 3, got {search_window}")

            if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
                flat = image[:, :, 0] if image.ndim == 3 else image
                denoised = cv2.fastNlMeansDenoising(
                    flat, None, h, template_window, search_window
                )
                return denoised[:, :, np.newaxis] if image.ndim == 3 else denoised
            elif image.ndim == 3 and image.shape[2] == 3:
                h_color = float(params.get("h_color", h))
                return cv2.fastNlMeansDenoisingColored(
                    image, None, h, h_color, template_window, search_window
                )
            elif image.ndim == 3 and image.shape[2] == 4:
                # Process RGB channels, retain alpha
                rgb = image[:, :, :3]
                alpha = image[:, :, 3:4]
                denoised_rgb = cv2.fastNlMeansDenoisingColored(
                    rgb, None, h, h, template_window, search_window
                )
                return np.concatenate([denoised_rgb, alpha], axis=2)

        raise RuntimeError(f"Unhandled denoise execution path for method '{method}'")
