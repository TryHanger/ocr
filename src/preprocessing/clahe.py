"""P3 — CLAHE (Contrast Limited Adaptive Histogram Equalization) Primitive."""

from __future__ import annotations

from typing import Any, Dict, Tuple
import cv2
import numpy as np

from src.core.schemas import PreprocessingSpec
from src.preprocessing.base import BasePreprocessingPrimitive


class CLAHEPreprocessor(BasePreprocessingPrimitive):
    """Contrast Limited Adaptive Histogram Equalization primitive.

    Conventions:
    - Input: 2D (H, W) or 3D (H, W, C) uint8.
    - Output: Same shape and channels as input, dtype uint8.
    - Color-space handling:
      * Grayscale (2D or HxWx1): applied directly to intensity values.
      * Color (3-channel RGB): converted to CIE LAB space, CLAHE applied strictly
        to the Luminance (L) channel, then converted back to RGB to prevent color shifting.
      * RGBA (4-channel): RGB processed via LAB, alpha channel preserved unchanged.
    """

    def _get_params(self, spec: PreprocessingSpec) -> Dict[str, Any]:
        params = spec.parameters.get("clahe", spec.parameters)
        return dict(params) if isinstance(params, dict) else {}

    def _process(self, image: np.ndarray, spec: PreprocessingSpec) -> np.ndarray:
        """Apply CLAHE with configurable clip_limit and tile_grid_size."""
        params = self._get_params(spec)

        clip_limit = float(params.get("clip_limit", 2.0))
        if clip_limit <= 0:
            raise ValueError(f"clip_limit for CLAHE must be positive, got {clip_limit}")

        tile_size_raw = params.get("tile_grid_size", (8, 8))
        if isinstance(tile_size_raw, (list, tuple)) and len(tile_size_raw) == 2:
            tile_grid_size = (int(tile_size_raw[0]), int(tile_size_raw[1]))
        elif isinstance(tile_size_raw, int):
            tile_grid_size = (int(tile_size_raw), int(tile_size_raw))
        else:
            raise TypeError(f"tile_grid_size must be a tuple/list of 2 ints, got {tile_size_raw}")

        if tile_grid_size[0] <= 0 or tile_grid_size[1] <= 0:
            raise ValueError(f"tile_grid_size components must be positive, got {tile_grid_size}")

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

        if image.ndim == 2:
            return clahe.apply(image)

        if image.ndim == 3:
            channels = image.shape[2]
            if channels == 1:
                res = clahe.apply(image[:, :, 0])
                return res[:, :, np.newaxis]
            elif channels == 3:
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            elif channels == 4:
                rgb = image[:, :, :3]
                alpha = image[:, :, 3:4]
                lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                enhanced_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                return np.concatenate([enhanced_rgb, alpha], axis=2)

        raise RuntimeError(f"Unsupported image format for CLAHE with shape {image.shape}")
