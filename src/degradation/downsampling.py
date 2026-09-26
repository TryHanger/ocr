"""D5 — Downsampling / Low Resolution degradation primitive."""

from __future__ import annotations

import cv2
import numpy as np

from src.core.schemas import DegradationSpec
from src.degradation.base import BaseDegradationPrimitive


class DownsamplingDegradation(BaseDegradationPrimitive):
    """Downsampling degradation that reduces resolution and restores original shape."""

    def _apply(self, image: np.ndarray, spec: DegradationSpec) -> np.ndarray:
        params = spec.parameters
        if "scale_factor" in params:
            scale_factor = float(params["scale_factor"])
        elif "factor" in params:
            factor = float(params["factor"])
            if factor <= 0:
                raise ValueError(f"factor must be positive, got {factor}")
            scale_factor = 1.0 / factor
        else:
            scale_factor = 0.5

        if not (0.0 < scale_factor <= 1.0):
            raise ValueError(f"scale_factor must be in (0.0, 1.0], got {scale_factor}")

        if scale_factor == 1.0:
            return image

        orig_h, orig_w = image.shape[:2]
        low_w = max(1, int(round(orig_w * scale_factor)))
        low_h = max(1, int(round(orig_h * scale_factor)))

        # Downsample using AREA interpolation (best for decimation)
        downscaled = cv2.resize(image, (low_w, low_h), interpolation=cv2.INTER_AREA)

        # Restore back to original resolution using LINEAR interpolation
        restored = cv2.resize(downscaled, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # If 2D image became 3D or vice versa, ensure ndim matches
        if image.ndim == 3 and restored.ndim == 2:
            restored = restored[:, :, np.newaxis]

        return restored
