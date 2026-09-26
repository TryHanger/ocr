"""D2 — Motion Blur degradation primitive."""

from __future__ import annotations

import math
import cv2
import numpy as np

from src.core.schemas import DegradationSpec
from src.degradation.base import BaseDegradationPrimitive


class MotionBlurDegradation(BaseDegradationPrimitive):
    """Linear directional motion blur degradation."""

    def _apply(self, image: np.ndarray, spec: DegradationSpec) -> np.ndarray:
        params = spec.parameters
        kernel_length = params.get("kernel_length", 5)
        if not isinstance(kernel_length, int) or isinstance(kernel_length, bool):
            raise TypeError(f"kernel_length must be int, got {type(kernel_length).__name__}")
        if kernel_length < 3:
            raise ValueError(f"kernel_length must be at least 3, got {kernel_length}")

        angle = float(params.get("angle", 0.0))

        # Build directional line kernel of size (L, L)
        L = kernel_length
        kernel = np.zeros((L, L), dtype=np.float32)
        cx, cy = (L - 1) / 2.0, (L - 1) / 2.0
        rad = math.radians(angle)

        half_len = (L - 1) / 2.0
        x1 = int(round(cx - half_len * math.cos(rad)))
        y1 = int(round(cy - half_len * math.sin(rad)))
        x2 = int(round(cx + half_len * math.cos(rad)))
        y2 = int(round(cy + half_len * math.sin(rad)))

        cv2.line(kernel, (x1, y1), (x2, y2), 1.0, thickness=1)

        k_sum = kernel.sum()
        if k_sum > 0:
            kernel /= k_sum
        else:
            kernel[int(cy), int(cx)] = 1.0

        blurred = cv2.filter2D(image, -1, kernel)
        return blurred
