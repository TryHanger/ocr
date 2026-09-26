"""D6 — Geometric Rotation degradation primitive."""

from __future__ import annotations

import cv2
import numpy as np

from src.core.schemas import DegradationSpec
from src.degradation.base import BaseDegradationPrimitive


class RotationDegradation(BaseDegradationPrimitive):
    """Geometric rotation degradation around image center.

    Border handling:
    - Supported modes: 'constant' (default) and 'replicate'.
    - For document receipts, 'constant' fills exposed corner areas with white
      paper background (255, 255, 255) by default to prevent artificial black borders.
    - Output dimensions are strictly identical to input dimensions.
    """

    def _apply(self, image: np.ndarray, spec: DegradationSpec) -> np.ndarray:
        params = spec.parameters
        angle = float(params.get("angle", 0.0))
        if angle == 0.0:
            return image

        border_mode_str = str(params.get("border_mode", "constant")).lower()
        if border_mode_str == "replicate":
            border_mode = cv2.BORDER_REPLICATE
            border_value = 0
        elif border_mode_str == "constant":
            border_mode = cv2.BORDER_CONSTANT
            default_val = (255, 255, 255) if image.ndim == 3 else 255
            border_value = params.get("border_value", default_val)
        else:
            raise ValueError(
                f"Unsupported border_mode '{border_mode_str}', must be 'constant' or 'replicate'"
            )

        h, w = image.shape[:2]
        center = ((w - 1) / 2.0, (h - 1) / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)

        rotated = cv2.warpAffine(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=border_mode,
            borderValue=border_value,
        )

        if image.ndim == 3 and rotated.ndim == 2:
            rotated = rotated[:, :, np.newaxis]

        return rotated
