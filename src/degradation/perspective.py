"""D7 — Perspective Distortion degradation primitive."""

from __future__ import annotations

import cv2
import numpy as np

from src.core.schemas import DegradationSpec
from src.degradation.base import BaseDegradationPrimitive


class PerspectiveDegradation(BaseDegradationPrimitive):
    """Four-point perspective distortion degradation.

    Applies deterministic corner displacement based on distortion scale and seed.
    Output dimensions are strictly preserved.
    """

    def _apply(self, image: np.ndarray, spec: DegradationSpec) -> np.ndarray:
        params = spec.parameters
        scale = float(params.get("distortion_scale", 0.1))
        if scale < 0.0 or scale >= 0.5:
            raise ValueError(f"distortion_scale must be in [0.0, 0.5), got {scale}")

        if scale == 0.0:
            return image

        default_val = (255, 255, 255) if image.ndim == 3 else 255
        border_value = params.get("border_value", default_val)

        h, w = image.shape[:2]
        src_pts = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])

        rng = np.random.default_rng(spec.seed)
        max_dx = w * scale
        max_dy = h * scale

        # Displace corners inward deterministically
        dxs = rng.uniform(0.0, max_dx, size=4)
        dys = rng.uniform(0.0, max_dy, size=4)

        dst_pts = np.float32(
            [
                [dxs[0], dys[0]],
                [w - 1 - dxs[1], dys[1]],
                [w - 1 - dxs[2], h - 1 - dys[2]],
                [dxs[3], h - 1 - dys[3]],
            ]
        )

        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )

        if image.ndim == 3 and warped.ndim == 2:
            warped = warped[:, :, np.newaxis]

        return warped
