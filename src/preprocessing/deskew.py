"""P5 — Document Deskew Preprocessing Primitive."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import cv2
import numpy as np

from src.core.schemas import PreprocessingSpec
from src.preprocessing.base import BasePreprocessingPrimitive


class DeskewPreprocessor(BasePreprocessingPrimitive):
    """Deterministic document deskew primitive.

    Estimates dominant text orientation from image features alone without Ground Truth:
    1. Converts image to grayscale (if color).
    2. Computes edge map using Canny edge detection.
    3. Detects linear segments via Probabilistic Hough Line Transform (HoughLinesP).
    4. Calculates line orientation angles and normalizes them to [-45, 45] degrees.
    5. Computes median angle across detected segments.
    6. Clamps correction to [-max_angle, max_angle] (default +/- 15 degrees).
    7. Rotates image around center with white border padding to preserve document appearance.

    Conventions:
    - Input: 2D (H, W) or 3D (H, W, C) uint8.
    - Output: Same shape and channels as input, dtype uint8.
    - Border fill: White (255 for grayscale, [255, 255, 255] for RGB).
    - GT Isolation: Purely image-derived; Ground Truth annotations are never accessed.
    - Heuristic Scope: Documented as an engineering heuristic suitable for receipts
      and document scans, not a universally optimal multi-column layout parser.
    """

    def _get_params(self, spec: PreprocessingSpec) -> Dict[str, Any]:
        params = spec.parameters.get("deskew", spec.parameters)
        return dict(params) if isinstance(params, dict) else {}

    def estimate_skew_angle(
        self, image: np.ndarray, max_angle: float = 15.0, min_lines: int = 3
    ) -> float:
        """Estimate dominant skew angle in degrees from image features alone.

        Args:
            image: 2D or 3D uint8 image.
            max_angle: Maximum allowable angle in degrees.
            min_lines: Minimum detected line segments required for angle consensus.

        Returns:
            Estimated skew angle in degrees (positive = clockwise rotation needed to correct).
        """
        # Convert to 2D grayscale
        if image.ndim == 2:
            gray = image
        elif image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.shape[2] >= 3 else image[:, :, 0]
        else:
            return 0.0

        # Canny edge detection
        edges = cv2.Canny(gray, threshold1=50, threshold2=150, apertureSize=3)

        # Probabilistic Hough Line Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1.0,
            theta=np.pi / 180.0,
            threshold=40,
            minLineLength=30,
            maxLineGap=5,
        )

        if lines is None or len(lines) < min_lines:
            return 0.0

        angles: List[float] = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 == x2 and y1 == y2:
                continue
            deg = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            # Normalize to [-45, 45] range
            while deg > 45.0:
                deg -= 90.0
            while deg < -45.0:
                deg += 90.0
            angles.append(deg)

        if len(angles) < min_lines:
            return 0.0

        median_angle = float(np.median(angles))

        # Ignore if estimated angle exceeds maximum correction limit
        if abs(median_angle) > max_angle:
            return 0.0

        return median_angle

    def _process(self, image: np.ndarray, spec: PreprocessingSpec) -> np.ndarray:
        """Deskew image by estimating angle and rotating with white padding."""
        params = self._get_params(spec)
        max_angle = float(params.get("max_angle", 15.0))
        min_lines = int(params.get("min_lines", 3))

        if max_angle <= 0:
            raise ValueError(f"max_angle must be positive, got {max_angle}")

        angle = self.estimate_skew_angle(image, max_angle=max_angle, min_lines=min_lines)

        # If angle is negligible (< 0.1 deg), return copy
        if abs(angle) < 0.1:
            return image.copy()

        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)
        # Rotation matrix to correct skew
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        border_val = (
            (255, 255, 255) if image.ndim == 3 and image.shape[2] >= 3 else 255
        )

        deskewed = cv2.warpAffine(
            image,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_val,
        )

        return deskewed
