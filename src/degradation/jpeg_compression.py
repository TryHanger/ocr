"""D4 — JPEG Compression degradation primitive."""

from __future__ import annotations

import cv2
import numpy as np

from src.core.schemas import DegradationSpec
from src.degradation.base import BaseDegradationPrimitive


class JPEGCompressionDegradation(BaseDegradationPrimitive):
    """JPEG compression degradation using in-memory encode/decode."""

    def _apply(self, image: np.ndarray, spec: DegradationSpec) -> np.ndarray:
        params = spec.parameters
        quality = params.get("quality", 50)
        if not isinstance(quality, int) or isinstance(quality, bool):
            raise TypeError(f"quality must be int, got {type(quality).__name__}")
        if not (1 <= quality <= 100):
            raise ValueError(f"quality must be between 1 and 100, got {quality}")

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

        if image.ndim == 3 and image.shape[2] == 3:
            # OpenCV imencode expects BGR channel order
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            success, enc = cv2.imencode(".jpg", bgr, encode_param)
            if not success:
                raise RuntimeError("Failed to JPEG-encode image")
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            output = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
        elif image.ndim == 3 and image.shape[2] == 1:
            success, enc = cv2.imencode(".jpg", image, encode_param)
            if not success:
                raise RuntimeError("Failed to JPEG-encode image")
            dec = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
            output = dec[:, :, np.newaxis]
        else:
            # 2D Grayscale
            success, enc = cv2.imencode(".jpg", image, encode_param)
            if not success:
                raise RuntimeError("Failed to JPEG-encode image")
            output = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)

        return output
