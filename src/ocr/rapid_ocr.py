"""RapidOCR engine implementation using PP-OCRv4 ONNX Runtime backend."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.core.schemas import BoundingBox, OCRToken
from src.ocr.base import BaseOCREnginePrimitive


class RapidOCREngine(BaseOCREnginePrimitive):
    """RapidOCR engine baseline utilizing PP-OCRv4 models via ONNX Runtime.

    Engineering properties:
    - 100% local, offline inference via ONNX Runtime (zero external APIs).
    - Unified text detection (DBNet) and text recognition (CRNN) pipeline.
    - Preserves oriented 4-point quadrilateral boxes in metadata alongside
      axis-aligned BoundingBox representations.
    - Deterministic execution on CPU.
    """

    def __init__(
        self,
        min_confidence: float = 0.0,
        line_tolerance_factor: float = 0.5,
        text_score: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(line_tolerance_factor=line_tolerance_factor)
        self.min_confidence = float(min_confidence)
        self.text_score = float(text_score)

        # Lazy import of RapidOCR to allow importing module even if package was absent
        try:
            from rapidocr_onnxruntime import RapidOCR
            self._engine = RapidOCR(text_score=self.text_score, **kwargs)
        except ImportError as exc:
            raise ImportError(
                "rapidocr-onnxruntime is not installed. Please install it with: "
                "pip install rapidocr-onnxruntime"
            ) from exc

    def _recognize_impl(
        self, image: np.ndarray, document_id: str
    ) -> Tuple[List[OCRToken], Dict[str, Any]]:
        # RapidOCR expects 3-channel image (H, W, 3)
        if image.ndim == 2:
            img_input = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 1:
            img_input = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
        else:
            # OpenCV color convention: Dataset adapter provides RGB.
            # Convert RGB -> BGR for OpenCV-based RapidOCR preprocessor
            img_input = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        t0 = time.perf_counter()
        raw_results, elapse = self._engine(img_input)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        tokens: List[OCRToken] = []
        raw_quads: List[List[List[float]]] = []

        if raw_results is not None:
            for item in raw_results:
                quad_pts, text_val, conf_val = item
                text_clean = str(text_val).strip()
                conf_float = float(conf_val) if conf_val is not None else 0.0
                conf_float = max(0.0, min(1.0, conf_float))

                if conf_float < self.min_confidence:
                    continue

                # Coordinate extraction from 4-point polygon
                pts = [[float(pt[0]), float(pt[1])] for pt in quad_pts]
                raw_quads.append(pts)

                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]

                x_min, x_max = float(min(xs)), float(max(xs))
                y_min, y_max = float(min(ys)), float(max(ys))

                # Ensure non-zero dimension invariant of BoundingBox
                if x_max <= x_min:
                    x_max = x_min + 1.0
                if y_max <= y_min:
                    y_max = y_min + 1.0

                bbox = BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
                tokens.append(OCRToken(text=text_clean, bbox=bbox, confidence=conf_float))

        meta = {
            "processing_time_ms": elapsed_ms,
            "model_name": "RapidOCR",
            "model_version": "1.4.4 (PP-OCRv4)",
            "metadata": {
                "raw_detections_count": len(raw_results) if raw_results is not None else 0,
                "extracted_tokens_count": len(tokens),
                "elapse_breakdown": elapse,
                "raw_quadrilaterals": raw_quads,
            },
        }

        return tokens, meta
