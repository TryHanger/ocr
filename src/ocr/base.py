"""Base OCR Engine contract with strict GT isolation, input validation, and reading order."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.core.contracts import BaseOCREngine
from src.core.schemas import BoundingBox, OCRResult, OCRToken


def sort_tokens_reading_order(
    tokens: List[OCRToken],
    line_tolerance_factor: float = 0.5,
) -> Tuple[List[OCRToken], str]:
    """Sort tokens deterministically in top-to-bottom, left-to-right reading order.

    Algorithm:
    1. Compute characteristic line height H_line as median token height.
    2. Group tokens into horizontal lines using vertical center proximity:
       |y_center_i - y_center_line| <= line_tolerance_factor * H_line.
    3. Sort lines vertically by line top (y_min).
    4. Within each line, sort tokens horizontally by x_min (tie-break on y_min and original index).
    5. Construct full_text by joining tokens with spaces and lines with newlines.

    Note: line_tolerance_factor is a configurable engineering heuristic,
    not a universally optimal constant.
    """
    if not tokens:
        return [], ""

    indexed_tokens = list(enumerate(tokens))

    # Calculate token heights
    heights = [t.bbox.height for _, t in indexed_tokens if t.bbox.height > 0]
    h_line = float(np.median(heights)) if heights else 15.0
    h_line = max(1.0, h_line)
    y_tol = max(1.0, line_tolerance_factor * h_line)

    # Sort primarily by vertical center
    sorted_by_yc = sorted(
        indexed_tokens,
        key=lambda item: ((item[1].bbox.y_min + item[1].bbox.y_max) / 2.0, item[1].bbox.x_min)
    )

    # Cluster into lines
    lines: List[List[Tuple[int, OCRToken]]] = []
    current_line: List[Tuple[int, OCRToken]] = []
    current_line_yc: Optional[float] = None

    for idx, tok in sorted_by_yc:
        yc = (tok.bbox.y_min + tok.bbox.y_max) / 2.0
        if current_line_yc is None:
            current_line = [(idx, tok)]
            current_line_yc = yc
        elif abs(yc - current_line_yc) <= y_tol:
            current_line.append((idx, tok))
            # Update running average center for the line
            current_line_yc = float(np.mean([(t.bbox.y_min + t.bbox.y_max) / 2.0 for _, t in current_line]))
        else:
            lines.append(current_line)
            current_line = [(idx, tok)]
            current_line_yc = yc

    if current_line:
        lines.append(current_line)

    # Sort lines by minimum y_min
    lines.sort(key=lambda line: min(tok.bbox.y_min for _, tok in line))

    # Within each line, sort left-to-right (x_min, tie-break on y_min, original index)
    sorted_tokens: List[OCRToken] = []
    line_strings: List[str] = []

    for line in lines:
        line_sorted = sorted(line, key=lambda item: (item[1].bbox.x_min, item[1].bbox.y_min, item[0]))
        line_tokens = [tok for _, tok in line_sorted]
        sorted_tokens.extend(line_tokens)
        line_strings.append(" ".join(t.text.strip() for t in line_tokens if t.text.strip()))

    full_text = "\n".join(s for s in line_strings if s)
    return sorted_tokens, full_text


class BaseOCREnginePrimitive(BaseOCREngine):
    """Abstract base class for OCR engines with common validation and GT isolation.

    Strict GT Isolation Rule:
    The primary end-to-end OCR method `recognize(image, document_id)` accepts ONLY
    the image and document identifier. GT bounding boxes, GT transcriptions,
    and GT entity annotations are strictly prohibited from this path.
    """

    def __init__(self, line_tolerance_factor: float = 0.5) -> None:
        self.line_tolerance_factor = float(line_tolerance_factor)

    @abstractmethod
    def _recognize_impl(self, image: np.ndarray, document_id: str) -> Tuple[List[OCRToken], Dict[str, Any]]:
        """Engine-specific recognition logic.

        Args:
            image: Input image as numpy ndarray (H, W, C) or (H, W), uint8.
            document_id: Identifier of the document.

        Returns:
            Tuple of (unsorted_tokens, metadata_dict).
        """
        pass

    def recognize(self, image: np.ndarray, document_id: str) -> OCRResult:
        """Primary End-to-End OCR recognition API.

        Accepts ONLY the image and document ID. GT annotations are strictly forbidden.

        Args:
            image: Input image as numpy ndarray, uint8.
            document_id: Non-empty document identifier.

        Returns:
            OCRResult with tokens sorted in deterministic reading order.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Image must be np.ndarray, got {type(image).__name__}")
        if image.dtype != np.uint8:
            raise TypeError(f"Image dtype must be np.uint8, got {image.dtype}")
        if image.ndim not in (2, 3):
            raise ValueError(f"Image ndim must be 2 (grayscale) or 3 (color), got {image.ndim}")
        if image.shape[0] <= 0 or image.shape[1] <= 0:
            raise ValueError(f"Image dimensions must be positive, got shape {image.shape}")

        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError(f"document_id must be a non-empty string, got {document_id!r}")

        # Execute engine-specific extraction
        tokens, meta = self._recognize_impl(image, document_id)

        # Apply deterministic reading order
        ordered_tokens, full_text = sort_tokens_reading_order(
            tokens, line_tolerance_factor=self.line_tolerance_factor
        )

        return OCRResult(
            document_id=document_id,
            full_text=full_text,
            tokens=ordered_tokens,
            metadata=meta.get("metadata", {}),
            processing_time_ms=meta.get("processing_time_ms"),
            model_name=meta.get("model_name"),
            model_version=meta.get("model_version"),
        )

    def diagnostic_gt_region_recognition(
        self,
        image: np.ndarray,
        document_id: str,
        gt_boxes: List[BoundingBox],
    ) -> OCRResult:
        """Diagnostic GT-Region Recognition mode (Isolated from Primary Benchmark).

        Crops GT regions and runs text recognition individually to diagnose whether
        errors stem from detection or recognition. This method is strictly isolated
        and must never be invoked in primary benchmark runs.
        """
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
            raise TypeError("Invalid image array")

        tokens: List[OCRToken] = []
        h, w = image.shape[:2]

        for box in gt_boxes:
            # Safe integer bounding box clipping for crop
            x1 = max(0, min(w - 1, int(round(box.x_min))))
            y1 = max(0, min(h - 1, int(round(box.y_min))))
            x2 = max(x1 + 1, min(w, int(round(box.x_max))))
            y2 = max(y1 + 1, min(h, int(round(box.y_max))))

            crop = image[y1:y2, x1:x2]
            crop_tokens, _ = self._recognize_impl(crop, f"{document_id}_crop")
            # If recognized, associate with original GT box geometry
            if crop_tokens:
                combined_text = " ".join(t.text for t in crop_tokens)
                mean_conf = float(np.mean([t.confidence for t in crop_tokens if t.confidence is not None])) if any(t.confidence is not None for t in crop_tokens) else None
                tokens.append(OCRToken(text=combined_text, bbox=box, confidence=mean_conf))
            else:
                tokens.append(OCRToken(text="", bbox=box, confidence=0.0))

        ordered_tokens, full_text = sort_tokens_reading_order(
            tokens, line_tolerance_factor=self.line_tolerance_factor
        )

        return OCRResult(
            document_id=document_id,
            full_text=full_text,
            tokens=ordered_tokens,
            metadata={"mode": "diagnostic_gt_region"},
            model_name="Diagnostic_GT_Region",
        )
