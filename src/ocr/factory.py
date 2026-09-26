"""OCR engine factory and registry."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type
import numpy as np

from src.core.schemas import BoundingBox, OCRToken
from src.ocr.base import BaseOCREnginePrimitive
from src.ocr.rapid_ocr import RapidOCREngine


class MockOCREngine(BaseOCREnginePrimitive):
    """Deterministic Mock OCR engine for fast offline tests and contract validation."""

    def __init__(
        self,
        mock_tokens: Optional[List[OCRToken]] = None,
        line_tolerance_factor: float = 0.5,
    ) -> None:
        super().__init__(line_tolerance_factor=line_tolerance_factor)
        self.mock_tokens = list(mock_tokens or [])

    def set_mock_tokens(self, tokens: List[OCRToken]) -> None:
        self.mock_tokens = list(tokens)

    def _recognize_impl(
        self, image: np.ndarray, document_id: str
    ) -> Tuple[List[OCRToken], Dict[str, Any]]:
        meta = {
            "processing_time_ms": 1.0,
            "model_name": "MockOCR",
            "model_version": "1.0.0",
            "metadata": {"mock": True},
        }
        return list(self.mock_tokens), meta


OCR_REGISTRY: Dict[str, Type[BaseOCREnginePrimitive]] = {
    "rapidocr": RapidOCREngine,
    "rapid_ocr": RapidOCREngine,
    "mock": MockOCREngine,
    "mock_ocr": MockOCREngine,
}


def get_ocr_engine(
    engine_name: str,
    config: Optional[Dict[str, Any]] = None,
) -> BaseOCREnginePrimitive:
    """Retrieve an instantiated OCR engine by registered name.

    Args:
        engine_name: Registered engine identifier ('rapidocr', 'mock').
        config: Optional parameter dictionary passed to engine constructor.

    Returns:
        Instantiated BaseOCREnginePrimitive.
    """
    key = engine_name.strip().lower()
    if key not in OCR_REGISTRY:
        supported = ", ".join(sorted(OCR_REGISTRY.keys()))
        raise KeyError(f"Unknown OCR engine '{engine_name}'. Supported: {supported}")

    cls = OCR_REGISTRY[key]
    params = config or {}
    return cls(**params)
