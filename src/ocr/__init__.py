"""OCR engine module supporting text detection, recognition, and reading order."""

from src.ocr.base import BaseOCREnginePrimitive, sort_tokens_reading_order
from src.ocr.factory import MockOCREngine, OCR_REGISTRY, get_ocr_engine
from src.ocr.rapid_ocr import RapidOCREngine

__all__ = [
    "BaseOCREnginePrimitive",
    "RapidOCREngine",
    "MockOCREngine",
    "get_ocr_engine",
    "sort_tokens_reading_order",
    "OCR_REGISTRY",
]
