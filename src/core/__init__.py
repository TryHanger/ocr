"""Core schemas, DTOs, contracts, and utilities for OCR/KIE robustness research."""

from src.core.contracts import (
    BaseDegradation,
    BaseEvaluator,
    BaseKIEEngine,
    BaseOCREngine,
    BasePreprocessor,
)
from src.core.schemas import (
    BoundingBox,
    DegradationSpec,
    DocumentMetadata,
    ExperimentResult,
    KIEGroundTruth,
    KIEResult,
    OCRGroundTruth,
    OCRResult,
    OCRToken,
    PreprocessingSpec,
)
from src.core.seed import set_seed

__all__ = [
    "BoundingBox",
    "OCRToken",
    "OCRResult",
    "KIEResult",
    "DocumentMetadata",
    "DegradationSpec",
    "PreprocessingSpec",
    "ExperimentResult",
    "OCRGroundTruth",
    "KIEGroundTruth",
    "BaseDegradation",
    "BasePreprocessor",
    "BaseOCREngine",
    "BaseKIEEngine",
    "BaseEvaluator",
    "set_seed",
]
