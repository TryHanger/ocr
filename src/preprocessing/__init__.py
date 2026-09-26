"""Preprocessing module exports."""

from src.preprocessing.base import BasePreprocessingPrimitive
from src.preprocessing.baseline import (
    BaselineComparisonResult,
    BaselineRunOutput,
    run_b0_b1_b2_comparison,
)
from src.preprocessing.binarization import BinarizationPreprocessor
from src.preprocessing.clahe import CLAHEPreprocessor
from src.preprocessing.denoise import DenoisePreprocessor
from src.preprocessing.deskew import DeskewPreprocessor
from src.preprocessing.grayscale import GrayscalePreprocessor
from src.preprocessing.pipeline import (
    PREPROCESSOR_REGISTRY,
    PreprocessingPipeline,
    get_preprocessor,
)

__all__ = [
    "BasePreprocessingPrimitive",
    "GrayscalePreprocessor",
    "DenoisePreprocessor",
    "CLAHEPreprocessor",
    "BinarizationPreprocessor",
    "DeskewPreprocessor",
    "PreprocessingPipeline",
    "PREPROCESSOR_REGISTRY",
    "get_preprocessor",
    "BaselineComparisonResult",
    "BaselineRunOutput",
    "run_b0_b1_b2_comparison",
]
