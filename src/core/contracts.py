"""Abstract module contracts and interfaces for OCR/KIE research pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np

from src.core.schemas import (
    DegradationSpec,
    ExperimentResult,
    KIEGroundTruth,
    KIEResult,
    OCRGroundTruth,
    OCRResult,
    PreprocessingSpec,
)


class BaseDegradation(ABC):
    """Abstract contract for document image degradation."""

    @abstractmethod
    def apply(
        self, image: np.ndarray, spec: DegradationSpec
    ) -> Tuple[np.ndarray, DegradationSpec]:
        """Apply synthetic degradation to an image according to specification.

        Args:
            image: Input image as numpy ndarray (H, W, C) or (H, W).
            spec: Degradation specification including severity (0-4), parameters, and seed.

        Returns:
            Tuple of (degraded_image, applied_spec).
        """
        pass


class BasePreprocessor(ABC):
    """Abstract contract for document image enhancement/preprocessing."""

    @abstractmethod
    def process(
        self, image: np.ndarray, spec: PreprocessingSpec
    ) -> Tuple[np.ndarray, PreprocessingSpec]:
        """Apply enhancement/preprocessing methods to an image prior to OCR.

        Args:
            image: Degraded or raw input image as numpy ndarray.
            spec: Preprocessing specification including methods and parameters.

        Returns:
            Tuple of (preprocessed_image, applied_spec).
        """
        pass


class BaseOCREngine(ABC):
    """Abstract contract for OCR text recognition engine."""

    @abstractmethod
    def recognize(
        self, image: np.ndarray, document_id: str
    ) -> OCRResult:
        """Extract text and spatial tokens from an image.

        Args:
            image: Input image as numpy ndarray.
            document_id: Identifier of the document being recognized.

        Returns:
            Standardized OCRResult containing full text, tokens, and metadata.
        """
        pass


class BaseKIEEngine(ABC):
    """Abstract contract for Key Information Extraction (KIE) engine."""

    @abstractmethod
    def extract(
        self, ocr_result: OCRResult, image: Optional[np.ndarray] = None
    ) -> KIEResult:
        """Extract structured fields from OCR tokens and optional image visual cues.

        Args:
            ocr_result: Standardized OCR tokens and full text.
            image: Optional original or preprocessed image (for multimodal / Layout models).

        Returns:
            Standardized KIEResult containing extracted field mappings and confidences.
        """
        pass


class BaseEvaluator(ABC):
    """Abstract contract for OCR and KIE metric evaluation."""

    @abstractmethod
    def evaluate_ocr(
        self, prediction: OCRResult, ground_truth: OCRGroundTruth
    ) -> Dict[str, Any]:
        """Compute OCR quality metrics (e.g. CER, WER, ExactMatch).

        Args:
            prediction: Recognized OCR result.
            ground_truth: Reference text ground truth.

        Returns:
            Dictionary mapping metric names to computed numerical values.
        """
        pass

    @abstractmethod
    def evaluate_kie(
        self, prediction: KIEResult, ground_truth: KIEGroundTruth
    ) -> Dict[str, Any]:
        """Compute KIE extraction metrics (e.g. Field F1, Precision, Recall, Doc-EM).

        Args:
            prediction: Extracted KIE fields.
            ground_truth: Reference ground truth fields.

        Returns:
            Dictionary mapping metric names to computed numerical values.
        """
        pass
