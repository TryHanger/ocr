"""Abstract module contracts and interfaces for OCR/KIE research pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np

from src.core.schemas import (
    DegradationSpec,
    DocumentMetadata,
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
        self, ocr_result: OCRResult, document_id: str
    ) -> KIEResult:
        """Extract structured fields from OCR tokens.

        Args:
            ocr_result: Standardized OCR tokens and full text.
            document_id: Identifier of the document being processed.

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


class BaseDatasetAdapter(ABC):
    """Abstract contract for document dataset access."""

    @abstractmethod
    def list_document_ids(self, split: str) -> list[str]:
        """List document IDs for a given split in deterministic order.

        Args:
            split: Dataset split identifier ('train', 'test', etc.).

        Returns:
            List of unique document IDs without file extensions.
        """
        pass

    @abstractmethod
    def get_metadata(self, document_id: str) -> DocumentMetadata:
        """Get standardized metadata for a document.

        Args:
            document_id: Unique document identifier.

        Returns:
            Standardized DocumentMetadata instance.
        """
        pass

    @abstractmethod
    def get_image(self, document_id: str) -> np.ndarray:
        """Load document image in canonical representation (RGB uint8 ndarray).

        Args:
            document_id: Unique document identifier.

        Returns:
            Image as numpy ndarray of shape (H, W, 3) and dtype uint8.
        """
        pass

    @abstractmethod
    def get_ocr_ground_truth(self, document_id: str) -> OCRGroundTruth:
        """Load immutable OCR ground truth for a document.

        Args:
            document_id: Unique document identifier.

        Returns:
            Standardized OCRGroundTruth instance with tokens and bounding boxes.
        """
        pass

    @abstractmethod
    def get_kie_ground_truth(self, document_id: str) -> KIEGroundTruth:
        """Load immutable KIE ground truth for a document.

        Args:
            document_id: Unique document identifier.

        Returns:
            Standardized KIEGroundTruth instance with raw entity field values.
        """
        pass

