"""Base preprocessing primitive contract and validation."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Tuple
import numpy as np

from src.core.contracts import BasePreprocessor
from src.core.schemas import PreprocessingSpec


class BasePreprocessingPrimitive(BasePreprocessor):
    """Abstract base class for document image preprocessing primitives.

    Guarantees:
    1. Input array is never mutated in-place (defensive copy provided).
    2. Disabled spec (spec.enabled == False) returns an exact copy of the input.
    3. Input validation enforces np.ndarray, np.uint8 dtype, 2D/3D shapes.
    4. Deterministic execution for reproducible document enhancement.
    5. Output conventions (dtype, channels, shape) are explicitly validated.
    """

    @abstractmethod
    def _process(self, image: np.ndarray, spec: PreprocessingSpec) -> np.ndarray:
        """Internal processing implementation.

        Subclasses must implement this method. The input image is already a defensive
        copy and can be safely modified or transformed.

        Args:
            image: Defensive copy of input image (uint8, 2D or 3D).
            spec: Validated PreprocessingSpec with enabled=True.

        Returns:
            Processed image as np.ndarray (uint8). Channel count and shape may change
            intentionally (e.g. Grayscale or Binarization returning 2D uint8).
        """
        pass

    def process(
        self, image: np.ndarray, spec: PreprocessingSpec
    ) -> Tuple[np.ndarray, PreprocessingSpec]:
        """Apply preprocessing operation to the given image.

        Args:
            image: Input image as numpy ndarray (H, W, C) or (H, W), dtype uint8.
            spec: Preprocessing specification.

        Returns:
            Tuple of (processed_image, applied_spec).
        """
        # 1. Input validation
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Image must be np.ndarray, got {type(image).__name__}")
        if image.dtype != np.uint8:
            raise TypeError(f"Image dtype must be np.uint8, got {image.dtype}")
        if image.ndim not in (2, 3):
            raise ValueError(f"Image ndim must be 2 (grayscale) or 3 (color), got {image.ndim}")
        if image.shape[0] <= 0 or image.shape[1] <= 0:
            raise ValueError(f"Image dimensions must be positive, got shape {image.shape}")
        if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
            raise ValueError(f"Image channels must be 1, 3, or 4, got {image.shape[2]}")

        if not isinstance(spec, PreprocessingSpec):
            raise TypeError(f"spec must be PreprocessingSpec, got {type(spec).__name__}")

        # 2. Disabled pass-through
        if not spec.enabled:
            return image.copy(), spec

        # 3. Apply transformation on a defensive copy
        processed = self._process(image.copy(), spec)

        # 4. Output contract validation
        if not isinstance(processed, np.ndarray):
            raise TypeError(f"Processed output must be np.ndarray, got {type(processed).__name__}")
        if processed.dtype != np.uint8:
            raise TypeError(f"Processed output dtype must be np.uint8, got {processed.dtype}")
        if processed.ndim not in (2, 3):
            raise ValueError(f"Processed output ndim must be 2 or 3, got {processed.ndim}")
        if processed.shape[0] <= 0 or processed.shape[1] <= 0:
            raise ValueError(f"Processed output dimensions must be positive, got shape {processed.shape}")
        if processed.ndim == 3 and processed.shape[2] not in (1, 3, 4):
            raise ValueError(f"Processed output channels must be 1, 3, or 4, got {processed.shape[2]}")

        return processed, spec
