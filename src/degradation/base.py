"""Base degradation primitive contract with common validation and severity 0 handling."""

from __future__ import annotations

from abc import abstractmethod
from typing import Tuple
import numpy as np

from src.core.contracts import BaseDegradation
from src.core.schemas import DegradationSpec


class BaseDegradationPrimitive(BaseDegradation):
    """Abstract base class for individual degradation primitives.

    Guarantees:
    1. Input array is never mutated in-place.
    2. Severity 0 returns an exact independent copy of the input image.
    3. Output array strictly matches input shape and uint8 dtype.
    4. Deterministic execution when random operations are seeded.
    """

    @abstractmethod
    def _apply(self, image: np.ndarray, spec: DegradationSpec) -> np.ndarray:
        """Internal degradation implementation for severity > 0.

        Subclasses must implement this method. The input image is already a copy
        and can be modified safely.

        Args:
            image: Input image copy as numpy ndarray (H, W, C) or (H, W), uint8.
            spec: Validated DegradationSpec with severity in [1, 4].

        Returns:
            Degraded image as numpy ndarray with exact same shape and dtype uint8.
        """
        pass

    def apply(
        self, image: np.ndarray, spec: DegradationSpec
    ) -> Tuple[np.ndarray, DegradationSpec]:
        """Apply degradation to an image according to specification.

        Args:
            image: Input image as numpy ndarray (H, W, C) or (H, W), dtype uint8.
            spec: Degradation specification.

        Returns:
            Tuple of (degraded_image, applied_spec).
        """
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

        if not isinstance(spec, DegradationSpec):
            raise TypeError(f"spec must be DegradationSpec, got {type(spec).__name__}")

        # Severity 0: Control reference / identity (bitwise identical, no modifications)
        if spec.severity == 0:
            return image.copy(), spec

        # Apply primitive on a fresh copy of the image
        degraded = self._apply(image.copy(), spec)

        # Enforce output contract
        if not isinstance(degraded, np.ndarray):
            raise TypeError(f"Degraded output must be np.ndarray, got {type(degraded).__name__}")
        if degraded.dtype != np.uint8:
            raise TypeError(f"Degraded output dtype must be np.uint8, got {degraded.dtype}")
        if degraded.shape != image.shape:
            raise ValueError(
                f"Degraded output shape {degraded.shape} must match input shape {image.shape}"
            )

        return degraded, spec
