"""Preprocessing Pipeline and Factory."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type
import numpy as np

from src.core.contracts import BasePreprocessor
from src.core.schemas import PreprocessingSpec
from src.preprocessing.base import BasePreprocessingPrimitive
from src.preprocessing.binarization import BinarizationPreprocessor
from src.preprocessing.clahe import CLAHEPreprocessor
from src.preprocessing.denoise import DenoisePreprocessor
from src.preprocessing.deskew import DeskewPreprocessor
from src.preprocessing.grayscale import GrayscalePreprocessor


PREPROCESSOR_REGISTRY: Dict[str, Type[BasePreprocessingPrimitive]] = {
    "grayscale": GrayscalePreprocessor,
    "denoise": DenoisePreprocessor,
    "clahe": CLAHEPreprocessor,
    "binarization": BinarizationPreprocessor,
    "deskew": DeskewPreprocessor,
}


def get_preprocessor(name: str, params: Optional[Dict[str, Any]] = None) -> BasePreprocessingPrimitive:
    """Instantiate a preprocessing primitive by name.

    Args:
        name: Name of the preprocessing primitive (e.g. 'grayscale', 'denoise').
        params: Optional dictionary of default parameters for the primitive.

    Returns:
        Instance of BasePreprocessingPrimitive.

    Raises:
        KeyError: If preprocessor name is not registered.
    """
    key = name.strip().lower()
    if key not in PREPROCESSOR_REGISTRY:
        raise KeyError(
            f"Unknown preprocessor '{name}'. Registered preprocessors: {list(PREPROCESSOR_REGISTRY.keys())}"
        )
    return PREPROCESSOR_REGISTRY[key]()


class PreprocessingPipeline(BasePreprocessor):
    """Deterministic composition pipeline for preprocessing primitives.

    Applies preprocessors sequentially in the exact order requested in spec.methods.
    Guarantees:
    - Input immutability.
    - Explicit ordering.
    - Deterministic execution.
    - Rejection of unknown preprocessing methods.
    """

    def __init__(self, steps: Optional[List[str]] = None):
        """Initialize pipeline with optional default step sequence.

        Args:
            steps: Optional list of method names. If omitted, uses spec.methods.
        """
        self.steps = steps

    def process(
        self, image: np.ndarray, spec: PreprocessingSpec
    ) -> Tuple[np.ndarray, PreprocessingSpec]:
        """Execute preprocessing pipeline sequentially.

        Args:
            image: Input image (2D or 3D uint8 array).
            spec: Preprocessing specification containing methods and parameters.

        Returns:
            Tuple of (preprocessed_image, applied_spec).
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Image must be np.ndarray, got {type(image).__name__}")
        if image.dtype != np.uint8:
            raise TypeError(f"Image dtype must be np.uint8, got {image.dtype}")
        if not isinstance(spec, PreprocessingSpec):
            raise TypeError(f"spec must be PreprocessingSpec, got {type(spec).__name__}")

        if not spec.enabled:
            return image.copy(), spec

        # Determine step order: spec.methods takes precedence if provided, else self.steps
        method_names = spec.methods if spec.methods else (self.steps or [])
        if not method_names:
            return image.copy(), spec

        current_img = image.copy()
        for method_name in method_names:
            key = method_name.strip().lower()
            if key not in PREPROCESSOR_REGISTRY:
                raise KeyError(
                    f"Unknown preprocessing step '{method_name}'. Registered: {list(PREPROCESSOR_REGISTRY.keys())}"
                )

            primitive = get_preprocessor(key)
            # Create a focused spec for this primitive preserving parameters
            step_params = spec.parameters.get(key, spec.parameters)
            step_spec = PreprocessingSpec(
                enabled=True,
                methods=[key],
                parameters=step_params if isinstance(step_params, dict) else {},
            )
            current_img, _ = primitive.process(current_img, step_spec)

        return current_img, spec
