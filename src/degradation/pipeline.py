"""Sequential degradation pipeline and registry factory."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Type
import numpy as np

from src.core.contracts import BaseDegradation
from src.core.schemas import DegradationSpec
from src.degradation.base import BaseDegradationPrimitive
from src.degradation.downsampling import DownsamplingDegradation
from src.degradation.gaussian_blur import GaussianBlurDegradation
from src.degradation.gaussian_noise import GaussianNoiseDegradation
from src.degradation.jpeg_compression import JPEGCompressionDegradation
from src.degradation.motion_blur import MotionBlurDegradation
from src.degradation.perspective import PerspectiveDegradation
from src.degradation.rotation import RotationDegradation
from src.degradation.shadow import ShadowDegradation


DEGRADATION_REGISTRY: Dict[str, Type[BaseDegradationPrimitive]] = {
    "gaussian_blur": GaussianBlurDegradation,
    "motion_blur": MotionBlurDegradation,
    "gaussian_noise": GaussianNoiseDegradation,
    "jpeg_compression": JPEGCompressionDegradation,
    "downsampling": DownsamplingDegradation,
    "rotation": RotationDegradation,
    "perspective": PerspectiveDegradation,
    "shadow": ShadowDegradation,
}


def get_degradation(type_name: str) -> BaseDegradationPrimitive:
    """Retrieve an instantiated degradation primitive by registered type name.

    Args:
        type_name: Name of the degradation primitive.

    Returns:
        Instance of BaseDegradationPrimitive.

    Raises:
        KeyError: If type_name is not registered.
    """
    normalized = type_name.strip().lower()
    if normalized not in DEGRADATION_REGISTRY:
        supported = ", ".join(sorted(DEGRADATION_REGISTRY.keys()))
        raise KeyError(f"Unknown degradation type '{type_name}'. Supported: {supported}")
    return DEGRADATION_REGISTRY[normalized]()


class DegradationPipeline:
    """Composable pipeline executing a sequence of degradation primitives in deterministic order.

    Guarantees:
    - Input image is never mutated in-place.
    - Transformations are applied strictly sequentially.
    - Resulting image matches input dimensions and uint8 dtype.
    """

    def __init__(
        self,
        steps: Optional[Sequence[Tuple[BaseDegradation, DegradationSpec]]] = None,
    ) -> None:
        self.steps: List[Tuple[BaseDegradation, DegradationSpec]] = list(steps or [])

    def add_step(self, degradation: BaseDegradation, spec: DegradationSpec) -> DegradationPipeline:
        """Append a degradation step to the pipeline."""
        if not isinstance(degradation, BaseDegradation):
            raise TypeError(f"Expected BaseDegradation, got {type(degradation).__name__}")
        if not isinstance(spec, DegradationSpec):
            raise TypeError(f"Expected DegradationSpec, got {type(spec).__name__}")
        self.steps.append((degradation, spec))
        return self

    def apply(
        self,
        image: np.ndarray,
        specs: Optional[Sequence[DegradationSpec]] = None,
    ) -> Tuple[np.ndarray, List[DegradationSpec]]:
        """Apply pipeline degradation sequence to an image.

        Args:
            image: Input image (H, W, C) or (H, W), uint8.
            specs: Optional sequence of specs. If provided, replaces/overrides
                   steps dynamically using registered degradation primitives.

        Returns:
            Tuple of (final_degraded_image, list_of_applied_specs).
        """
        current_img = image.copy()
        applied_specs: List[DegradationSpec] = []

        if specs is not None:
            pipeline_steps = [(get_degradation(s.type), s) for s in specs]
        else:
            pipeline_steps = self.steps

        for deg_op, spec in pipeline_steps:
            current_img, applied_spec = deg_op.apply(current_img, spec)
            applied_specs.append(applied_spec)

        return current_img, applied_specs
