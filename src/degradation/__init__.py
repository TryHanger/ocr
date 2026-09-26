"""Degradation engine module for synthetic document image distortions."""

from src.degradation.base import BaseDegradationPrimitive
from src.degradation.downsampling import DownsamplingDegradation
from src.degradation.gaussian_blur import GaussianBlurDegradation
from src.degradation.gaussian_noise import GaussianNoiseDegradation
from src.degradation.jpeg_compression import JPEGCompressionDegradation
from src.degradation.motion_blur import MotionBlurDegradation
from src.degradation.perspective import PerspectiveDegradation
from src.degradation.pipeline import (
    DEGRADATION_REGISTRY,
    DegradationPipeline,
    get_degradation,
)
from src.degradation.rotation import RotationDegradation
from src.degradation.shadow import ShadowDegradation

__all__ = [
    "BaseDegradationPrimitive",
    "GaussianBlurDegradation",
    "MotionBlurDegradation",
    "GaussianNoiseDegradation",
    "JPEGCompressionDegradation",
    "DownsamplingDegradation",
    "RotationDegradation",
    "PerspectiveDegradation",
    "ShadowDegradation",
    "DegradationPipeline",
    "get_degradation",
    "DEGRADATION_REGISTRY",
]
