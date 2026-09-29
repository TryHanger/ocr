"""Document Quality Schemas."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class QualityResultDTO(BaseModel):
    """Quality diagnostic breakdown and score for a document image."""

    document_id: str
    blur_score: float  # Laplacian variance
    contrast_score: float  # RMS contrast
    noise_score: float  # High-frequency noise estimate
    rotation_angle: float  # Deskew angle in degrees
    resolution_dpi: float
    quality_score: float = Field(ge=0.0, le=1.0)  # Aggregate score
    profile_summary: str
    metrics: Dict[str, Any] = Field(default_factory=dict)
    recommendations: Dict[str, bool] = Field(default_factory=dict)
