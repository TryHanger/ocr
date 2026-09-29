"""Normalized OCR Schemas."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class NormalizedTokenDTO(BaseModel):
    """Token representation with coordinates normalized to 0.0 - 1.0 range."""

    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    # [x_min, y_min, x_max, y_max] in normalized 0.0..1.0 relative coordinates
    bbox: List[float] = Field(min_length=4, max_length=4)
    page_number: int = 1


class NormalizedOCRResult(BaseModel):
    """Standardized OCR output format independent of underlying ML engine."""

    document_id: str
    full_text: str
    tokens: List[NormalizedTokenDTO]
    processing_time_ms: float
    model_name: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
