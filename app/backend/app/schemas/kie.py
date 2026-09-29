"""Normalized KIE Schemas."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class NormalizedFieldDTO(BaseModel):
    """Normalized structured field extracted by KIE engine."""

    field_name: str
    value: str  # Active value (corrected_value if present, else original_value)
    original_value: Optional[str] = None
    corrected_value: Optional[str] = None
    is_corrected: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
    # [x_min, y_min, x_max, y_max] in normalized 0.0..1.0 relative coordinates
    bbox: Optional[List[float]] = None
    source_token_indices: List[int] = Field(default_factory=list)
    validation_status: str = "VALID"
    rule_applied: Optional[str] = None


class NormalizedKIEResult(BaseModel):
    """Standardized KIE result independent of underlying ML engine."""

    document_id: str
    fields: Dict[str, NormalizedFieldDTO]
    processing_time_ms: float
    model_name: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
