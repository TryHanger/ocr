"""ProcessingStep Schemas."""

from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel


class ProcessingStepDTO(BaseModel):
    id: str
    step_order: int
    operation: str
    parameters: Optional[Dict[str, Any]] = None
    duration_ms: float
    status: str
    created_at: datetime
