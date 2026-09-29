"""Decision & Automation Explainability Schemas."""

from typing import Any, List, Optional, Union
from pydantic import BaseModel, Field


class DecisionReason(BaseModel):
    """Specific condition or evidence contributing to an automation decision."""

    code: str
    category: str  # "confidence" | "validation" | "quality" | "pipeline" | "none"
    message: str
    blocking: bool = True
    field: Optional[str] = None
    observed_value: Optional[Union[float, str]] = None
    threshold: Optional[Union[float, str]] = None


class DecisionExplanation(BaseModel):
    """Complete, structured, and factual explanation of an automation decision."""

    decision: str  # "automatic" | "manual_review" | "rejected" | "pending"
    primary_reason: Optional[str] = None
    reasons: List[DecisionReason] = Field(default_factory=list)
    positive_criteria: List[str] = Field(default_factory=list)
    automatic_eligible: bool
    summary: str
