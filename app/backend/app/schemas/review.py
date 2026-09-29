"""Manual Review Schemas."""

from typing import Optional
from pydantic import BaseModel, Field


class FieldCorrectionRequest(BaseModel):
    """Operator modification of an extracted field value."""

    field_name: str
    corrected_value: str
    reviewer_name: str = "operator"
    notes: Optional[str] = None


class ApproveDocumentRequest(BaseModel):
    """Operator approval of document with review resolution."""

    reviewer_name: str = "operator"
    notes: Optional[str] = None


class RejectDocumentRequest(BaseModel):
    """Operator rejection of document."""

    reviewer_name: str = "operator"
    reason: str
    notes: Optional[str] = None


class ReviewResultResponse(BaseModel):
    document_id: str
    status: str
    decision: str
    message: str
