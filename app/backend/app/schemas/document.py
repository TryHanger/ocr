"""Document API Schemas."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.schemas.decision import DecisionExplanation
from app.schemas.kie import NormalizedFieldDTO
from app.schemas.ocr import NormalizedTokenDTO
from app.schemas.quality import QualityResultDTO
from app.schemas.research import ResearchEvidenceReference
from app.schemas.step import ProcessingStepDTO


class DocumentPageDTO(BaseModel):
    id: str
    page_number: int
    width: int
    height: int
    image_url: str
    full_text: Optional[str] = None
    tokens: List[NormalizedTokenDTO] = Field(default_factory=list)


class AuditEventDTO(BaseModel):
    id: str
    actor: str
    action: str
    field_name: Optional[str] = None
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    created_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentResponse(BaseModel):
    """Concise representation of a document in queues and lists."""

    id: str
    filename: str
    document_type: str
    status: str
    decision: Optional[str] = None
    confidence: Optional[float] = None
    review_reason: Optional[str] = None
    review_details: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime
    processing_started_at: Optional[datetime] = None
    processing_finished_at: Optional[datetime] = None
    processing_duration_ms: Optional[float] = None
    review_started_at: Optional[datetime] = None
    review_finished_at: Optional[datetime] = None
    review_duration_ms: Optional[float] = None
    review_wait_duration_ms: Optional[float] = None


class DocumentFieldCorrectionDTO(BaseModel):
    """Specific field correction recorded in the document audit trail."""

    field: str
    previous_value: Optional[str] = None
    final_value: Optional[str] = None
    corrected_at: datetime
    actor: str


class DocumentDetailResponse(DocumentResponse):
    """Detailed representation of a document for Document Inspector."""

    file_size_bytes: int
    mime_type: str
    file_url: str
    pages: List[DocumentPageDTO] = Field(default_factory=list)
    fields: Dict[str, NormalizedFieldDTO] = Field(default_factory=dict)
    quality: Optional[QualityResultDTO] = None
    audit_events: List[AuditEventDTO] = Field(default_factory=list)
    steps: List[ProcessingStepDTO] = Field(default_factory=list)
    research_evidence: List[ResearchEvidenceReference] = Field(default_factory=list)
    decision_explanation: Optional[DecisionExplanation] = None
    field_corrections: List[DocumentFieldCorrectionDTO] = Field(default_factory=list)


class DocumentListResponse(BaseModel):
    total: int
    items: List[DocumentResponse]
