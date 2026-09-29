"""Pydantic schemas for MVP-10: Document AI Improvement Loop."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.core.enums import ResearchQuestionStatus, ResearchSignalType
from app.schemas.analytics import CorrectionDocumentTypeMetric, CorrectionReasonMetric


class FieldCorrectionMetric(BaseModel):
    """Field-level operator correction frequencies with dual rate and volume metrics."""

    field: str
    correction_count: int
    evaluated_count: int
    correction_rate: float  # (correction_count / evaluated_count) * 100
    share_of_all_corrections: float  # (correction_count / total_field_corrections) * 100
    affected_documents: int


class ConfidenceCorrectionBucket(BaseModel):
    """Evaluated and corrected field counts grouped by original KIE confidence interval."""

    bucket: str
    evaluated_fields: int
    corrected_fields: int
    correction_rate: float  # (corrected_fields / evaluated_fields) * 100
    share_of_corrections: float  # (corrected_fields / total_corrected_fields) * 100


class ResearchSignal(BaseModel):
    """Derived, non-persisted factual observation highlighting patterns in production data."""

    id: str  # Deterministic hash ID
    signal_type: ResearchSignalType
    title: str
    description: str
    population: Dict[str, Any] = Field(default_factory=dict)
    field: Optional[str] = None
    document_type: Optional[str] = None
    evidence: Dict[str, Any] = Field(default_factory=dict)
    period: str = "all"
    related_evidence_refs: List[Dict[str, Any]] = Field(default_factory=list)


class ResearchQuestionCreate(BaseModel):
    """Payload to create a new persistent research investigation."""

    title: str
    description: str
    source_signal_id: Optional[str] = None
    field: Optional[str] = None
    document_type: Optional[str] = None
    related_evidence_refs: Optional[List[Dict[str, Any]]] = None
    created_by: str = "researcher"


class ResearchQuestionUpdate(BaseModel):
    """Payload to update status or descriptive contents of a research question."""

    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[ResearchQuestionStatus] = None
    related_evidence_refs: Optional[List[Dict[str, Any]]] = None


class ResearchQuestionResponse(BaseModel):
    """Structured response for a persistent research question."""

    id: str
    title: str
    description: str
    source_signal_id: Optional[str] = None
    field: Optional[str] = None
    document_type: Optional[str] = None
    status: str
    related_evidence_refs: List[Dict[str, Any]] = Field(default_factory=list)
    created_by: str
    created_at: datetime
    updated_at: datetime


class ResearchQuestionListResponse(BaseModel):
    """Paginated list of persistent research questions."""

    items: List[ResearchQuestionResponse]
    total: int
    limit: int
    offset: int


class ResearchQuestionDetailResponse(ResearchQuestionResponse):
    """Enriched research question response decorated with research benchmark summaries."""

    enriched_evidence: Optional[Dict[str, Any]] = None


class ImprovementOverviewSummary(BaseModel):
    """Factual high-level KPIs for the Improvement Loop."""

    documents_processed: int
    documents_reviewed: int
    documents_corrected: int
    total_field_corrections: int
    correction_rate: float
    no_change_review_rate: float


class ImprovementOverviewResponse(BaseModel):
    """Consolidated improvement loop operational summary and distributions."""

    period: str
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    summary: ImprovementOverviewSummary
    by_field: List[FieldCorrectionMetric] = Field(default_factory=list)
    by_confidence: List[ConfidenceCorrectionBucket] = Field(default_factory=list)
    by_reason: List[CorrectionReasonMetric] = Field(default_factory=list)
    by_document_type: List[CorrectionDocumentTypeMetric] = Field(default_factory=list)
    active_signals_count: int = 0
    open_questions_count: int = 0
