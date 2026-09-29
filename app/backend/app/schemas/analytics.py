from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from app.schemas.document import DocumentResponse


class OverviewKPIs(BaseModel):
    total_documents: int
    processed_count: int
    automatic_count: int
    manual_review_count: int
    completed_manual_count: int = 0
    error_count: int
    automation_rate: float
    manual_review_rate: float = 0.0
    failure_rate: float = 0.0
    corrections_total: int = 0
    average_confidence: float
    average_processing_time_ms: float
    average_review_time_ms: float = 0.0
    queue_pending_count: int


class BreakdownItem(BaseModel):
    category: str
    count: int
    percentage: float


class AnalyticsOverviewResponse(BaseModel):
    kpis: OverviewKPIs
    review_reasons: List[BreakdownItem]
    document_type_distribution: List[BreakdownItem]
    status_distribution: List[BreakdownItem]


class ReviewReasonMetric(BaseModel):
    reason: str
    count: int
    percentage: float


class ReviewReasonsResponse(BaseModel):
    total_reviews: int
    items: List[ReviewReasonMetric]


class FieldCorrectionMetric(BaseModel):
    field: str
    total_occurrences: int
    corrected_count: int
    correction_rate: float


class FieldCorrectionsResponse(BaseModel):
    total_fields: int
    total_corrections: int
    items: List[FieldCorrectionMetric]


class ConfidenceBucket(BaseModel):
    bucket: str
    total_count: int
    manual_review_count: int
    corrected_count: int
    review_rate: float = 0.0


class ConfidenceAnalyticsResponse(BaseModel):
    distribution: List[ConfidenceBucket]
    average_confidence: float


class DocumentTypeAnalytics(BaseModel):
    """Granular operational and automation performance metrics per document type."""

    document_type: str
    total_documents: int
    processed_documents: int
    automatic_count: int
    manual_review_count: int
    error_count: int
    automation_rate: float
    manual_review_rate: float
    failure_rate: float
    average_confidence: float
    average_processing_time_ms: float
    average_review_time_ms: float


class DocumentTypesResponse(BaseModel):
    items: List[DocumentTypeAnalytics]


class AutomationTrendItem(BaseModel):
    """Daily operational and automation volume breakdown."""

    date: str  # YYYY-MM-DD
    total_documents: int
    processed_count: int
    automatic_count: int
    manual_review_count: int
    error_count: int
    automation_rate: float


class AutomationTrendsResponse(BaseModel):
    items: List[AutomationTrendItem]


class QueueAnalyticsResponse(BaseModel):
    manual_review: int
    processing: int
    errors: int


class DecisionReasonItem(BaseModel):
    code: str
    category: str
    count: int
    percentage: float
    description: str


class DecisionReasonCategoryMetric(BaseModel):
    category: str
    count: int
    percentage: float


class DecisionReasonFieldMetric(BaseModel):
    field: str
    count: int
    percentage: float


class DecisionReasonsAnalyticsResponse(BaseModel):
    """Operational and historical decision breakdown derived from DECISION_EVALUATED events."""

    total_decisions: int
    automatic_count: int
    manual_review_count: int
    automation_rate: float
    manual_review_rate: float
    by_reason: List[DecisionReasonItem] = Field(default_factory=list)
    by_category: List[DecisionReasonCategoryMetric] = Field(default_factory=list)
    by_field: List[DecisionReasonFieldMetric] = Field(default_factory=list)


class CorrectionSummary(BaseModel):
    """High-level production feedback and review outcome metrics."""

    total_correction_events: int
    documents_with_corrections: int
    completed_manual_review_documents: int
    correction_rate: float
    no_change_review_rate: float
    avg_corrections_per_corrected_document: float


class FeedbackFunnel(BaseModel):
    """Operational automation loss and human review resolution funnel."""

    processed: int
    automatic: int
    manual_review: int
    completed_manual_review: int
    with_corrections: int
    without_corrections: int
    automatic_rate: float
    manual_review_rate: float
    correction_rate: float
    no_change_review_rate: float


class CorrectionFieldMetric(BaseModel):
    """Field-level operator correction frequencies and share."""

    field: str
    correction_count: int
    affected_documents: int
    share: float


class CorrectionDocumentTypeMetric(BaseModel):
    """Correction frequencies and rates segmented by document type."""

    document_type: str
    reviewed_documents: int
    documents_with_corrections: int
    correction_events: int
    correction_rate: float


class CorrectionReasonMetric(BaseModel):
    """Correction frequencies and rates segmented by triggering review reason."""

    reason: str
    reviewed_documents: int
    documents_with_corrections: int
    correction_events: int
    correction_rate: float


class CorrectionReasonFieldMetric(BaseModel):
    """Co-occurrence matrix between triggering review reason and corrected field."""

    review_reason: str
    field: str
    correction_count: int
    affected_documents: int


class ProductionFeedbackResponse(BaseModel):
    """Complete production feedback analytics response."""

    summary: CorrectionSummary
    funnel: FeedbackFunnel
    by_field: List[CorrectionFieldMetric] = Field(default_factory=list)
    by_document_type: List[CorrectionDocumentTypeMetric] = Field(default_factory=list)
    by_reason: List[CorrectionReasonMetric] = Field(default_factory=list)
    by_reason_field: List[CorrectionReasonFieldMetric] = Field(default_factory=list)


class ControlCenterPeriod(BaseModel):
    """Temporal window parameters active for the Control Center query."""

    period: str
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None


class ControlCenterSummary(BaseModel):
    """Consolidated operational overview KPIs for the selected reporting period."""

    total_documents: int
    processed_count: int
    automatic_count: int
    manual_review_count: int
    completed_manual_count: int
    error_count: int
    automation_rate: float
    manual_review_rate: float
    correction_rate: float
    no_change_review_rate: float
    average_confidence: float  # Normalized 0.0..1.0 scale
    average_processing_time_ms: float
    average_review_time_ms: float


class ControlCenterQuality(BaseModel):
    """Aggregate quality and confidence signals across processed documents (normalized 0.0..1.0)."""

    average_quality_score: Optional[float] = None  # 0.0..1.0 image quality score
    average_document_confidence: Optional[float] = None  # 0.0..1.0 composite confidence
    average_ocr_confidence: Optional[float] = None  # 0.0..1.0 token confidence
    average_kie_confidence: Optional[float] = None  # 0.0..1.0 field confidence
    validation_pass_rate: Optional[float] = None  # 0.0..1.0 document validation pass rate
    total_documents_validated: int = 0
    passed_validation_count: int = 0
    failed_validation_count: int = 0


class ControlCenterResponse(BaseModel):
    """Unified Control Center aggregation payload."""

    period: ControlCenterPeriod
    summary: ControlCenterSummary
    queue: QueueAnalyticsResponse
    funnel: FeedbackFunnel
    quality: ControlCenterQuality
    review_reasons: List[DecisionReasonItem] = Field(default_factory=list)
    feedback: ProductionFeedbackResponse
    trends: List[AutomationTrendItem] = Field(default_factory=list)
    recent_documents: List[DocumentResponse] = Field(default_factory=list)


