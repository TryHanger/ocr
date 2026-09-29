"""Operational Action & Resolution Center Schemas.

Defines typed DTOs for derived operational issues, declarative action registry,
evidence references, and live backlog projections.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class IssueCategory(str, Enum):
    """Semantic domain category of an operational issue."""

    QUALITY = "quality"
    CONFIDENCE = "confidence"
    VALIDATION = "validation"
    MANUAL_REVIEW = "manual_review"
    PROCESSING = "processing"
    CORRECTION = "correction"


class IssueSeverity(str, Enum):
    """Deterministic operational classification for triage.

    Note: This is strictly rule-derived from application states; it is NOT an AI/ML score.
    """

    CRITICAL = "critical"  # Terminal processing failure / unreadable document
    HIGH = "high"          # Blocking validation error / mandatory manual review condition
    MEDIUM = "medium"      # Low extraction confidence / image quality degradation
    LOW = "low"            # Non-blocking advisory or warning
    INFO = "info"          # Resolved state / historical correction record


class ResolutionStatus(str, Enum):
    """Derived human-in-the-loop lifecycle state."""

    OPEN = "open"          # Requires manual review; review not yet started
    IN_REVIEW = "in_review"  # Active operator review underway
    RESOLVED = "resolved"  # Processed straight-through, or completed by human operator


class ActionType(str, Enum):
    """Supported declarative operator actions."""

    OPEN_DOCUMENT = "open_document"
    OPEN_REVIEW = "open_review"
    OPEN_FIELD = "open_field"
    OPEN_QUALITY_EVIDENCE = "open_quality_evidence"
    OPEN_VALIDATION = "open_validation"
    OPEN_RESEARCH_EVIDENCE = "open_research_evidence"
    RETRY_PROCESSING = "retry_processing"


class OperationalAction(BaseModel):
    """Declarative operator action pointing to existing workflows."""

    id: str
    type: ActionType
    label: str
    description: str
    enabled: bool = True
    target: str = Field(..., description="Target UI tab or view (e.g. 'fields', 'quality', 'review', 'research')")
    params: Dict[str, Any] = Field(default_factory=dict, description="Contextual parameters (field_name, condition_id, etc.)")


class IssueEvidence(BaseModel):
    """Tri-layer evidence container for operational explainability."""

    production: Optional[str] = Field(default=None, description="Factual observed value vs policy threshold")
    historical: Optional[str] = Field(default=None, description="Descriptive counts of similar reviews from audit trail")
    research: Optional[str] = Field(default=None, description="Informational reference to B1/B2 benchmark experiments")


class OperationalIssue(BaseModel):
    """Derived read-model representing an actionable operational issue."""

    id: str
    document_id: str
    category: IssueCategory
    severity: IssueSeverity
    code: str
    title: str
    description: str
    observed_value: Optional[Union[float, str]] = None
    threshold: Optional[Union[float, str]] = None
    field: Optional[str] = None
    source: str
    status: ResolutionStatus
    evidence: IssueEvidence = Field(default_factory=IssueEvidence)
    available_actions: List[OperationalAction] = Field(default_factory=list)
    created_at: Optional[datetime] = None


class DocumentOperationsResponse(BaseModel):
    """Complete operational state response for a document."""

    document_id: str
    overall_status: ResolutionStatus
    total_issues: int
    blocking_issues: int
    issues: List[OperationalIssue] = Field(default_factory=list)
    summary: str


class OperationsBacklogSummary(BaseModel):
    """Live projection of operational queue backlog."""

    total_open: int = 0
    total_in_review: int = 0
    by_category: Dict[str, int] = Field(default_factory=dict)
    by_reason: Dict[str, int] = Field(default_factory=dict)
