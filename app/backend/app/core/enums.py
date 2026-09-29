"""Core enumerations for Document AI Control Center."""

from enum import Enum


class DocumentStatus(str, Enum):
    """Explicit lifecycle status for a document."""

    UPLOADED = "uploaded"
    PROCESSING = "processing"
    QUALITY_ANALYZED = "quality_analyzed"
    PREPROCESSED = "preprocessed"
    OCR_COMPLETED = "ocr_completed"
    KIE_COMPLETED = "kie_completed"
    VALIDATED = "validated"
    DECISION_MADE = "decision_made"
    COMPLETED_AUTOMATIC = "completed_automatic"
    COMPLETED_MANUAL = "completed_manual"
    MANUAL_REVIEW = "manual_review"
    ERROR = "error"


class JobStatus(str, Enum):
    """Status of background processing jobs."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class DocumentType(str, Enum):
    """Supported document types."""

    RECEIPT = "receipt"
    INVOICE = "invoice"
    ACT = "act"
    CONTRACT = "contract"
    UNKNOWN = "unknown"


class AutomationDecision(str, Enum):
    """Business decision resulting from confidence and quality evaluation."""

    AUTOMATIC = "automatic"
    MANUAL_REVIEW = "manual_review"
    REJECTED = "rejected"


class ReviewReason(str, Enum):
    """Structured root cause for manual review routing."""

    NONE = "none"
    LOW_CONFIDENCE = "low_confidence"
    LOW_IMAGE_QUALITY = "low_image_quality"
    MISSING_REQUIRED_FIELD = "missing_required_field"
    VALIDATION_FAILED = "validation_failed"
    INVALID_NUMBER = "invalid_number"
    INVALID_DATE = "invalid_date"
    EMPTY_VALUE = "empty_value"
    DOCUMENT_INTEGRITY = "document_integrity"
    OCR_ERROR = "ocr_error"
    KIE_ERROR = "kie_error"
    PIPELINE_FAILURE = "pipeline_failure"
    SYSTEM_ERROR = "system_error"


class AuditAction(str, Enum):
    """Actions recorded in append-only audit trail."""

    DOCUMENT_UPLOADED = "document_uploaded"
    JOB_STARTED = "job_started"
    QUALITY_ANALYZED = "quality_analyzed"
    PREPROCESSING_APPLIED = "preprocessing_applied"
    OCR_COMPLETED = "ocr_completed"
    KIE_COMPLETED = "kie_completed"
    DECISION_EVALUATED = "decision_evaluated"
    FIELD_CORRECTED = "field_corrected"
    DOCUMENT_APPROVED = "document_approved"
    REVIEW_COMPLETED = "review_completed"
    DOCUMENT_REJECTED = "document_rejected"
    JOB_RETRIED = "job_retried"
    PROCESSING_FAILED = "processing_failed"


class ResearchQuestionStatus(str, Enum):
    """Lifecycle state of an intentional research investigation."""

    OPEN = "OPEN"
    IN_PROGRESS = "IN_PROGRESS"
    EXPERIMENT_AVAILABLE = "EXPERIMENT_AVAILABLE"
    CLOSED = "CLOSED"


class ResearchSignalType(str, Enum):
    """Canonical classification for derived observational signals."""

    CORRECTION_CONCENTRATION = "CORRECTION_CONCENTRATION"
    CONFIDENCE_CORRECTION_PATTERN = "CONFIDENCE_CORRECTION_PATTERN"
    REVIEW_REASON_CORRECTION_PATTERN = "REVIEW_REASON_CORRECTION_PATTERN"
    QUALITY_CORRECTION_PATTERN = "QUALITY_CORRECTION_PATTERN"
    FIELD_VALIDATION_PATTERN = "FIELD_VALIDATION_PATTERN"

