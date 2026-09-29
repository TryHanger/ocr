"""Document relational model."""

from datetime import datetime
import uuid
from typing import Any, List, Optional
from sqlalchemy import DateTime, Enum as SQLEnum, Float, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.enums import AutomationDecision, DocumentStatus, DocumentType, ReviewReason
from app.db.base import Base


class Document(Base):
    """Primary document entity tracking lifecycle, automation decision, and relationships."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    document_type: Mapped[str] = mapped_column(
        String(50), nullable=False, default=DocumentType.RECEIPT.value
    )

    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default=DocumentStatus.UPLOADED.value, index=True
    )
    decision: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True, default=None, index=True
    )
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    review_reason: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True, default=ReviewReason.NONE.value
    )
    review_details_json: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)

    # Lifecycle Timestamps & Durations
    processing_started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    processing_finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    processing_duration_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    review_started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    review_finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    review_duration_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    review_wait_duration_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships with cascade delete
    pages: Mapped[List["DocumentPage"]] = relationship(
        "DocumentPage", back_populates="document", cascade="all, delete-orphan", order_by="DocumentPage.page_number"
    )
    jobs: Mapped[List["ProcessingJob"]] = relationship(
        "ProcessingJob", back_populates="document", cascade="all, delete-orphan", order_by="ProcessingJob.created_at.desc()"
    )
    fields: Mapped[List["ExtractedField"]] = relationship(
        "ExtractedField", back_populates="document", cascade="all, delete-orphan"
    )
    quality: Mapped[Optional["DocumentQuality"]] = relationship(
        "DocumentQuality", back_populates="document", uselist=False, cascade="all, delete-orphan"
    )
    reviews: Mapped[List["ReviewAction"]] = relationship(
        "ReviewAction", back_populates="document", cascade="all, delete-orphan", order_by="ReviewAction.created_at"
    )
    audit_events: Mapped[List["AuditEvent"]] = relationship(
        "AuditEvent", back_populates="document", cascade="all, delete-orphan", order_by="AuditEvent.created_at"
    )
    steps: Mapped[List["ProcessingStep"]] = relationship(
        "ProcessingStep", back_populates="document", cascade="all, delete-orphan", order_by="ProcessingStep.step_order"
    )
