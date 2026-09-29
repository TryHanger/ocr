"""ProcessingJob relational model for worker queue execution."""

from datetime import datetime
import uuid
from typing import Optional
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.enums import DocumentStatus, JobStatus
from app.db.base import Base


class ProcessingJob(Base):
    """Tracks background execution, retries, attempts, and error diagnoses."""

    __tablename__ = "processing_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default=JobStatus.PENDING.value, index=True
    )
    current_stage: Mapped[str] = mapped_column(
        String(50), nullable=False, default=DocumentStatus.UPLOADED.value
    )
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=0, index=True)

    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_stage: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    document: Mapped["Document"] = relationship("Document", back_populates="jobs")
