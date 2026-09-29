"""ProcessingStep relational model for recording end-to-end pipeline provenance."""

from datetime import datetime
import uuid
from typing import Any, Optional
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.db.base import Base


class ProcessingStep(Base):
    """Immutable audit record of an individual operation applied to a document."""

    __tablename__ = "processing_steps"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    step_order: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    operation: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    parameters_json: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    duration_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="APPLIED")

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    document: Mapped["Document"] = relationship("Document", back_populates="steps")
