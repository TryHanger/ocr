"""ExtractedField relational model for structured KIE output."""

from datetime import datetime
import uuid
from typing import Any, Optional
from sqlalchemy import DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.db.base import Base


class ExtractedField(Base):
    """Represents a structured field extracted from the document with provenance and confidence."""

    __tablename__ = "extracted_fields"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    field_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    corrected_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    validation_status: Mapped[str] = mapped_column(String(50), nullable=False, default="VALID")

    # Normalized bounding box [x_min, y_min, x_max, y_max] in range 0.0..1.0
    bbox_json: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    source_tokens_json: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    document: Mapped["Document"] = relationship("Document", back_populates="fields")

    @property
    def original_value(self) -> str:
        return self.value
