"""DocumentQuality relational model."""

from datetime import datetime
import uuid
from typing import Any, Optional
from sqlalchemy import DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.db.base import Base


class DocumentQuality(Base):
    """Image quality metrics and diagnostic breakdown for a document."""

    __tablename__ = "document_qualities"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), unique=True, nullable=False, index=True
    )

    blur_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)  # Laplacian variance
    contrast_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)  # RMS contrast
    noise_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)  # High-freq estimate
    rotation_angle: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)  # Deskew angle in degrees
    resolution_dpi: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    quality_score: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)  # 0.0 - 1.0 aggregate
    profile_summary: Mapped[str] = mapped_column(String(255), nullable=False, default="Normal")
    metrics_json: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    document: Mapped["Document"] = relationship("Document", back_populates="quality")
