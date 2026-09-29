"""ReviewAction relational model for operator interventions."""

from datetime import datetime
import uuid
from typing import Optional
from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.enums import ReviewReason
from app.db.base import Base


class ReviewAction(Base):
    """Tracks human review approvals, corrections, or rejections."""

    __tablename__ = "review_actions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    reviewer_name: Mapped[str] = mapped_column(String(100), nullable=False)
    action: Mapped[str] = mapped_column(String(50), nullable=False)  # APPROVED, REJECTED, CORRECTED
    reason: Mapped[str] = mapped_column(String(50), nullable=False, default=ReviewReason.NONE.value)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    document: Mapped["Document"] = relationship("Document", back_populates="reviews")
