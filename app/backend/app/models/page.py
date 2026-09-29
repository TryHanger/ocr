"""DocumentPage relational model for multi-page documents."""

from datetime import datetime
import uuid
from typing import Any, Dict, List, Optional
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.db.base import Base


class DocumentPage(Base):
    """Represents a single rasterized or uploaded page of a document."""

    __tablename__ = "document_pages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    image_path: Mapped[str] = mapped_column(String(512), nullable=False)

    full_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Normalized tokens: [{text: "...", bbox: [x1, y1, x2, y2], confidence: 0.95}]
    tokens_json: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    document: Mapped["Document"] = relationship("Document", back_populates="pages")
