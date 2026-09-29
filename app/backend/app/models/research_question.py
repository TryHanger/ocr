"""ResearchQuestion relational model for user-created investigation hypotheses."""

from datetime import datetime
from typing import Any, Optional
import uuid
from sqlalchemy import DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from app.core.enums import ResearchQuestionStatus
from app.db.base import Base


class ResearchQuestion(Base):
    """Represents an intentional, human-formulated research question or hypothesis.

    Can originate from an observed ResearchSignal or be manually created by a researcher.
    Preserves links to affected fields, triggering signals, and relevant B0/B1/B2 evidence references.
    """

    __tablename__ = "research_questions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)

    # Deterministic reference to originating signal (if derived from one)
    source_signal_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)

    # Contextual scope
    field: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    document_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Status: OPEN, IN_PROGRESS, EXPERIMENT_AVAILABLE, CLOSED
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default=ResearchQuestionStatus.OPEN.value, index=True
    )

    # Stored list of informational experimental evidence references (B0, B1, B2)
    related_evidence_refs_json: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)

    created_by: Mapped[str] = mapped_column(String(100), nullable=False, default="researcher")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
