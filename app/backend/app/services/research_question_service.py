"""Research Question Service for managing persistent researcher questions and investigations."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import uuid
from fastapi import HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import ResearchQuestionStatus
from app.models.research_question import ResearchQuestion
from app.schemas.improvement import (
    ResearchQuestionCreate,
    ResearchQuestionDetailResponse,
    ResearchQuestionResponse,
    ResearchQuestionUpdate,
)
from app.services.research_service import default_research_service


class ResearchQuestionService:
    """Provides persistent lifecycle management for human research hypotheses and questions."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def list_questions(
        self,
        status_filter: Optional[str] = None,
        field_filter: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[int, List[ResearchQuestion]]:
        """Retrieve paginated research questions with optional status/field filtering."""
        count_stmt = select(func.count(ResearchQuestion.id))
        query_stmt = select(ResearchQuestion).order_by(ResearchQuestion.created_at.desc())

        if status_filter:
            count_stmt = count_stmt.where(ResearchQuestion.status == status_filter)
            query_stmt = query_stmt.where(ResearchQuestion.status == status_filter)

        if field_filter:
            count_stmt = count_stmt.where(ResearchQuestion.field == field_filter)
            query_stmt = query_stmt.where(ResearchQuestion.field == field_filter)

        total = (await self.session.execute(count_stmt)).scalar() or 0
        items = (
            await self.session.execute(query_stmt.limit(limit).offset(offset))
        ).scalars().all()

        return total, list(items)

    async def get_question(self, question_id: str) -> Optional[ResearchQuestion]:
        """Fetch a specific research question by its unique identifier."""
        stmt = select(ResearchQuestion).where(ResearchQuestion.id == question_id)
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def create_question(self, data: ResearchQuestionCreate) -> ResearchQuestion:
        """Persist a new human-formulated research question with optional signal provenance."""
        question = ResearchQuestion(
            id=str(uuid.uuid4()),
            title=data.title.strip(),
            description=data.description.strip(),
            source_signal_id=data.source_signal_id,
            field=data.field,
            document_type=data.document_type,
            status=ResearchQuestionStatus.OPEN.value,
            related_evidence_refs_json=data.related_evidence_refs,
            created_by=data.created_by or "researcher",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        self.session.add(question)
        await self.session.commit()
        await self.session.refresh(question)
        return question

    async def update_question(
        self,
        question_id: str,
        data: ResearchQuestionUpdate,
    ) -> ResearchQuestion:
        """Update research question content or transition its status (soft lifecycle end-state is CLOSED)."""
        question = await self.get_question(question_id)
        if not question:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Research question {question_id} not found",
            )

        if data.title is not None:
            question.title = data.title.strip()
        if data.description is not None:
            question.description = data.description.strip()
        if data.status is not None:
            question.status = data.status.value if hasattr(data.status, "value") else str(data.status)
        if data.related_evidence_refs is not None:
            question.related_evidence_refs_json = data.related_evidence_refs

        question.updated_at = datetime.utcnow()
        await self.session.commit()
        await self.session.refresh(question)
        return question

    async def get_question_detail(self, question_id: str) -> ResearchQuestionDetailResponse:
        """Fetch question and enrich with read-only summaries of linked B0/B1/B2 research evidence."""
        question = await self.get_question(question_id)
        if not question:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Research question {question_id} not found",
            )

        refs = question.related_evidence_refs_json or []
        enriched_info: Dict[str, Any] = {}

        # Look up any referenced experiment details safely
        for ref in refs:
            if isinstance(ref, dict):
                exp_id = ref.get("experiment_id") or ref.get("b1_experiment_id")
                if exp_id and exp_id in default_research_service.registry:
                    try:
                        detail = await default_research_service.get_experiment_detail(exp_id)
                        enriched_info[exp_id] = {
                            "name": detail.name,
                            "track": detail.track,
                            "metrics_summary": detail.metrics_summary,
                        }
                    except Exception:
                        pass

        return ResearchQuestionDetailResponse(
            id=question.id,
            title=question.title,
            description=question.description,
            source_signal_id=question.source_signal_id,
            field=question.field,
            document_type=question.document_type,
            status=question.status,
            related_evidence_refs=refs,
            created_by=question.created_by,
            created_at=question.created_at,
            updated_at=question.updated_at,
            enriched_evidence=enriched_info if enriched_info else None,
        )
