"""API endpoints for MVP-10: Document AI Improvement Loop."""

from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import ResearchQuestionStatus
from app.db.session import get_db
from app.schemas.improvement import (
    ImprovementOverviewResponse,
    ResearchQuestionCreate,
    ResearchQuestionDetailResponse,
    ResearchQuestionListResponse,
    ResearchQuestionResponse,
    ResearchQuestionUpdate,
    ResearchSignal,
)
from app.services.feedback_analytics_service import FeedbackAnalyticsService
from app.services.research_question_service import ResearchQuestionService
from app.services.research_signal_service import ResearchSignalService

router = APIRouter(prefix="/improvement", tags=["Improvement Loop"])


def _to_question_response(q) -> ResearchQuestionResponse:
    return ResearchQuestionResponse(
        id=q.id,
        title=q.title,
        description=q.description,
        source_signal_id=q.source_signal_id,
        field=q.field,
        document_type=q.document_type,
        status=q.status,
        related_evidence_refs=q.related_evidence_refs_json or [],
        created_by=q.created_by,
        created_at=q.created_at,
        updated_at=q.updated_at,
    )


@router.get("/overview", response_model=ImprovementOverviewResponse)
async def get_improvement_overview(
    period: Optional[str] = Query("all", description="Temporal period preset: all, today, 7d, 30d, 90d"),
    date_from: Optional[datetime] = Query(None, description="Explicit start timestamp boundary (ISO 8601)"),
    date_to: Optional[datetime] = Query(None, description="Explicit end timestamp boundary (ISO 8601)"),
    db: AsyncSession = Depends(get_db),
) -> ImprovementOverviewResponse:
    """Consolidated improvement metrics, top corrected fields, confidence buckets, and counts."""
    analytics_svc = FeedbackAnalyticsService(db)
    signal_svc = ResearchSignalService(db)
    question_svc = ResearchQuestionService(db)

    overview = await analytics_svc.get_feedback_overview(
        date_from=date_from, date_to=date_to, period=period
    )

    signals = await signal_svc.derive_signals(
        date_from=date_from, date_to=date_to, period=period
    )

    open_q_count, _ = await question_svc.list_questions(
        status_filter=ResearchQuestionStatus.OPEN.value, limit=1
    )

    overview.active_signals_count = len(signals)
    overview.open_questions_count = open_q_count
    return overview


@router.get("/signals", response_model=List[ResearchSignal])
async def get_research_signals(
    period: Optional[str] = Query("all", description="Temporal period preset: all, today, 7d, 30d, 90d"),
    date_from: Optional[datetime] = Query(None, description="Explicit start timestamp boundary (ISO 8601)"),
    date_to: Optional[datetime] = Query(None, description="Explicit end timestamp boundary (ISO 8601)"),
    field: Optional[str] = Query(None, description="Optional target field name filter"),
    db: AsyncSession = Depends(get_db),
) -> List[ResearchSignal]:
    """Retrieve derived, non-persisted factual research observations meeting sample size thresholds."""
    signal_svc = ResearchSignalService(db)
    return await signal_svc.derive_signals(
        date_from=date_from, date_to=date_to, period=period, field_filter=field
    )


@router.get("/questions", response_model=ResearchQuestionListResponse)
async def list_research_questions(
    status: Optional[str] = Query(None, description="Optional status filter: OPEN, IN_PROGRESS, EXPERIMENT_AVAILABLE, CLOSED"),
    field: Optional[str] = Query(None, description="Optional field name filter"),
    limit: int = Query(50, ge=1, le=100, description="Page limit"),
    offset: int = Query(0, ge=0, description="Page offset"),
    db: AsyncSession = Depends(get_db),
) -> ResearchQuestionListResponse:
    """Retrieve paginated persistent research questions."""
    svc = ResearchQuestionService(db)
    total, items = await svc.list_questions(
        status_filter=status, field_filter=field, limit=limit, offset=offset
    )
    return ResearchQuestionListResponse(
        items=[_to_question_response(q) for q in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post("/questions", response_model=ResearchQuestionResponse, status_code=status.HTTP_201_CREATED)
async def create_research_question(
    data: ResearchQuestionCreate,
    db: AsyncSession = Depends(get_db),
) -> ResearchQuestionResponse:
    """Create and persist a new human-formulated research question."""
    svc = ResearchQuestionService(db)
    created = await svc.create_question(data)
    return _to_question_response(created)


@router.get("/questions/{question_id}", response_model=ResearchQuestionDetailResponse)
async def get_research_question_detail(
    question_id: str,
    db: AsyncSession = Depends(get_db),
) -> ResearchQuestionDetailResponse:
    """Retrieve research question detail enriched with related B0/B1/B2 benchmark data."""
    svc = ResearchQuestionService(db)
    return await svc.get_question_detail(question_id)


@router.patch("/questions/{question_id}", response_model=ResearchQuestionResponse)
async def update_research_question(
    question_id: str,
    data: ResearchQuestionUpdate,
    db: AsyncSession = Depends(get_db),
) -> ResearchQuestionResponse:
    """Update research question content or transition status (CLOSED preserves audit trail)."""
    svc = ResearchQuestionService(db)
    updated = await svc.update_question(question_id, data)
    return _to_question_response(updated)
