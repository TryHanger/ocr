"""Analytics and Telemetry API router."""

from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.analytics import (
    AnalyticsOverviewResponse,
    AutomationTrendsResponse,
    ConfidenceAnalyticsResponse,
    ControlCenterResponse,
    DecisionReasonsAnalyticsResponse,
    DocumentTypesResponse,
    FieldCorrectionsResponse,
    ProductionFeedbackResponse,
    QueueAnalyticsResponse,
    ReviewReasonsResponse,
)
from app.services.analytics_service import AnalyticsService

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/overview", response_model=AnalyticsOverviewResponse)
async def get_analytics_overview(
    date_from: Optional[datetime] = Query(None, alias="from"),
    date_to: Optional[datetime] = Query(None, alias="to"),
    period: Optional[str] = Query("all"),
    db: AsyncSession = Depends(get_db),
) -> AnalyticsOverviewResponse:
    """Retrieve operational KPIs, automation rates, and root-cause distributions."""
    service = AnalyticsService(db)
    return await service.get_overview(date_from=date_from, date_to=date_to, period=period)


@router.get("/document-types", response_model=DocumentTypesResponse)
async def get_document_types_analytics(
    date_from: Optional[datetime] = Query(None, alias="from"),
    date_to: Optional[datetime] = Query(None, alias="to"),
    period: Optional[str] = Query("all"),
    db: AsyncSession = Depends(get_db),
) -> DocumentTypesResponse:
    """Retrieve volume, automation rates, review rates, and latency per document category."""
    service = AnalyticsService(db)
    return await service.get_document_types_analytics(date_from=date_from, date_to=date_to, period=period)


@router.get("/trends", response_model=AutomationTrendsResponse)
async def get_automation_trends(
    date_from: Optional[datetime] = Query(None, alias="from"),
    date_to: Optional[datetime] = Query(None, alias="to"),
    period: Optional[str] = Query("7d"),
    db: AsyncSession = Depends(get_db),
) -> AutomationTrendsResponse:
    """Retrieve daily volume and automation rate timeline."""
    service = AnalyticsService(db)
    return await service.get_automation_trends(date_from=date_from, date_to=date_to, period=period)


@router.get("/queue", response_model=QueueAnalyticsResponse)
async def get_queue_analytics(
    db: AsyncSession = Depends(get_db),
) -> QueueAnalyticsResponse:
    """Retrieve real-time operational backlog breakdown."""
    service = AnalyticsService(db)
    return await service.get_queue()


@router.get("/review-reasons", response_model=ReviewReasonsResponse)
async def get_review_reasons(
    date_from: Optional[datetime] = Query(None, alias="from"),
    date_to: Optional[datetime] = Query(None, alias="to"),
    period: Optional[str] = Query("all"),
    db: AsyncSession = Depends(get_db),
) -> ReviewReasonsResponse:
    """Retrieve diagnostic distribution of manual review root cause reasons."""
    service = AnalyticsService(db)
    return await service.get_review_reasons(date_from=date_from, date_to=date_to, period=period)


@router.get("/corrections", response_model=FieldCorrectionsResponse)
async def get_field_corrections(
    date_from: Optional[datetime] = Query(None, alias="from"),
    date_to: Optional[datetime] = Query(None, alias="to"),
    period: Optional[str] = Query("all"),
    db: AsyncSession = Depends(get_db),
) -> FieldCorrectionsResponse:
    """Retrieve per-field operator correction rates and volumes."""
    service = AnalyticsService(db)
    return await service.get_field_corrections(date_from=date_from, date_to=date_to, period=period)


@router.get("/confidence", response_model=ConfidenceAnalyticsResponse)
async def get_confidence_distribution(
    date_from: Optional[datetime] = Query(None, alias="from"),
    date_to: Optional[datetime] = Query(None, alias="to"),
    period: Optional[str] = Query("all"),
    db: AsyncSession = Depends(get_db),
) -> ConfidenceAnalyticsResponse:
    """Retrieve confidence score buckets and their correlation with manual review and corrections."""
    service = AnalyticsService(db)
    return await service.get_confidence_distribution(date_from=date_from, date_to=date_to, period=period)


@router.get("/decision-reasons", response_model=DecisionReasonsAnalyticsResponse)
async def get_decision_reasons_analytics(
    date_from: Optional[datetime] = Query(None, alias="from"),
    date_to: Optional[datetime] = Query(None, alias="to"),
    period: Optional[str] = Query("all"),
    db: AsyncSession = Depends(get_db),
) -> DecisionReasonsAnalyticsResponse:
    """Retrieve aggregated breakdown of decision evaluation reasons, categories, and failing fields."""
    service = AnalyticsService(db)
    return await service.get_decision_reasons_analytics(date_from=date_from, date_to=date_to, period=period)


@router.get("/feedback", response_model=ProductionFeedbackResponse)
async def get_production_feedback_analytics(
    date_from: Optional[datetime] = Query(None, alias="from"),
    date_to: Optional[datetime] = Query(None, alias="to"),
    period: Optional[str] = Query("all"),
    db: AsyncSession = Depends(get_db),
) -> ProductionFeedbackResponse:
    """Retrieve production feedback analytics, review-correction correlations, and automation loss funnel."""
    service = AnalyticsService(db)
    return await service.get_production_feedback(date_from=date_from, date_to=date_to, period=period)


@router.get("/control-center", response_model=ControlCenterResponse)
async def get_control_center_analytics(
    date_from: Optional[datetime] = Query(None, alias="from"),
    date_to: Optional[datetime] = Query(None, alias="to"),
    period: Optional[str] = Query("30d"),
    db: AsyncSession = Depends(get_db),
) -> ControlCenterResponse:
    """Retrieve unified operational Control Center aggregation payload."""
    service = AnalyticsService(db)
    return await service.get_control_center(date_from=date_from, date_to=date_to, period=period)

