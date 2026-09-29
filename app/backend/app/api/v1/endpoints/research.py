from typing import Optional

from fastapi import APIRouter, Query

from app.adapters.research_metrics import default_research_metrics
from app.schemas.research import (
    DegradationExplorerResponse,
    PreprocessingExplorerResponse,
    ResearchExperimentDetail,
    ResearchExperimentsResponse,
    ResearchOverviewResponse,
    ResearchSummaryResponse,
)
from app.services.research_service import default_research_service

router = APIRouter(prefix="/research", tags=["Research Mode"])


@router.get("/degradations", response_model=DegradationExplorerResponse)
async def get_degradations(
    degradation: Optional[str] = Query(None, description="Optional degradation filter (e.g. b1_gaussian_blur, gaussian_blur, or D1)")
) -> DegradationExplorerResponse:
    """Retrieve detailed degradation explorer data and curves across B1 conditions."""
    return await default_research_service.get_degradations(degradation=degradation)


@router.get("/preprocessing", response_model=PreprocessingExplorerResponse)
async def get_preprocessing(
    degradation: Optional[str] = Query(None, description="Optional degradation filter (e.g. D6, rotation, or b1_rotation)"),
    severity: Optional[str] = Query(None, description="Optional severity filter (e.g. S1, S2, S3, S4, or 1..4)"),
    policy: Optional[str] = Query(None, description="Optional policy filter (e.g. standard_receipt_enhancement or p_standard_receipt_enhancement)"),
) -> PreprocessingExplorerResponse:
    """Retrieve detailed preprocessing explorer data across B2 conditions."""
    return await default_research_service.get_preprocessing(
        degradation=degradation,
        severity=severity,
        policy=policy,
    )


@router.get("/overview", response_model=ResearchOverviewResponse)
async def get_research_overview() -> ResearchOverviewResponse:
    """Retrieve high-level overview of available research tracks and metrics."""
    return await default_research_service.get_overview()


@router.get("/experiments", response_model=ResearchExperimentsResponse)
async def list_research_experiments(
    track: Optional[str] = Query(None, description="Optional track filter: B0, B1, B2")
) -> ResearchExperimentsResponse:
    """List registered research experiments with optional track filtering."""
    return await default_research_service.get_experiments(track=track)


@router.get("/experiments/{experiment_id}", response_model=ResearchExperimentDetail)
async def get_research_experiment(experiment_id: str) -> ResearchExperimentDetail:
    """Retrieve detailed condition metrics and findings for a specific experiment ID."""
    return await default_research_service.get_experiment_detail(experiment_id=experiment_id)


@router.get("/summary", response_model=ResearchSummaryResponse)
async def get_research_summary() -> ResearchSummaryResponse:
    """Retrieve actual B0, B1, and B2 research benchmarks from experimental runs (legacy)."""
    return await default_research_metrics.get_summary()
