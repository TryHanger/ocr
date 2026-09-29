"""Operational Action & Resolution Center API router."""

from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import AuditAction, ReviewReason
from app.db.session import get_db
from app.models.audit import AuditEvent
from app.models.document import Document
from app.schemas.operations import (
    DocumentOperationsResponse,
    OperationalAction,
    OperationsBacklogSummary,
)
from app.services.document_service import DocumentService
from app.services.operational_issues_service import OperationalIssuesService

router = APIRouter(prefix="/operations", tags=["Operations"])


async def _fetch_factual_historical_summary(
    db: AsyncSession,
    document: Document,
) -> Optional[Dict[str, Dict[str, int]]]:
    """Retrieve strictly factual historical review & correction counts for the document's review reason."""
    reason = document.review_reason
    if not reason or reason == ReviewReason.NONE.value:
        return None

    # Count other completed or reviewed documents with identical canonical review reason
    stmt = (
        select(Document.id, func.count(AuditEvent.id))
        .outerjoin(
            AuditEvent,
            (AuditEvent.document_id == Document.id) & (AuditEvent.action == AuditAction.FIELD_CORRECTED.value),
        )
        .where(Document.review_reason == reason, Document.id != document.id)
        .group_by(Document.id)
    )
    rows = (await db.execute(stmt)).all()
    if not rows:
        return None

    total = len(rows)
    corrections = sum(1 for r in rows if r[1] > 0)
    no_corrections = total - corrections

    return {
        reason: {
            "total": total,
            "corrections": corrections,
            "no_corrections": no_corrections,
        }
    }


@router.get("/documents/{document_id}/issues", response_model=DocumentOperationsResponse)
async def get_document_operational_issues(
    document_id: str,
    db: AsyncSession = Depends(get_db),
) -> DocumentOperationsResponse:
    """Derive all actionable operational issues, tri-layer evidence, and available actions for a document."""
    service = DocumentService(db)
    doc = await service.get_document_by_id(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    historical_summary = await _fetch_factual_historical_summary(db, doc)
    return OperationalIssuesService.derive_document_issues(doc, historical_summary=historical_summary)


@router.get("/documents/{document_id}/actions", response_model=List[OperationalAction])
async def get_document_operational_actions(
    document_id: str,
    db: AsyncSession = Depends(get_db),
) -> List[OperationalAction]:
    """Fetch aggregated unique declarative actions available for this document across all its issues."""
    service = DocumentService(db)
    doc = await service.get_document_by_id(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    resp = OperationalIssuesService.derive_document_issues(doc)

    # De-duplicate actions by action id
    seen = set()
    unique_actions: List[OperationalAction] = []
    for issue in resp.issues:
        for act in issue.available_actions:
            if act.id not in seen:
                seen.add(act.id)
                unique_actions.append(act)

    return unique_actions


@router.get("/backlog", response_model=OperationsBacklogSummary)
async def get_operations_backlog(
    db: AsyncSession = Depends(get_db),
) -> OperationsBacklogSummary:
    """Retrieve live operational backlog aggregation grouped by reason and category."""
    return await OperationalIssuesService.get_backlog_summary(db)
