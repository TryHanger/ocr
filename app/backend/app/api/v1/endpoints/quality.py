"""Document Quality API router."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.quality import QualityResultDTO
from app.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["Quality"])


@router.get("/{document_id}/quality", response_model=QualityResultDTO)
async def get_document_quality(
    document_id: str,
    db: AsyncSession = Depends(get_db),
) -> QualityResultDTO:
    """Retrieve detailed scan quality metrics for a document."""
    service = DocumentService(db)
    doc = await service.get_document_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    if not doc.quality:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Quality analysis not yet available for this document",
        )

    return QualityResultDTO(
        document_id=doc.id,
        blur_score=doc.quality.blur_score,
        contrast_score=doc.quality.contrast_score,
        noise_score=doc.quality.noise_score,
        rotation_angle=doc.quality.rotation_angle,
        resolution_dpi=doc.quality.resolution_dpi,
        quality_score=doc.quality.quality_score,
        profile_summary=doc.quality.profile_summary,
        metrics=doc.quality.metrics_json or {},
    )
