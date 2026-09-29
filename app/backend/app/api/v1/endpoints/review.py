"""Review and operator interventions API router."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.document import DocumentDetailResponse
from app.schemas.kie import NormalizedFieldDTO
from app.schemas.review import (
    ApproveDocumentRequest,
    FieldCorrectionRequest,
    RejectDocumentRequest,
    ReviewResultResponse,
)
from app.services.document_service import DocumentService
from app.services.review_service import ReviewService

router = APIRouter(prefix="/documents", tags=["Manual Review"])


@router.patch("/{document_id}/fields/{field_name}", response_model=NormalizedFieldDTO)
async def correct_field(
    document_id: str,
    field_name: str,
    req: FieldCorrectionRequest,
    db: AsyncSession = Depends(get_db),
) -> NormalizedFieldDTO:
    """Operator updates the extracted value of a field with full audit logging and re-validation."""
    service = ReviewService(db)
    try:
        updated = await service.correct_field(
            document_id=document_id,
            field_name=field_name,
            corrected_value=req.corrected_value,
            reviewer_name=req.reviewer_name,
            notes=req.notes,
        )
        return NormalizedFieldDTO(
            field_name=updated.field_name,
            value=updated.corrected_value or updated.value,
            original_value=updated.original_value or updated.value,
            corrected_value=updated.corrected_value,
            is_corrected=bool(updated.corrected_value),
            confidence=round(updated.confidence, 4),
            bbox=updated.bbox_json,
            source_token_indices=updated.source_tokens_json or [],
            validation_status=updated.validation_status,
        )
    except ValueError as e:
        err_msg = str(e)
        if "not found" in err_msg.lower():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=err_msg)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=err_msg)


@router.post("/{document_id}/review/complete", response_model=ReviewResultResponse)
async def complete_review(
    document_id: str,
    req: ApproveDocumentRequest,
    db: AsyncSession = Depends(get_db),
) -> ReviewResultResponse:
    """Operator completes review; gated by validation checks before final sign-off."""
    service = ReviewService(db)
    try:
        doc = await service.complete_review(
            document_id=document_id,
            reviewer_name=req.reviewer_name,
            notes=req.notes,
        )
        return ReviewResultResponse(
            document_id=doc.id,
            status=doc.status,
            decision=doc.decision or "manual_review",
            message="Manual review successfully completed and verified",
        )
    except ValueError as e:
        err_msg = str(e)
        if "not found" in err_msg.lower():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=err_msg)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=err_msg)


@router.post("/{document_id}/approve", response_model=ReviewResultResponse)
async def approve_document(
    document_id: str,
    req: ApproveDocumentRequest,
    db: AsyncSession = Depends(get_db),
) -> ReviewResultResponse:
    """Operator signs off and approves document processing (alias for complete_review)."""
    return await complete_review(document_id, req, db)


@router.post("/{document_id}/reject", response_model=ReviewResultResponse)
async def reject_document(
    document_id: str,
    req: RejectDocumentRequest,
    db: AsyncSession = Depends(get_db),
) -> ReviewResultResponse:
    """Operator rejects document with explicit reason."""
    service = ReviewService(db)
    try:
        doc = await service.reject_document(
            document_id=document_id,
            reason=req.reason,
            reviewer_name=req.reviewer_name,
            notes=req.notes,
        )
        return ReviewResultResponse(
            document_id=doc.id,
            status=doc.status,
            decision=doc.decision or "rejected",
            message=f"Document rejected: {req.reason}",
        )
    except ValueError as e:
        err_msg = str(e)
        if "not found" in err_msg.lower():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=err_msg)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=err_msg)
