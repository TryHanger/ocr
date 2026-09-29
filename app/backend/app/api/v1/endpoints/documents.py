"""Documents API v1 router."""

import io
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.enums import AuditAction, DocumentType
from app.db.session import get_db
from app.models.document import Document
from app.schemas.document import (
    AuditEventDTO,
    DocumentDetailResponse,
    DocumentFieldCorrectionDTO,
    DocumentListResponse,
    DocumentPageDTO,
    DocumentResponse,
)
from app.schemas.kie import NormalizedFieldDTO
from app.schemas.ocr import NormalizedTokenDTO
from app.schemas.quality import QualityResultDTO
from app.services.document_service import DocumentService
from app.services.explainability_service import ExplainabilityService
from app.services.quality_bridge import default_evidence_bridge
from app.services.storage.local import default_storage

router = APIRouter(prefix="/documents", tags=["Documents"])


def _to_document_response(doc: Document) -> DocumentResponse:
    return DocumentResponse(
        id=doc.id,
        filename=doc.filename,
        document_type=doc.document_type,
        status=doc.status,
        decision=doc.decision,
        confidence=round(doc.confidence * 100.0, 1) if doc.confidence is not None else None,
        review_reason=doc.review_reason,
        review_details=doc.review_details_json,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
        processing_started_at=doc.processing_started_at,
        processing_finished_at=doc.processing_finished_at,
        processing_duration_ms=doc.processing_duration_ms,
        review_started_at=doc.review_started_at,
        review_finished_at=doc.review_finished_at,
        review_duration_ms=doc.review_duration_ms,
        review_wait_duration_ms=doc.review_wait_duration_ms,
    )


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    document_type: str = Form(DocumentType.RECEIPT.value),
    db: AsyncSession = Depends(get_db),
) -> DocumentResponse:
    """Upload a new document for automated processing and quality analysis."""
    file_bytes = await file.read()
    service = DocumentService(db)

    try:
        doc = await service.ingest_document(
            file_bytes=file_bytes,
            original_filename=file.filename or "document.jpg",
            declared_mime=file.content_type or "",
            document_type=document_type,
        )
        return _to_document_response(doc)
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    status: Optional[str] = Query(None),
    decision: Optional[str] = Query(None),
    document_type: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> DocumentListResponse:
    """List documents with pagination and status/decision filtering."""
    service = DocumentService(db)
    total, items = await service.list_documents(
        status=status,
        decision=decision,
        document_type=document_type,
        search=search,
        limit=limit,
        offset=offset,
    )
    return DocumentListResponse(
        total=total,
        items=[_to_document_response(d) for d in items],
    )


@router.get("/{document_id}", response_model=DocumentDetailResponse)
async def get_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
) -> DocumentDetailResponse:
    """Fetch complete document details with pages, extracted fields, quality, and audit log."""
    service = DocumentService(db)
    doc = await service.get_document_by_id(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    pages_dto = []
    for p in doc.pages:
        tokens_parsed = []
        if p.tokens_json:
            for t in p.tokens_json:
                tokens_parsed.append(NormalizedTokenDTO(**t))

        pages_dto.append(
            DocumentPageDTO(
                id=p.id,
                page_number=p.page_number,
                width=p.width,
                height=p.height,
                image_url=f"/api/v1/documents/{doc.id}/pages/{p.page_number}/image",
                full_text=p.full_text,
                tokens=tokens_parsed,
            )
        )

    fields_dto = {}
    for f in doc.fields:
        active_val = f.corrected_value if f.corrected_value is not None else f.value
        fields_dto[f.field_name] = NormalizedFieldDTO(
            field_name=f.field_name,
            value=active_val,
            original_value=f.value,
            corrected_value=f.corrected_value,
            is_corrected=f.corrected_value is not None,
            confidence=round(f.confidence, 4),
            bbox=f.bbox_json,
            source_token_indices=f.source_tokens_json or [],
            validation_status=f.validation_status,
        )

    quality_dto = None
    if doc.quality:
        quality_dto = QualityResultDTO(
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

    audit_dto = [
        AuditEventDTO(
            id=a.id,
            actor=a.actor,
            action=a.action,
            field_name=a.field_name,
            old_value=a.old_value,
            new_value=a.new_value,
            created_at=a.created_at,
            metadata=a.metadata_json or {},
        )
        for a in doc.audit_events
    ]

    from app.schemas.step import ProcessingStepDTO
    steps_dto = [
        ProcessingStepDTO(
            id=s.id,
            step_order=s.step_order,
            operation=s.operation,
            parameters=s.parameters_json,
            duration_ms=s.duration_ms,
            status=s.status,
            created_at=s.created_at,
        )
        for s in doc.steps
    ]

    base_resp = _to_document_response(doc)
    research_ev = default_evidence_bridge.get_evidence_for_quality(doc.quality)
    explanation = ExplainabilityService.explain_document(doc)

    field_corrections_dto = [
        DocumentFieldCorrectionDTO(
            field=ev.field_name or "unknown",
            previous_value=ev.old_value,
            final_value=ev.new_value,
            corrected_at=ev.created_at,
            actor=ev.actor,
        )
        for ev in (doc.audit_events or [])
        if ev.action == AuditAction.FIELD_CORRECTED.value
    ]

    return DocumentDetailResponse(
        **base_resp.model_dump(),
        file_size_bytes=doc.file_size_bytes,
        mime_type=doc.mime_type,
        file_url=f"/api/v1/documents/{doc.id}/file",
        pages=pages_dto,
        fields=fields_dto,
        quality=quality_dto,
        audit_events=audit_dto,
        steps=steps_dto,
        research_evidence=research_ev,
        decision_explanation=explanation,
        field_corrections=field_corrections_dto,
    )


@router.get("/{document_id}/file")
async def get_document_file(
    document_id: str,
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Stream document binary content."""
    service = DocumentService(db)
    doc = await service.get_document_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    try:
        content = await default_storage.get(doc.file_path)
        return Response(content=content, media_type=doc.mime_type)
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Stored file not found")


@router.get("/{document_id}/pages/{page_number}/image")
async def get_page_image(
    document_id: str,
    page_number: int,
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Stream page raster image."""
    service = DocumentService(db)
    doc = await service.get_document_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    target_page = next((p for p in doc.pages if p.page_number == page_number), None)
    if not target_page:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Page not found")

    try:
        content = await default_storage.get(target_page.image_path)
        return Response(content=content, media_type="image/jpeg")
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Stored page not found")


@router.post("/{document_id}/retry")
async def retry_document_processing(
    document_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Schedule a retry processing job for a failed or review-requested document."""
    service = DocumentService(db)
    job = await service.create_retry_job(document_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return {"message": "Retry job created successfully", "job_id": job.id}
