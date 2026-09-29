"""Document Service for intake, storage, page setup, and queries."""

from datetime import datetime
import io
from pathlib import Path
from typing import List, Optional, Tuple
import uuid
from PIL import Image
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.core.enums import AuditAction, DocumentStatus, DocumentType, JobStatus, ReviewReason
from app.core.security import sanitize_filename, validate_upload
from app.models.audit import AuditEvent
from app.models.document import Document
from app.models.job import ProcessingJob
from app.models.page import DocumentPage
from app.services.storage.local import default_storage


class DocumentService:
    """Manages document lifecycle from intake to persistence and queries."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def ingest_document(
        self,
        file_bytes: bytes,
        original_filename: str,
        declared_mime: str,
        document_type: str = DocumentType.RECEIPT.value,
    ) -> Document:
        """Ingest new document with security validation, storage persistence, and job creation."""
        # 1. Validate file
        is_valid, actual_mime, error_msg = validate_upload(
            header_bytes=file_bytes[:32],
            declared_mime=declared_mime,
            file_size=len(file_bytes),
            max_size=settings.MAX_UPLOAD_SIZE_BYTES,
        )
        if not is_valid:
            raise ValueError(f"Document upload rejected: {error_msg}")

        # 2. Sanitize filename and prepare path
        doc_id = str(uuid.uuid4())
        safe_name = sanitize_filename(original_filename)
        now = datetime.utcnow()
        storage_rel_path = f"{now.strftime('%Y/%m')}/{doc_id}/{safe_name}"

        # 3. Save to storage
        saved_path = await default_storage.save(file_bytes, storage_rel_path)

        # 4. Determine dimensions
        width, height = 800, 1000
        try:
            with Image.open(io.BytesIO(file_bytes)) as pil_img:
                width, height = pil_img.size
        except Exception:
            pass

        # 5. Create Document record
        doc = Document(
            id=doc_id,
            filename=safe_name,
            file_path=saved_path,
            file_size_bytes=len(file_bytes),
            mime_type=actual_mime,
            document_type=document_type,
            status=DocumentStatus.UPLOADED.value,
            review_reason=ReviewReason.NONE.value,
        )
        self.session.add(doc)

        # 6. Create initial Page record (Page 1)
        page = DocumentPage(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            page_number=1,
            width=width,
            height=height,
            image_path=saved_path,
        )
        self.session.add(page)

        # 7. Create background ProcessingJob
        job = ProcessingJob(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            status=JobStatus.PENDING.value,
            current_stage=DocumentStatus.UPLOADED.value,
            attempt=1,
            max_attempts=settings.MAX_JOB_ATTEMPTS,
        )
        self.session.add(job)

        # 8. Record audit event
        audit = AuditEvent(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            actor="system",
            action=AuditAction.DOCUMENT_UPLOADED.value,
            metadata_json={"filename": safe_name, "size_bytes": len(file_bytes), "mime": actual_mime},
        )
        self.session.add(audit)

        await self.session.commit()
        await self.session.refresh(doc)
        return doc

    async def get_document_by_id(self, document_id: str) -> Optional[Document]:
        """Fetch document with all related pages, fields, quality, reviews, and audit events."""
        stmt = (
            select(Document)
            .where(Document.id == document_id)
            .execution_options(populate_existing=True)
            .options(
                selectinload(Document.pages),
                selectinload(Document.fields),
                selectinload(Document.quality),
                selectinload(Document.reviews),
                selectinload(Document.audit_events),
                selectinload(Document.jobs),
                selectinload(Document.steps),
            )
        )
        res = await self.session.execute(stmt)
        return res.scalar_one_or_none()

    async def list_documents(
        self,
        status: Optional[str] = None,
        decision: Optional[str] = None,
        document_type: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[int, List[Document]]:
        """Paginated list of documents with optional status/search filters."""
        query = select(Document)

        if status:
            query = query.where(Document.status == status)
        if decision:
            query = query.where(Document.decision == decision)
        if document_type:
            query = query.where(Document.document_type == document_type)
        if search:
            query = query.where(Document.filename.ilike(f"%{search}%"))

        count_query = select(func.count()).select_from(query.subquery())
        total = (await self.session.execute(count_query)).scalar() or 0

        query = query.order_by(desc(Document.created_at)).limit(limit).offset(offset)
        result = await self.session.execute(query)
        return total, list(result.scalars().all())

    async def create_retry_job(self, document_id: str) -> Optional[ProcessingJob]:
        """Create a new retry job for a document in error or manual review status."""
        doc = await self.get_document_by_id(document_id)
        if not doc:
            return None

        # Reset document status
        doc.status = DocumentStatus.PROCESSING.value
        doc.updated_at = datetime.utcnow()

        new_job = ProcessingJob(
            id=str(uuid.uuid4()),
            document_id=document_id,
            status=JobStatus.PENDING.value,
            current_stage=DocumentStatus.PROCESSING.value,
            attempt=1,
            max_attempts=settings.MAX_JOB_ATTEMPTS,
        )
        self.session.add(new_job)

        audit = AuditEvent(
            id=str(uuid.uuid4()),
            document_id=document_id,
            actor="operator",
            action=AuditAction.JOB_RETRIED.value,
            metadata_json={"retry_job_id": new_job.id},
        )
        self.session.add(audit)

        await self.session.commit()
        await self.session.refresh(new_job)
        return new_job
