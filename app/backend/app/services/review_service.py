"""Review Service for human-in-the-loop corrections and approvals."""

from datetime import datetime
from typing import Dict, Optional
import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.enums import AuditAction, AutomationDecision, DocumentStatus, ReviewReason
from app.models.audit import AuditEvent
from app.models.document import Document
from app.models.field import ExtractedField
from app.models.review import ReviewAction
from app.services.document_validator import DocumentValidator


class ReviewService:
    """Manages operator corrections, manual approvals, and audit trail generation."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def correct_field(
        self,
        document_id: str,
        field_name: str,
        corrected_value: str,
        reviewer_name: str = "operator",
        notes: Optional[str] = None,
    ) -> ExtractedField:
        """Apply an operator correction to an extracted field with audit logging and re-validation."""
        stmt = (
            select(Document)
            .where(Document.id == document_id)
            .options(selectinload(Document.fields))
        )
        res = await self.session.execute(stmt)
        doc = res.scalar_one_or_none()

        if not doc:
            raise ValueError(f"Document {document_id} not found")

        target_field = next((f for f in doc.fields if f.field_name == field_name), None)
        if not target_field:
            target_field = ExtractedField(
                id=str(uuid.uuid4()),
                document_id=document_id,
                field_name=field_name,
                value=corrected_value,
                corrected_value=corrected_value,
                confidence=1.0,
                validation_status="VALID",
            )
            self.session.add(target_field)
            doc.fields.append(target_field)
            old_val = None
        else:
            old_val = target_field.corrected_value if target_field.corrected_value is not None else target_field.value
            target_field.corrected_value = corrected_value
            target_field.updated_at = datetime.utcnow()

        # Append-only audit record
        audit = AuditEvent(
            id=str(uuid.uuid4()),
            document_id=document_id,
            actor=f"operator:{reviewer_name}",
            action=AuditAction.FIELD_CORRECTED.value,
            field_name=field_name,
            old_value=old_val,
            new_value=corrected_value,
            metadata_json={"notes": notes},
        )
        self.session.add(audit)

        # Log review action
        review = ReviewAction(
            id=str(uuid.uuid4()),
            document_id=document_id,
            reviewer_name=reviewer_name,
            action="CORRECTED",
            reason=ReviewReason.NONE.value,
            notes=f"Updated field '{field_name}' to '{corrected_value}'. {notes or ''}".strip(),
        )
        self.session.add(review)

        # Re-validate all active field values on this document
        fields_map: Dict[str, str] = {
            f.field_name: f.corrected_value if f.corrected_value is not None else f.value
            for f in doc.fields
        }
        val_result = DocumentValidator.validate(doc.document_type, fields_map)

        # Update validation status per field
        for f in doc.fields:
            if f.field_name in val_result.field_statuses:
                f.validation_status = val_result.field_statuses[f.field_name]

        # Update document diagnostics without closing the review
        doc.review_details_json = val_result.review_details
        if val_result.is_valid:
            doc.review_reason = ReviewReason.NONE.value
        else:
            doc.review_reason = val_result.failed_reason or ReviewReason.VALIDATION_FAILED.value

        if not doc.review_started_at:
            doc.review_started_at = datetime.utcnow()

        doc.updated_at = datetime.utcnow()

        await self.session.commit()
        self.session.expire_all()
        await self.session.refresh(target_field)
        return target_field

    async def complete_review(
        self,
        document_id: str,
        reviewer_name: str = "operator",
        notes: Optional[str] = None,
    ) -> Document:
        """Perform validation gate and complete human review, transitioning to COMPLETED_MANUAL."""
        stmt = (
            select(Document)
            .where(Document.id == document_id)
            .options(selectinload(Document.fields))
        )
        res = await self.session.execute(stmt)
        doc = res.scalar_one_or_none()

        if not doc:
            raise ValueError(f"Document {document_id} not found")

        if doc.status != DocumentStatus.MANUAL_REVIEW.value:
            raise ValueError(
                f"Document {document_id} is in status '{doc.status}', not in '{DocumentStatus.MANUAL_REVIEW.value}'."
            )

        fields_map: Dict[str, str] = {
            f.field_name: f.corrected_value if f.corrected_value is not None else f.value
            for f in doc.fields
        }
        val_result = DocumentValidator.validate(doc.document_type, fields_map)

        # Update validation status on fields
        for f in doc.fields:
            if f.field_name in val_result.field_statuses:
                f.validation_status = val_result.field_statuses[f.field_name]

        if not val_result.is_valid:
            # Block completion, remain in MANUAL_REVIEW
            doc.review_reason = val_result.failed_reason or ReviewReason.VALIDATION_FAILED.value
            doc.review_details_json = val_result.review_details
            doc.review_finished_at = None
            doc.updated_at = datetime.utcnow()
            await self.session.commit()
            err_summary = "; ".join(e.message for e in val_result.errors)
            raise ValueError(f"Validation failed: {err_summary}")

        # Validation passed: finalize manual review
        now_finished = datetime.utcnow()
        doc.status = DocumentStatus.COMPLETED_MANUAL.value
        doc.decision = AutomationDecision.MANUAL_REVIEW.value
        doc.review_reason = ReviewReason.NONE.value
        doc.review_details_json = None
        doc.review_finished_at = now_finished

        if not doc.review_started_at:
            doc.review_started_at = doc.processing_finished_at or doc.created_at

        doc.review_duration_ms = round(
            (now_finished - doc.review_started_at).total_seconds() * 1000.0, 2
        )
        doc.updated_at = now_finished

        review = ReviewAction(
            id=str(uuid.uuid4()),
            document_id=document_id,
            reviewer_name=reviewer_name,
            action="APPROVED",
            reason=ReviewReason.NONE.value,
            notes=notes,
        )
        self.session.add(review)

        audit = AuditEvent(
            id=str(uuid.uuid4()),
            document_id=document_id,
            actor=f"operator:{reviewer_name}",
            action=AuditAction.REVIEW_COMPLETED.value,
            metadata_json={
                "notes": notes,
                "review_duration_ms": doc.review_duration_ms,
            },
        )
        self.session.add(audit)

        await self.session.commit()
        await self.session.refresh(doc)
        return doc

    async def approve_document(
        self,
        document_id: str,
        reviewer_name: str = "operator",
        notes: Optional[str] = None,
    ) -> Document:
        """Operator explicitly approves document (delegates to complete_review)."""
        return await self.complete_review(document_id, reviewer_name, notes)

    async def reject_document(
        self,
        document_id: str,
        reason: str,
        reviewer_name: str = "operator",
        notes: Optional[str] = None,
    ) -> Document:
        """Operator explicitly rejects document."""
        stmt = select(Document).where(Document.id == document_id)
        res = await self.session.execute(stmt)
        doc = res.scalar_one_or_none()

        if not doc:
            raise ValueError(f"Document {document_id} not found")

        now_finished = datetime.utcnow()
        doc.status = DocumentStatus.ERROR.value
        doc.decision = AutomationDecision.REJECTED.value
        doc.review_reason = reason
        doc.review_finished_at = now_finished
        if doc.review_started_at:
            doc.review_duration_ms = round(
                (now_finished - doc.review_started_at).total_seconds() * 1000.0, 2
            )
        doc.updated_at = now_finished

        review = ReviewAction(
            id=str(uuid.uuid4()),
            document_id=document_id,
            reviewer_name=reviewer_name,
            action="REJECTED",
            reason=reason,
            notes=notes,
        )
        self.session.add(review)

        audit = AuditEvent(
            id=str(uuid.uuid4()),
            document_id=document_id,
            actor=f"operator:{reviewer_name}",
            action=AuditAction.DOCUMENT_REJECTED.value,
            metadata_json={"reason": reason, "notes": notes},
        )
        self.session.add(audit)

        await self.session.commit()
        await self.session.refresh(doc)
        return doc

