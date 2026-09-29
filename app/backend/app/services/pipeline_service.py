"""Pipeline Service implementing state-machine processing, idempotency, and provenance persistence."""

from datetime import datetime
import time
from typing import Optional
import uuid
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.adapters.base import KIEProvider, OCRProvider, PreprocessingProvider, QualityAnalyzer
from app.adapters.kie_adapter import default_kie_adapter
from app.adapters.ocr_adapter import default_ocr_adapter
from app.adapters.prep_adapter import default_prep_adapter
from app.adapters.quality_adapter import default_quality_adapter
from app.core.enums import AuditAction, AutomationDecision, DocumentStatus, JobStatus, ReviewReason
from app.core.logging import logger
from app.models.audit import AuditEvent
from app.models.document import Document
from app.models.field import ExtractedField
from app.models.job import ProcessingJob
from app.models.page import DocumentPage
from app.models.quality import DocumentQuality
from app.models.step import ProcessingStep
from app.services.confidence_engine import ConfidenceEngine
from app.services.decision_engine import DecisionEngine
from app.services.storage.local import default_storage


class PipelineService:
    """Orchestrates pipeline execution through distinct, idempotent state machine stages."""

    def __init__(
        self,
        session: AsyncSession,
        ocr_provider: OCRProvider = default_ocr_adapter,
        kie_provider: KIEProvider = default_kie_adapter,
        quality_analyzer: QualityAnalyzer = default_quality_adapter,
        prep_provider: PreprocessingProvider = default_prep_adapter,
    ) -> None:
        self.session = session
        self.ocr_provider = ocr_provider
        self.kie_provider = kie_provider
        self.quality_analyzer = quality_analyzer
        self.prep_provider = prep_provider

    async def execute_job(self, job_id: str) -> bool:
        """Execute a processing job through the entire state-machine pipeline."""
        start_time = time.perf_counter()

        # 1. Fetch Job and Document
        job_stmt = (
            select(ProcessingJob)
            .where(ProcessingJob.id == job_id)
            .options(selectinload(ProcessingJob.document).selectinload(Document.pages))
        )
        job_res = await self.session.execute(job_stmt)
        job = job_res.scalar_one_or_none()

        if not job or not job.document:
            logger.error(f"Job {job_id} or associated document not found")
            return False

        doc = job.document
        doc_id = doc.id

        job.status = JobStatus.RUNNING.value
        job.current_stage = DocumentStatus.PROCESSING.value
        job.started_at = datetime.utcnow()
        doc.processing_started_at = datetime.utcnow()
        doc.status = DocumentStatus.PROCESSING.value
        await self.session.commit()

        # Idempotency: clear previously persisted steps for this document on retry
        await self.session.execute(
            delete(ProcessingStep).where(ProcessingStep.document_id == doc_id)
        )
        await self.session.commit()

        current_step_order = 1

        try:
            # Load primary image bytes
            image_bytes = await default_storage.get(doc.file_path)

            # -------------------------------------------------------------
            # Stage 1: Quality Analyzer
            # -------------------------------------------------------------
            t_q0 = time.perf_counter()
            job.current_stage = DocumentStatus.QUALITY_ANALYZED.value
            quality_dto = await self.quality_analyzer.analyze(image_bytes, doc_id)
            t_q_duration = round((time.perf_counter() - t_q0) * 1000.0, 2)

            # Idempotently update or create DocumentQuality
            q_stmt = select(DocumentQuality).where(DocumentQuality.document_id == doc_id)
            existing_q = (await self.session.execute(q_stmt)).scalar_one_or_none()
            if existing_q:
                existing_q.blur_score = quality_dto.blur_score
                existing_q.contrast_score = quality_dto.contrast_score
                existing_q.noise_score = quality_dto.noise_score
                existing_q.rotation_angle = quality_dto.rotation_angle
                existing_q.resolution_dpi = quality_dto.resolution_dpi
                existing_q.quality_score = quality_dto.quality_score
                existing_q.profile_summary = quality_dto.profile_summary
                existing_q.metrics_json = quality_dto.metrics
            else:
                doc_q = DocumentQuality(
                    id=str(uuid.uuid4()),
                    document_id=doc_id,
                    blur_score=quality_dto.blur_score,
                    contrast_score=quality_dto.contrast_score,
                    noise_score=quality_dto.noise_score,
                    rotation_angle=quality_dto.rotation_angle,
                    resolution_dpi=quality_dto.resolution_dpi,
                    quality_score=quality_dto.quality_score,
                    profile_summary=quality_dto.profile_summary,
                    metrics_json=quality_dto.metrics,
                )
                self.session.add(doc_q)

            # Record Quality Analysis Provenance Step
            self.session.add(
                ProcessingStep(
                    id=str(uuid.uuid4()),
                    document_id=doc_id,
                    step_order=current_step_order,
                    operation="quality_analysis",
                    parameters_json={
                        "blur_score": quality_dto.blur_score,
                        "contrast_score": quality_dto.contrast_score,
                        "rotation_angle": quality_dto.rotation_angle,
                        "noise_score": quality_dto.noise_score,
                        "quality_score": quality_dto.quality_score,
                        "profile": quality_dto.profile_summary,
                    },
                    duration_ms=t_q_duration,
                    status="APPLIED",
                )
            )
            current_step_order += 1

            doc.status = DocumentStatus.QUALITY_ANALYZED.value
            self.session.add(
                AuditEvent(
                    id=str(uuid.uuid4()),
                    document_id=doc_id,
                    actor="system",
                    action=AuditAction.QUALITY_ANALYZED.value,
                    metadata_json={
                        "quality_score": quality_dto.quality_score,
                        "profile": quality_dto.profile_summary,
                    },
                )
            )
            await self.session.commit()

            # -------------------------------------------------------------
            # Stage 2: Smart Preprocessing
            # -------------------------------------------------------------
            job.current_stage = DocumentStatus.PREPROCESSED.value
            processed_bytes, applied_steps = await self.prep_provider.process(
                image_bytes, quality_dto
            )

            if applied_steps:
                for step_info in applied_steps:
                    self.session.add(
                        ProcessingStep(
                            id=str(uuid.uuid4()),
                            document_id=doc_id,
                            step_order=current_step_order,
                            operation=step_info["operation"],
                            parameters_json=step_info.get("parameters"),
                            duration_ms=step_info.get("duration_ms", 0.0),
                            status=step_info.get("status", "APPLIED"),
                        )
                    )
                    current_step_order += 1

                self.session.add(
                    AuditEvent(
                        id=str(uuid.uuid4()),
                        document_id=doc_id,
                        actor="system",
                        action=AuditAction.PREPROCESSING_APPLIED.value,
                        metadata_json={"steps": applied_steps},
                    )
                )
            else:
                self.session.add(
                    ProcessingStep(
                        id=str(uuid.uuid4()),
                        document_id=doc_id,
                        step_order=current_step_order,
                        operation="preprocessing_evaluation",
                        parameters_json={"decision": "no_op", "reason": "document_quality_sufficient"},
                        duration_ms=0.5,
                        status="SKIPPED",
                    )
                )
                current_step_order += 1

            doc.status = DocumentStatus.PREPROCESSED.value
            await self.session.commit()

            # -------------------------------------------------------------
            # Stage 3: OCR Engine
            # -------------------------------------------------------------
            job.current_stage = DocumentStatus.OCR_COMPLETED.value
            ocr_result = await self.ocr_provider.process(processed_bytes, doc_id)

            # Update DocumentPage with OCR results
            p_stmt = select(DocumentPage).where(DocumentPage.document_id == doc_id).limit(1)
            page = (await self.session.execute(p_stmt)).scalar_one_or_none()
            if page:
                page.full_text = ocr_result.full_text
                page.tokens_json = [t.model_dump() for t in ocr_result.tokens]
                meta_w = ocr_result.metadata.get("width")
                meta_h = ocr_result.metadata.get("height")
                if meta_w and meta_h:
                    page.width = meta_w
                    page.height = meta_h

            # Record OCR Step Provenance
            self.session.add(
                ProcessingStep(
                    id=str(uuid.uuid4()),
                    document_id=doc_id,
                    step_order=current_step_order,
                    operation="ocr_recognition",
                    parameters_json={
                        "model": ocr_result.model_name,
                        "tokens_count": len(ocr_result.tokens),
                    },
                    duration_ms=ocr_result.processing_time_ms,
                    status="APPLIED",
                )
            )
            current_step_order += 1

            doc.status = DocumentStatus.OCR_COMPLETED.value
            self.session.add(
                AuditEvent(
                    id=str(uuid.uuid4()),
                    document_id=doc_id,
                    actor="system",
                    action=AuditAction.OCR_COMPLETED.value,
                    metadata_json={
                        "tokens_count": len(ocr_result.tokens),
                        "model": ocr_result.model_name,
                        "duration_ms": ocr_result.processing_time_ms,
                    },
                )
            )
            await self.session.commit()

            # -------------------------------------------------------------
            # Stage 4: KIE Extraction
            # -------------------------------------------------------------
            job.current_stage = DocumentStatus.KIE_COMPLETED.value
            kie_result = await self.kie_provider.extract(ocr_result, doc_id)

            # Idempotency: delete previous extracted fields before inserting new ones
            await self.session.execute(
                delete(ExtractedField).where(ExtractedField.document_id == doc_id)
            )

            for field_name, f_dto in kie_result.fields.items():
                field_entity = ExtractedField(
                    id=str(uuid.uuid4()),
                    document_id=doc_id,
                    field_name=field_name,
                    value=f_dto.value,
                    confidence=f_dto.confidence,
                    validation_status=f_dto.validation_status,
                    bbox_json=f_dto.bbox,
                    source_tokens_json=f_dto.source_token_indices,
                )
                self.session.add(field_entity)

            # Record KIE Step Provenance
            self.session.add(
                ProcessingStep(
                    id=str(uuid.uuid4()),
                    document_id=doc_id,
                    step_order=current_step_order,
                    operation="kie_extraction",
                    parameters_json={
                        "model": kie_result.model_name,
                        "extracted_fields": list(kie_result.fields.keys()),
                    },
                    duration_ms=kie_result.processing_time_ms,
                    status="APPLIED",
                )
            )
            current_step_order += 1

            doc.status = DocumentStatus.KIE_COMPLETED.value
            self.session.add(
                AuditEvent(
                    id=str(uuid.uuid4()),
                    document_id=doc_id,
                    actor="system",
                    action=AuditAction.KIE_COMPLETED.value,
                    metadata_json={
                        "extracted_fields": list(kie_result.fields.keys()),
                        "model": kie_result.model_name,
                        "duration_ms": kie_result.processing_time_ms,
                    },
                )
            )
            await self.session.commit()

            # -------------------------------------------------------------
            # Stage 5: Confidence & Decision Engine
            # -------------------------------------------------------------
            t_dec0 = time.perf_counter()
            job.current_stage = DocumentStatus.DECISION_MADE.value

            # Calculate document-level confidence
            doc_confidence = ConfidenceEngine.calculate_document_confidence(
                fields=kie_result.fields,
                ocr_tokens=ocr_result.tokens,
                image_quality_score=quality_dto.quality_score,
            )

            # Evaluate automation policy
            decision, review_reason, explanation, review_details = DecisionEngine.evaluate(
                document_type=doc.document_type,
                fields=kie_result.fields,
                quality=quality_dto,
                document_confidence=doc_confidence,
            )
            t_dec_duration = round((time.perf_counter() - t_dec0) * 1000.0, 2)

            now_finished = datetime.utcnow()
            total_duration_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            doc.processing_finished_at = now_finished
            doc.processing_duration_ms = total_duration_ms
            doc.confidence = doc_confidence
            doc.decision = decision.value
            doc.review_reason = review_reason.value
            doc.review_details_json = review_details

            if decision == AutomationDecision.AUTOMATIC:
                doc.status = DocumentStatus.COMPLETED_AUTOMATIC.value
                doc.review_started_at = None
                doc.review_finished_at = None
                doc.review_duration_ms = None
            else:
                doc.status = DocumentStatus.MANUAL_REVIEW.value
                doc.review_started_at = now_finished
                doc.review_wait_duration_ms = 0.0

            # Record Decision Step Provenance
            self.session.add(
                ProcessingStep(
                    id=str(uuid.uuid4()),
                    document_id=doc_id,
                    step_order=current_step_order,
                    operation="decision_engine",
                    parameters_json={
                        "decision": decision.value,
                        "confidence": doc_confidence,
                        "reason": review_reason.value,
                        "explanation": explanation,
                    },
                    duration_ms=t_dec_duration,
                    status="APPLIED",
                )
            )

            # Finalize Job
            job.status = JobStatus.COMPLETED.value
            job.finished_at = datetime.utcnow()

            self.session.add(
                AuditEvent(
                    id=str(uuid.uuid4()),
                    document_id=doc_id,
                    actor="system",
                    action=AuditAction.DECISION_EVALUATED.value,
                    metadata_json={
                        "decision": decision.value,
                        "confidence": doc_confidence,
                        "reason": review_reason.value,
                        "explanation": explanation,
                        "review_details": review_details,
                        "total_duration_ms": total_duration_ms,
                    },
                )
            )
            await self.session.commit()
            logger.info(
                f"Successfully completed processing for document {doc_id}",
                extra={
                    "document_id": doc_id,
                    "job_id": job_id,
                    "decision": decision.value,
                    "confidence": doc_confidence,
                    "duration_ms": total_duration_ms,
                },
            )
            return True

        except Exception as exc:
            await self.session.rollback()
            logger.error(
                f"Pipeline execution failure on document {doc_id} at stage {job.current_stage}: {exc}",
                exc_info=True,
            )

            job_res = await self.session.execute(select(ProcessingJob).where(ProcessingJob.id == job_id))
            job = job_res.scalar_one_or_none()
            if job:
                job.error_message = str(exc)
                job.error_stage = job.current_stage

                if job.attempt < job.max_attempts:
                    job.status = JobStatus.RETRYING.value
                    job.attempt += 1
                else:
                    job.status = JobStatus.FAILED.value
                    job.finished_at = datetime.utcnow()

                    doc_res = await self.session.execute(select(Document).where(Document.id == doc_id))
                    doc = doc_res.scalar_one_or_none()
                    if doc:
                        doc.status = DocumentStatus.ERROR.value
                        doc.review_reason = ReviewReason.SYSTEM_ERROR.value

                    self.session.add(
                        AuditEvent(
                            id=str(uuid.uuid4()),
                            document_id=doc_id,
                            actor="system",
                            action=AuditAction.PROCESSING_FAILED.value,
                            metadata_json={"error": str(exc), "stage": job.error_stage},
                        )
                    )
                await self.session.commit()
            return False
