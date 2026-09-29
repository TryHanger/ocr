"""Feedback Analytics Service for calculating factual review, correction, and bucket metrics."""

from datetime import datetime
import json
from typing import Any, Dict, List, Optional, Set, Tuple
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.enums import AuditAction, AutomationDecision, DocumentStatus, ReviewReason
from app.core.improvement_constants import CONFIDENCE_BUCKETS
from app.models.audit import AuditEvent
from app.models.document import Document
from app.models.field import ExtractedField
from app.schemas.analytics import CorrectionDocumentTypeMetric, CorrectionReasonMetric
from app.schemas.improvement import (
    ConfidenceCorrectionBucket,
    FieldCorrectionMetric,
    ImprovementOverviewResponse,
    ImprovementOverviewSummary,
)
from app.services.analytics_service import AnalyticsService, _safe_rate


class FeedbackAnalyticsService:
    """Computes descriptive aggregations on production reviews, corrections, and confidence intervals."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_feedback_overview(
        self,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        period: Optional[str] = "all",
    ) -> ImprovementOverviewResponse:
        """Compute consolidated improvement loop metrics with strict denominators and dual metrics."""
        start, end = AnalyticsService._parse_date_bounds(date_from, date_to, period)

        # 1. Fetch relevant processed / reviewed documents
        proc_stmt = (
            select(Document)
            .where(
                Document.status.in_([
                    DocumentStatus.COMPLETED_AUTOMATIC.value,
                    DocumentStatus.COMPLETED_MANUAL.value,
                    DocumentStatus.MANUAL_REVIEW.value,
                    DocumentStatus.ERROR.value,
                ])
            )
            .options(selectinload(Document.fields))
        )
        if start:
            proc_stmt = proc_stmt.where(
                (Document.processing_finished_at >= start) | (Document.created_at >= start)
            )
        if end:
            proc_stmt = proc_stmt.where(
                (Document.processing_finished_at <= end) | (Document.created_at <= end)
            )

        proc_docs = (await self.session.execute(proc_stmt)).scalars().all()
        doc_map: Dict[str, Document] = {d.id: d for d in proc_docs}

        # Reviewed documents: documents that reached COMPLETED_MANUAL
        completed_manual_docs = [
            d for d in proc_docs if d.status == DocumentStatus.COMPLETED_MANUAL.value
        ]

        # 2. Fetch FIELD_CORRECTED audit events in period
        corr_stmt = select(AuditEvent).where(
            AuditEvent.action == AuditAction.FIELD_CORRECTED.value
        )
        if start:
            corr_stmt = corr_stmt.where(AuditEvent.created_at >= start)
        if end:
            corr_stmt = corr_stmt.where(AuditEvent.created_at <= end)

        corr_events = (await self.session.execute(corr_stmt)).scalars().all()

        # Query documents for any corrected document_ids that might not have been in proc_docs
        missing_doc_ids = [ev.document_id for ev in corr_events if ev.document_id not in doc_map]
        if missing_doc_ids:
            miss_stmt = (
                select(Document)
                .where(Document.id.in_(missing_doc_ids))
                .options(selectinload(Document.fields))
            )
            miss_docs = (await self.session.execute(miss_stmt)).scalars().all()
            for md in miss_docs:
                doc_map[md.id] = md
                if md.status == DocumentStatus.COMPLETED_MANUAL.value and md not in completed_manual_docs:
                    completed_manual_docs.append(md)

        # Map corrections by doc_id and field_name
        corr_by_doc: Dict[str, List[AuditEvent]] = {}
        corrected_field_keys: Set[Tuple[str, str]] = set()  # (doc_id, field_name)
        for ev in corr_events:
            corr_by_doc.setdefault(ev.document_id, []).append(ev)
            if ev.field_name:
                corrected_field_keys.add((ev.document_id, ev.field_name))

        # Distinct documents with corrections
        corrected_doc_ids = set(corr_by_doc.keys())
        documents_processed = len(doc_map)
        documents_reviewed = len(completed_manual_docs)
        documents_corrected = len(corrected_doc_ids.intersection(set(doc_map.keys())))
        total_field_corrections = len(corr_events)

        effective_reviewed = max(documents_reviewed, documents_corrected)
        without_corrections = max(0, effective_reviewed - documents_corrected)

        summary = ImprovementOverviewSummary(
            documents_processed=documents_processed,
            documents_reviewed=effective_reviewed,
            documents_corrected=documents_corrected,
            total_field_corrections=total_field_corrections,
            correction_rate=_safe_rate(documents_corrected, effective_reviewed),
            no_change_review_rate=_safe_rate(without_corrections, effective_reviewed),
        )

        # 3. By Field (Dual Metrics: Correction Rate + Share of Total Corrections)
        # Evaluated fields count per field name (denominator: fields with confidence is not None)
        field_eval_counts: Dict[str, int] = {}
        for d in doc_map.values():
            for f in d.fields:
                if f.confidence is not None:
                    field_eval_counts[f.field_name] = field_eval_counts.get(f.field_name, 0) + 1

        field_event_counts: Dict[str, int] = {}
        field_affected_docs: Dict[str, Set[str]] = {}
        for ev in corr_events:
            fn = ev.field_name or "unknown"
            field_event_counts[fn] = field_event_counts.get(fn, 0) + 1
            field_affected_docs.setdefault(fn, set()).add(ev.document_id)

        by_field: List[FieldCorrectionMetric] = []
        for fn, corr_cnt in sorted(field_event_counts.items(), key=lambda x: (-x[1], x[0])):
            eval_cnt = max(field_eval_counts.get(fn, 0), corr_cnt)
            by_field.append(
                FieldCorrectionMetric(
                    field=fn,
                    correction_count=corr_cnt,
                    evaluated_count=eval_cnt,
                    correction_rate=_safe_rate(corr_cnt, eval_cnt),
                    share_of_all_corrections=_safe_rate(corr_cnt, total_field_corrections),
                    affected_documents=len(field_affected_docs.get(fn, set())),
                )
            )

        # 4. By Confidence Buckets (Strict evaluated field denominator)
        bucket_data = {
            label: {"evaluated": 0, "corrected": 0}
            for label, _, _ in CONFIDENCE_BUCKETS
        }

        total_corrected_fields_in_buckets = 0
        for d in doc_map.values():
            for f in d.fields:
                if f.confidence is None:
                    continue
                c = float(f.confidence)
                is_corrected = (d.id, f.field_name) in corrected_field_keys or (
                    f.corrected_value is not None and f.corrected_value != f.value
                )

                for label, low, high in CONFIDENCE_BUCKETS:
                    if low <= c < high or (high >= 1.0 and c >= 1.0 and label == "0.95–1.00"):
                        bucket_data[label]["evaluated"] += 1
                        if is_corrected:
                            bucket_data[label]["corrected"] += 1
                            total_corrected_fields_in_buckets += 1
                        break

        # If audit events exist but fields were deleted or had no confidence, fallback to match event counts
        if total_field_corrections > 0 and total_corrected_fields_in_buckets == 0:
            total_corrected_fields_in_buckets = total_field_corrections

        by_confidence: List[ConfidenceCorrectionBucket] = []
        for label, low, high in CONFIDENCE_BUCKETS:
            eval_fields = bucket_data[label]["evaluated"]
            corr_fields = bucket_data[label]["corrected"]
            by_confidence.append(
                ConfidenceCorrectionBucket(
                    bucket=label,
                    evaluated_fields=eval_fields,
                    corrected_fields=corr_fields,
                    correction_rate=_safe_rate(corr_fields, eval_fields),
                    share_of_corrections=_safe_rate(corr_fields, total_corrected_fields_in_buckets),
                )
            )

        # 5. Resolve Triggering Review Reasons from DECISION_EVALUATED
        all_doc_ids = set(doc_map.keys())
        doc_reasons: Dict[str, str] = {}
        if all_doc_ids:
            eval_stmt = select(AuditEvent).where(
                AuditEvent.document_id.in_(all_doc_ids),
                AuditEvent.action == AuditAction.DECISION_EVALUATED.value,
            )
            eval_events = (await self.session.execute(eval_stmt)).scalars().all()
            for ev in eval_events:
                meta = ev.metadata_json or {}
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except Exception:
                        meta = {}
                r = meta.get("reason")
                if r and r != ReviewReason.NONE.value:
                    doc_reasons[ev.document_id] = r

        for d_id, d in doc_map.items():
            if d_id not in doc_reasons and d.review_reason and d.review_reason != ReviewReason.NONE.value:
                doc_reasons[d_id] = d.review_reason
            elif d_id not in doc_reasons:
                doc_reasons[d_id] = "manual_review"

        # 6. Breakdown by Review Reason
        reason_reviewed_docs: Dict[str, Set[str]] = {}
        reason_corrected_docs: Dict[str, Set[str]] = {}
        reason_events_count: Dict[str, int] = {}

        for d_id in completed_manual_docs:
            r = doc_reasons.get(d_id.id, "manual_review")
            reason_reviewed_docs.setdefault(r, set()).add(d_id.id)

        for d_id, evs in corr_by_doc.items():
            r = doc_reasons.get(d_id, "manual_review")
            reason_corrected_docs.setdefault(r, set()).add(d_id)
            reason_events_count[r] = reason_events_count.get(r, 0) + len(evs)
            if d_id in doc_map:
                reason_reviewed_docs.setdefault(r, set()).add(d_id)

        by_reason = [
            CorrectionReasonMetric(
                reason=r,
                reviewed_documents=len(reason_reviewed_docs.get(r, set())),
                documents_with_corrections=len(reason_corrected_docs.get(r, set())),
                correction_events=reason_events_count.get(r, 0),
                correction_rate=_safe_rate(
                    len(reason_corrected_docs.get(r, set())),
                    len(reason_reviewed_docs.get(r, set())),
                ),
            )
            for r in sorted(
                reason_reviewed_docs.keys(),
                key=lambda r: (-reason_events_count.get(r, 0), -len(reason_reviewed_docs.get(r, set()))),
            )
        ]

        # 7. Breakdown by Document Type
        type_reviewed_docs: Dict[str, Set[str]] = {}
        type_corrected_docs: Dict[str, Set[str]] = {}
        type_events_count: Dict[str, int] = {}

        for d in completed_manual_docs:
            dt = d.document_type or "unknown"
            type_reviewed_docs.setdefault(dt, set()).add(d.id)

        for d_id, evs in corr_by_doc.items():
            doc = doc_map.get(d_id)
            dt = doc.document_type if doc else "unknown"
            type_corrected_docs.setdefault(dt, set()).add(d_id)
            type_events_count[dt] = type_events_count.get(dt, 0) + len(evs)
            if d_id in doc_map:
                type_reviewed_docs.setdefault(dt, set()).add(d_id)

        by_document_type = [
            CorrectionDocumentTypeMetric(
                document_type=dt,
                reviewed_documents=len(type_reviewed_docs.get(dt, set())),
                documents_with_corrections=len(type_corrected_docs.get(dt, set())),
                correction_events=type_events_count.get(dt, 0),
                correction_rate=_safe_rate(
                    len(type_corrected_docs.get(dt, set())),
                    len(type_reviewed_docs.get(dt, set())),
                ),
            )
            for dt in sorted(
                type_reviewed_docs.keys(),
                key=lambda dt: (-type_events_count.get(dt, 0), -len(type_reviewed_docs.get(dt, set()))),
            )
        ]

        return ImprovementOverviewResponse(
            period=period or "all",
            date_from=start,
            date_to=end,
            summary=summary,
            by_field=by_field,
            by_confidence=by_confidence,
            by_reason=by_reason,
            by_document_type=by_document_type,
            active_signals_count=0,
            open_questions_count=0,
        )
