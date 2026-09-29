"""Analytics Service for calculating Control Center KPIs, telemetry, and aggregations."""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.enums import AuditAction, AutomationDecision, DocumentStatus, JobStatus, ReviewReason
from app.models.audit import AuditEvent
from app.models.document import Document
from app.models.field import ExtractedField
from app.models.job import ProcessingJob
from app.models.page import DocumentPage
from app.models.quality import DocumentQuality
from app.schemas.document import DocumentResponse
from app.services.explainability_service import ExplainabilityService
from app.schemas.analytics import (
    AnalyticsOverviewResponse,
    AutomationTrendItem,
    AutomationTrendsResponse,
    BreakdownItem,
    ConfidenceAnalyticsResponse,
    ConfidenceBucket,
    ControlCenterPeriod,
    ControlCenterQuality,
    ControlCenterResponse,
    ControlCenterSummary,
    CorrectionDocumentTypeMetric,
    CorrectionFieldMetric,
    CorrectionReasonFieldMetric,
    CorrectionReasonMetric,
    CorrectionSummary,
    DecisionReasonCategoryMetric,
    DecisionReasonFieldMetric,
    DecisionReasonItem,
    DecisionReasonsAnalyticsResponse,
    DocumentTypeAnalytics,
    DocumentTypesResponse,
    FeedbackFunnel,
    FieldCorrectionMetric,
    FieldCorrectionsResponse,
    OverviewKPIs,
    ProductionFeedbackResponse,
    QueueAnalyticsResponse,
    ReviewReasonMetric,
    ReviewReasonsResponse,
)


def _safe_rate(numerator: float, denominator: float) -> float:
    """Safely calculate percentage rate, returning 0.0 on zero denominator without NaN/Inf."""
    if denominator <= 0.0:
        return 0.0
    return round((float(numerator) / float(denominator)) * 100.0, 1)


class AnalyticsService:
    """Computes real-time operational analytics, HITL metrics, and telemetry dashboards."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    @staticmethod
    def _parse_date_bounds(
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        period: Optional[str] = "all",
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Resolve active temporal scope based on preset period or explicit bounds."""
        if date_from or date_to:
            return date_from, date_to

        if not period or period.lower() == "all":
            return None, None

        now = datetime.utcnow()
        today_start = datetime(now.year, now.month, now.day, 0, 0, 0)
        p = period.lower().strip()
        if p == "today":
            return today_start, now
        elif p == "7d":
            return today_start - timedelta(days=6), now
        elif p == "30d":
            return today_start - timedelta(days=29), now
        elif p == "90d":
            return today_start - timedelta(days=89), now
        return None, None

    async def get_overview(
        self,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        period: Optional[str] = "all",
    ) -> AnalyticsOverviewResponse:
        """Calculate complete operational overview statistics with temporal filtering."""
        start, end = self._parse_date_bounds(date_from, date_to, period)

        # 1. Total intake documents (Document.created_at)
        total_stmt = select(func.count(Document.id))
        if start:
            total_stmt = total_stmt.where(Document.created_at >= start)
        if end:
            total_stmt = total_stmt.where(Document.created_at <= end)
        total_docs = (await self.session.execute(total_stmt)).scalar() or 0

        # 2. Status counts for processed documents (Document.processing_finished_at)
        # Note semantics: processed documents are scoped by when processing finished.
        status_stmt = select(Document.status, func.count(Document.id)).where(
            Document.status.in_([
                DocumentStatus.COMPLETED_AUTOMATIC.value,
                DocumentStatus.COMPLETED_MANUAL.value,
                DocumentStatus.MANUAL_REVIEW.value,
                DocumentStatus.ERROR.value,
            ])
        )
        if start:
            status_stmt = status_stmt.where(Document.processing_finished_at >= start)
        if end:
            status_stmt = status_stmt.where(Document.processing_finished_at <= end)
        status_stmt = status_stmt.group_by(Document.status)

        status_rows = (await self.session.execute(status_stmt)).all()
        status_map = {row[0]: row[1] for row in status_rows}

        automatic_count = status_map.get(DocumentStatus.COMPLETED_AUTOMATIC.value, 0)
        completed_manual_count = status_map.get(DocumentStatus.COMPLETED_MANUAL.value, 0)
        manual_review_count = status_map.get(DocumentStatus.MANUAL_REVIEW.value, 0)
        error_count = status_map.get(DocumentStatus.ERROR.value, 0)
        processed_count = automatic_count + completed_manual_count + manual_review_count + error_count

        # 3. Pending jobs in queue (current state)
        pending_stmt = select(func.count(ProcessingJob.id)).where(
            ProcessingJob.status.in_([JobStatus.PENDING.value, JobStatus.RUNNING.value, JobStatus.RETRYING.value])
        )
        queue_pending = (await self.session.execute(pending_stmt)).scalar() or 0

        # 4. Operational Rates
        automation_rate = _safe_rate(automatic_count, processed_count)
        manual_review_rate = _safe_rate(manual_review_count + completed_manual_count, processed_count)
        failure_rate = _safe_rate(error_count, processed_count)

        # 5. Corrections count in period (AuditEvent.created_at)
        corr_stmt = select(func.count(AuditEvent.id)).where(
            AuditEvent.action == AuditAction.FIELD_CORRECTED.value
        )
        if start:
            corr_stmt = corr_stmt.where(AuditEvent.created_at >= start)
        if end:
            corr_stmt = corr_stmt.where(AuditEvent.created_at <= end)
        corrections_total = (await self.session.execute(corr_stmt)).scalar() or 0

        # Fallback if no audit events exist yet: count ExtractedField.corrected_value
        if corrections_total == 0:
            ef_stmt = select(func.count(ExtractedField.id)).where(ExtractedField.corrected_value.isnot(None))
            corrections_total = (await self.session.execute(ef_stmt)).scalar() or 0

        # 6. Average confidence & durations
        avg_stmt = select(
            func.avg(Document.confidence),
            func.avg(Document.processing_duration_ms),
        ).where(Document.confidence.isnot(None))
        if start:
            avg_stmt = avg_stmt.where(Document.processing_finished_at >= start)
        if end:
            avg_stmt = avg_stmt.where(Document.processing_finished_at <= end)
        avg_res = (await self.session.execute(avg_stmt)).one_or_none()

        avg_conf = round(float(avg_res[0] or 0.0) * 100.0, 1) if avg_res and avg_res[0] is not None else 0.0
        avg_proc_duration = round(float(avg_res[1] or 0.0), 1) if avg_res and avg_res[1] is not None else 0.0

        # Review turnaround latency (Document.review_finished_at)
        avg_rev_stmt = select(func.avg(Document.review_duration_ms)).where(Document.review_duration_ms.isnot(None))
        if start:
            avg_rev_stmt = avg_rev_stmt.where(Document.review_finished_at >= start)
        if end:
            avg_rev_stmt = avg_rev_stmt.where(Document.review_finished_at <= end)
        avg_rev_res = (await self.session.execute(avg_rev_stmt)).scalar()
        avg_rev_duration = round(float(avg_rev_res or 0.0), 1)

        # 7. Review reasons breakdown
        reason_stmt = (
            select(Document.review_reason, func.count(Document.id))
            .where(Document.review_reason != ReviewReason.NONE.value)
        )
        if start:
            reason_stmt = reason_stmt.where(Document.created_at >= start)
        if end:
            reason_stmt = reason_stmt.where(Document.created_at <= end)
        reason_stmt = reason_stmt.group_by(Document.review_reason)

        reason_rows = (await self.session.execute(reason_stmt)).all()
        total_review_docs = sum(r[1] for r in reason_rows) or 1
        review_reasons: List[BreakdownItem] = []
        for r_name, r_cnt in reason_rows:
            pct = _safe_rate(r_cnt, total_review_docs)
            review_reasons.append(BreakdownItem(category=str(r_name), count=r_cnt, percentage=pct))

        # 8. Document type distribution
        dtype_stmt = select(Document.document_type, func.count(Document.id))
        if start:
            dtype_stmt = dtype_stmt.where(Document.created_at >= start)
        if end:
            dtype_stmt = dtype_stmt.where(Document.created_at <= end)
        dtype_stmt = dtype_stmt.group_by(Document.document_type)

        dtype_rows = (await self.session.execute(dtype_stmt)).all()
        dtype_dist: List[BreakdownItem] = []
        for dt_name, dt_cnt in dtype_rows:
            pct = _safe_rate(dt_cnt, total_docs)
            dtype_dist.append(BreakdownItem(category=str(dt_name), count=dt_cnt, percentage=pct))

        # 9. Status distribution
        status_dist: List[BreakdownItem] = []
        for st_name, st_cnt in status_map.items():
            pct = _safe_rate(st_cnt, processed_count)
            status_dist.append(BreakdownItem(category=str(st_name), count=st_cnt, percentage=pct))

        kpis = OverviewKPIs(
            total_documents=total_docs,
            processed_count=processed_count,
            automatic_count=automatic_count,
            manual_review_count=manual_review_count,
            completed_manual_count=completed_manual_count,
            error_count=error_count,
            automation_rate=automation_rate,
            manual_review_rate=manual_review_rate,
            failure_rate=failure_rate,
            corrections_total=corrections_total,
            average_confidence=avg_conf,
            average_processing_time_ms=avg_proc_duration,
            average_review_time_ms=avg_rev_duration,
            queue_pending_count=queue_pending,
        )

        return AnalyticsOverviewResponse(
            kpis=kpis,
            review_reasons=review_reasons,
            document_type_distribution=dtype_dist,
            status_distribution=status_dist,
        )

    async def get_document_types_analytics(
        self,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        period: Optional[str] = "all",
    ) -> DocumentTypesResponse:
        """Granular automation and operational performance metrics grouped by document category."""
        start, end = self._parse_date_bounds(date_from, date_to, period)

        # Get all distinct document types present in system
        types_stmt = select(Document.document_type).distinct()
        type_rows = (await self.session.execute(types_stmt)).scalars().all()

        items: List[DocumentTypeAnalytics] = []

        for dt in sorted(type_rows):
            # Total intake for this type (created_at)
            tot_stmt = select(func.count(Document.id)).where(Document.document_type == dt)
            if start:
                tot_stmt = tot_stmt.where(Document.created_at >= start)
            if end:
                tot_stmt = tot_stmt.where(Document.created_at <= end)
            total_docs = (await self.session.execute(tot_stmt)).scalar() or 0

            # Processed docs for this type (processing_finished_at)
            base_stmt = select(Document).where(
                Document.document_type == dt,
                Document.processing_finished_at.isnot(None),
            )
            if start:
                base_stmt = base_stmt.where(Document.processing_finished_at >= start)
            if end:
                base_stmt = base_stmt.where(Document.processing_finished_at <= end)

            docs_res = await self.session.execute(base_stmt)
            docs = docs_res.scalars().all()

            processed_docs = len(docs)
            auto_cnt = sum(1 for d in docs if d.status == DocumentStatus.COMPLETED_AUTOMATIC.value)
            # Count both active manual_review and completed_manual without double counting
            manual_cnt = sum(
                1 for d in docs if d.status in (DocumentStatus.MANUAL_REVIEW.value, DocumentStatus.COMPLETED_MANUAL.value) or d.decision == AutomationDecision.MANUAL_REVIEW.value
            )
            err_cnt = sum(1 for d in docs if d.status == DocumentStatus.ERROR.value)

            auto_rate = _safe_rate(auto_cnt, processed_docs)
            manual_rate = _safe_rate(manual_cnt, processed_docs)
            fail_rate = _safe_rate(err_cnt, processed_docs)

            conf_vals = [d.confidence for d in docs if d.confidence is not None]
            avg_conf = round((sum(conf_vals) / len(conf_vals)) * 100.0, 1) if conf_vals else 0.0

            proc_times = [d.processing_duration_ms for d in docs if d.processing_duration_ms is not None]
            avg_proc = round(sum(proc_times) / len(proc_times), 1) if proc_times else 0.0

            rev_times = [d.review_duration_ms for d in docs if d.review_duration_ms is not None]
            avg_rev = round(sum(rev_times) / len(rev_times), 1) if rev_times else 0.0

            items.append(
                DocumentTypeAnalytics(
                    document_type=dt,
                    total_documents=total_docs,
                    processed_documents=processed_docs,
                    automatic_count=auto_cnt,
                    manual_review_count=manual_cnt,
                    error_count=err_cnt,
                    automation_rate=auto_rate,
                    manual_review_rate=manual_rate,
                    failure_rate=fail_rate,
                    average_confidence=avg_conf,
                    average_processing_time_ms=avg_proc,
                    average_review_time_ms=avg_rev,
                )
            )

        return DocumentTypesResponse(items=items)

    async def get_automation_trends(
        self,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        period: Optional[str] = "7d",
    ) -> AutomationTrendsResponse:
        """Daily volume and automation rate timeline."""
        start, end = self._parse_date_bounds(date_from, date_to, period)

        now = datetime.utcnow()
        if not end:
            end = now
        if not start:
            # Default to 7 days if 'all' or unspecified
            start = end - timedelta(days=6)

        # Generate continuous list of dates from start to end
        curr = datetime(start.year, start.month, start.day)
        end_date = datetime(end.year, end.month, end.day)

        calendar_days = []
        while curr <= end_date:
            calendar_days.append(curr.strftime("%Y-%m-%d"))
            curr += timedelta(days=1)

        # Query all processed documents in range
        docs_stmt = select(Document).where(
            Document.processing_finished_at.isnot(None),
            Document.processing_finished_at >= start,
            Document.processing_finished_at <= end,
        )
        docs = (await self.session.execute(docs_stmt)).scalars().all()

        # Query total created in range
        created_stmt = select(Document).where(
            Document.created_at >= start,
            Document.created_at <= end,
        )
        created_docs = (await self.session.execute(created_stmt)).scalars().all()

        daily_stats: Dict[str, Dict[str, int]] = {
            day: {"total": 0, "processed": 0, "auto": 0, "manual": 0, "error": 0}
            for day in calendar_days
        }

        for cd in created_docs:
            d_str = cd.created_at.strftime("%Y-%m-%d")
            if d_str in daily_stats:
                daily_stats[d_str]["total"] += 1

        for d in docs:
            if not d.processing_finished_at:
                continue
            d_str = d.processing_finished_at.strftime("%Y-%m-%d")
            if d_str in daily_stats:
                daily_stats[d_str]["processed"] += 1
                if d.status == DocumentStatus.COMPLETED_AUTOMATIC.value:
                    daily_stats[d_str]["auto"] += 1
                elif d.status in (DocumentStatus.MANUAL_REVIEW.value, DocumentStatus.COMPLETED_MANUAL.value):
                    daily_stats[d_str]["manual"] += 1
                elif d.status == DocumentStatus.ERROR.value:
                    daily_stats[d_str]["error"] += 1

        items: List[AutomationTrendItem] = []
        for day in calendar_days:
            st = daily_stats[day]
            proc = st["processed"]
            auto = st["auto"]
            # Automation rate as normalized 0.0..1.0 float rounded to 3 decimal places
            rate = round(float(auto / proc), 3) if proc > 0 else 0.0
            items.append(
                AutomationTrendItem(
                    date=day,
                    total_documents=st["total"],
                    processed_count=proc,
                    automatic_count=auto,
                    manual_review_count=st["manual"],
                    error_count=st["error"],
                    automation_rate=rate,
                )
            )

        return AutomationTrendsResponse(items=items)

    async def get_queue(self) -> QueueAnalyticsResponse:
        """Current operational queue backlog metrics."""
        mr_stmt = select(func.count(Document.id)).where(Document.status == DocumentStatus.MANUAL_REVIEW.value)
        proc_stmt = select(func.count(Document.id)).where(Document.status == DocumentStatus.PROCESSING.value)
        err_stmt = select(func.count(Document.id)).where(Document.status == DocumentStatus.ERROR.value)

        mr_count = (await self.session.execute(mr_stmt)).scalar() or 0
        proc_count = (await self.session.execute(proc_stmt)).scalar() or 0
        err_count = (await self.session.execute(err_stmt)).scalar() or 0

        return QueueAnalyticsResponse(
            manual_review=mr_count,
            processing=proc_count,
            errors=err_count,
        )

    async def get_review_reasons(
        self,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        period: Optional[str] = "all",
    ) -> ReviewReasonsResponse:
        """Diagnostic aggregation of reasons why documents were routed to manual review."""
        start, end = self._parse_date_bounds(date_from, date_to, period)

        reason_stmt = (
            select(Document.review_reason, func.count(Document.id))
            .where(Document.review_reason != ReviewReason.NONE.value)
        )
        if start:
            reason_stmt = reason_stmt.where(Document.created_at >= start)
        if end:
            reason_stmt = reason_stmt.where(Document.created_at <= end)
        reason_stmt = reason_stmt.group_by(Document.review_reason)

        reason_rows = (await self.session.execute(reason_stmt)).all()
        total_reasons = sum(r[1] for r in reason_rows)
        denom = max(total_reasons, 1)

        items: List[ReviewReasonMetric] = []
        for r_name, r_cnt in reason_rows:
            pct = _safe_rate(r_cnt, denom)
            items.append(ReviewReasonMetric(reason=str(r_name), count=r_cnt, percentage=pct))

        return ReviewReasonsResponse(total_reviews=total_reasons, items=items)

    async def get_field_corrections(
        self,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        period: Optional[str] = "all",
    ) -> FieldCorrectionsResponse:
        """Dynamic per-field correction frequency telemetry."""
        start, end = self._parse_date_bounds(date_from, date_to, period)

        stmt = (
            select(
                ExtractedField.field_name,
                func.count(ExtractedField.id).label("total_occ"),
                func.count(case((ExtractedField.corrected_value.isnot(None), 1))).label("corr_cnt"),
            )
            .group_by(ExtractedField.field_name)
            .order_by(ExtractedField.field_name)
        )
        rows = (await self.session.execute(stmt)).all()

        total_fields = sum(r[1] for r in rows)
        total_corrections = sum(r[2] for r in rows)

        items: List[FieldCorrectionMetric] = []
        for f_name, occ, corr in rows:
            rate = _safe_rate(corr, occ)
            items.append(
                FieldCorrectionMetric(
                    field=f_name,
                    total_occurrences=occ,
                    corrected_count=corr,
                    correction_rate=rate,
                )
            )

        return FieldCorrectionsResponse(
            total_fields=total_fields,
            total_corrections=total_corrections,
            items=items,
        )

    async def get_confidence_distribution(
        self,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        period: Optional[str] = "all",
    ) -> ConfidenceAnalyticsResponse:
        """Analyze confidence ranges against review routing and operator corrections."""
        start, end = self._parse_date_bounds(date_from, date_to, period)

        stmt = (
            select(Document)
            .options(selectinload(Document.fields))
            .where(Document.confidence.isnot(None))
        )
        if start:
            stmt = stmt.where(Document.created_at >= start)
        if end:
            stmt = stmt.where(Document.created_at <= end)

        docs = (await self.session.execute(stmt)).scalars().all()

        bucket_defs = [
            ("0.00–0.59", 0.0, 0.60),
            ("0.60–0.69", 0.60, 0.70),
            ("0.70–0.79", 0.70, 0.80),
            ("0.80–0.89", 0.80, 0.90),
            ("0.90–1.00", 0.90, 1.01),
        ]

        buckets_data = {
            name: {"total": 0, "manual": 0, "corrected": 0}
            for name, _, _ in bucket_defs
        }

        conf_sum = 0.0
        conf_count = len(docs)

        for d in docs:
            c = float(d.confidence or 0.0)
            conf_sum += c
            has_correction = any(f.corrected_value is not None for f in d.fields)
            is_manual = (
                d.status in (DocumentStatus.MANUAL_REVIEW.value, DocumentStatus.COMPLETED_MANUAL.value)
                or d.decision == AutomationDecision.MANUAL_REVIEW.value
            )

            for name, low, high in bucket_defs:
                if low <= c < high or (high >= 1.0 and c >= 1.0 and name == "0.90–1.00"):
                    buckets_data[name]["total"] += 1
                    if is_manual:
                        buckets_data[name]["manual"] += 1
                    if has_correction:
                        buckets_data[name]["corrected"] += 1
                    break

        items = [
            ConfidenceBucket(
                bucket=name,
                total_count=data["total"],
                manual_review_count=data["manual"],
                corrected_count=data["corrected"],
                review_rate=_safe_rate(data["manual"], data["total"]),
            )
            for name, data in buckets_data.items()
        ]

        avg_conf = round((conf_sum / max(conf_count, 1)) * 100.0, 1) if conf_count > 0 else 0.0
        return ConfidenceAnalyticsResponse(distribution=items, average_confidence=avg_conf)

    async def get_decision_reasons_analytics(
        self,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        period: Optional[str] = "all",
    ) -> DecisionReasonsAnalyticsResponse:
        """Historical decision analytics grounded strictly in DECISION_EVALUATED audit events.

        Counts actual historical decision events rather than current mutated document states,
        ensuring accurate analytics across reprocessing and operator reviews.
        """
        start, end = self._parse_date_bounds(date_from, date_to, period)

        stmt = select(AuditEvent).where(AuditEvent.action == AuditAction.DECISION_EVALUATED.value)
        if start:
            stmt = stmt.where(AuditEvent.created_at >= start)
        if end:
            stmt = stmt.where(AuditEvent.created_at <= end)

        events = (await self.session.execute(stmt)).scalars().all()

        auto_count = 0
        manual_count = 0

        reason_counts: Dict[str, int] = {}
        category_counts: Dict[str, int] = {}
        field_counts: Dict[str, int] = {}

        for ev in events:
            meta = ev.metadata_json or {}
            dec = str(meta.get("decision", "")).lower()
            if dec == AutomationDecision.AUTOMATIC.value:
                auto_count += 1
            else:
                manual_count += 1
                details = meta.get("review_details")
                if details and isinstance(details, dict) and details.get("reasons"):
                    # Event contains full structured reasons
                    for r in details["reasons"]:
                        if isinstance(r, dict):
                            code = r.get("code") or meta.get("reason") or "unknown"
                            cat = ExplainabilityService.get_category_for_code(code)
                            reason_counts[code] = reason_counts.get(code, 0) + 1
                            category_counts[cat] = category_counts.get(cat, 0) + 1
                            f = r.get("field")
                            if f:
                                field_counts[f] = field_counts.get(f, 0) + 1
                else:
                    # Legacy event without review_details: use persisted reason without reconstructing
                    code = meta.get("reason")
                    if code and code != ReviewReason.NONE.value:
                        cat = ExplainabilityService.get_category_for_code(code)
                        reason_counts[code] = reason_counts.get(code, 0) + 1
                        category_counts[cat] = category_counts.get(cat, 0) + 1

        total_decisions = auto_count + manual_count
        auto_rate = _safe_rate(auto_count, total_decisions)
        manual_rate = _safe_rate(manual_count, total_decisions)

        total_reasons_sum = sum(reason_counts.values())
        reasons_denom = max(total_reasons_sum, 1)

        by_reason = [
            DecisionReasonItem(
                code=code,
                category=ExplainabilityService.get_category_for_code(code),
                count=cnt,
                percentage=_safe_rate(cnt, reasons_denom),
                description=code.replace("_", " ").capitalize(),
            )
            for code, cnt in sorted(reason_counts.items(), key=lambda x: -x[1])
        ]

        total_cat_sum = sum(category_counts.values())
        cat_denom = max(total_cat_sum, 1)
        by_category = [
            DecisionReasonCategoryMetric(
                category=cat,
                count=cnt,
                percentage=_safe_rate(cnt, cat_denom),
            )
            for cat, cnt in sorted(category_counts.items(), key=lambda x: -x[1])
        ]

        total_f_sum = sum(field_counts.values())
        f_denom = max(total_f_sum, 1)
        by_field = [
            DecisionReasonFieldMetric(
                field=f,
                count=cnt,
                percentage=_safe_rate(cnt, f_denom),
            )
            for f, cnt in sorted(field_counts.items(), key=lambda x: -x[1])
        ]

        return DecisionReasonsAnalyticsResponse(
            total_decisions=total_decisions,
            automatic_count=auto_count,
            manual_review_count=manual_count,
            automation_rate=auto_rate,
            manual_review_rate=manual_rate,
            by_reason=by_reason,
            by_category=by_category,
            by_field=by_field,
        )

    async def get_production_feedback(
        self,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        period: Optional[str] = "all",
    ) -> ProductionFeedbackResponse:
        """Calculate production feedback metrics, review-correction correlations, and automation loss funnel."""
        start, end = self._parse_date_bounds(date_from, date_to, period)

        # 1. Funnel overview counts
        proc_stmt = select(Document).where(
            Document.status.in_([
                DocumentStatus.COMPLETED_AUTOMATIC.value,
                DocumentStatus.COMPLETED_MANUAL.value,
                DocumentStatus.MANUAL_REVIEW.value,
                DocumentStatus.ERROR.value,
            ])
        )
        if start:
            proc_stmt = proc_stmt.where(
                (Document.processing_finished_at >= start)
                | (Document.created_at >= start)
            )
        if end:
            proc_stmt = proc_stmt.where(
                (Document.processing_finished_at <= end)
                | (Document.created_at <= end)
            )
        proc_docs = (await self.session.execute(proc_stmt)).scalars().all()

        auto_docs = [
            d for d in proc_docs
            if d.status == DocumentStatus.COMPLETED_AUTOMATIC.value or d.decision == AutomationDecision.AUTOMATIC.value
        ]
        manual_docs = [
            d for d in proc_docs
            if d.status in (DocumentStatus.MANUAL_REVIEW.value, DocumentStatus.COMPLETED_MANUAL.value)
            or d.decision == AutomationDecision.MANUAL_REVIEW.value
        ]
        completed_manual_docs = [
            d for d in proc_docs
            if d.status == DocumentStatus.COMPLETED_MANUAL.value
        ]

        processed_count = len(proc_docs)
        auto_count = len(auto_docs)
        manual_count = len(manual_docs)
        completed_manual_count = len(completed_manual_docs)

        # 2. Query FIELD_CORRECTED audit events in temporal scope
        corr_stmt = select(AuditEvent).where(
            AuditEvent.action == AuditAction.FIELD_CORRECTED.value
        )
        if start:
            corr_stmt = corr_stmt.where(AuditEvent.created_at >= start)
        if end:
            corr_stmt = corr_stmt.where(AuditEvent.created_at <= end)
        corr_events = (await self.session.execute(corr_stmt)).scalars().all()

        total_correction_events = len(corr_events)

        # Group corrections by document_id
        corr_by_doc: Dict[str, List[AuditEvent]] = {}
        for ev in corr_events:
            corr_by_doc.setdefault(ev.document_id, []).append(ev)

        # Map completed manual review document IDs
        completed_doc_map = {d.id: d for d in completed_manual_docs}

        # Also query documents for any corrected document_ids that might not have been in proc_docs
        missing_doc_ids = [doc_id for doc_id in corr_by_doc if doc_id not in completed_doc_map]
        if missing_doc_ids:
            missing_stmt = select(Document).where(Document.id.in_(missing_doc_ids))
            missing_docs = (await self.session.execute(missing_stmt)).scalars().all()
            for md in missing_docs:
                completed_doc_map[md.id] = md
                if md.status == DocumentStatus.COMPLETED_MANUAL.value and md not in completed_manual_docs:
                    completed_manual_docs.append(md)
                    completed_manual_count = len(completed_manual_docs)

        # Distinct documents with corrections (among completed manual reviews or in general)
        docs_with_corrections_set = (
            set(corr_by_doc.keys()).intersection(completed_doc_map.keys())
            if completed_doc_map
            else set(corr_by_doc.keys())
        )
        documents_with_corrections = len(docs_with_corrections_set)

        effective_reviewed_count = max(completed_manual_count, documents_with_corrections)
        without_corrections = max(0, effective_reviewed_count - documents_with_corrections)

        # Operational Rates (guarded against division by zero)
        corr_rate = _safe_rate(documents_with_corrections, effective_reviewed_count)
        no_change_rate = _safe_rate(without_corrections, effective_reviewed_count)
        avg_corrections = (
            round(total_correction_events / documents_with_corrections, 2)
            if documents_with_corrections > 0
            else 0.0
        )

        auto_rate = _safe_rate(auto_count, processed_count)
        manual_rate = _safe_rate(manual_count, processed_count)

        funnel = FeedbackFunnel(
            processed=processed_count,
            automatic=auto_count,
            manual_review=manual_count,
            completed_manual_review=effective_reviewed_count,
            with_corrections=documents_with_corrections,
            without_corrections=without_corrections,
            automatic_rate=auto_rate,
            manual_review_rate=manual_rate,
            correction_rate=corr_rate,
            no_change_review_rate=no_change_rate,
        )

        summary = CorrectionSummary(
            total_correction_events=total_correction_events,
            documents_with_corrections=documents_with_corrections,
            completed_manual_review_documents=effective_reviewed_count,
            correction_rate=corr_rate,
            no_change_review_rate=no_change_rate,
            avg_corrections_per_corrected_document=avg_corrections,
        )

        # 3. Breakdown by Field
        field_event_counts: Dict[str, int] = {}
        field_affected_docs: Dict[str, set] = {}
        for ev in corr_events:
            fn = ev.field_name or "unknown"
            field_event_counts[fn] = field_event_counts.get(fn, 0) + 1
            field_affected_docs.setdefault(fn, set()).add(ev.document_id)

        by_field = [
            CorrectionFieldMetric(
                field=fn,
                correction_count=cnt,
                affected_documents=len(field_affected_docs.get(fn, set())),
                share=_safe_rate(cnt, total_correction_events),
            )
            for fn, cnt in sorted(
                field_event_counts.items(),
                key=lambda x: (-x[1], x[0]),
            )
        ]

        # 4. Resolve Triggering Review Reason and Document Type for each document
        all_doc_ids = set(completed_doc_map.keys()).union(corr_by_doc.keys())
        doc_reasons: Dict[str, str] = {}
        doc_types: Dict[str, str] = {d_id: d.document_type for d_id, d in completed_doc_map.items()}

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

        # Fallback to doc.review_reason or "manual_review"
        for d_id, d in completed_doc_map.items():
            if d_id not in doc_reasons and d.review_reason and d.review_reason != ReviewReason.NONE.value:
                doc_reasons[d_id] = d.review_reason
            elif d_id not in doc_reasons:
                doc_reasons[d_id] = "manual_review"

        for d_id in corr_by_doc:
            if d_id not in doc_reasons:
                doc_reasons[d_id] = "manual_review"

        # 5. Breakdown by Document Type
        type_reviewed_docs: Dict[str, set] = {}
        type_corrected_docs: Dict[str, set] = {}
        type_correction_events: Dict[str, int] = {}

        for d_id, d in completed_doc_map.items():
            dt = d.document_type or "unknown"
            type_reviewed_docs.setdefault(dt, set()).add(d_id)

        for d_id, evs in corr_by_doc.items():
            dt = doc_types.get(d_id, "unknown")
            type_corrected_docs.setdefault(dt, set()).add(d_id)
            type_correction_events[dt] = type_correction_events.get(dt, 0) + len(evs)
            if d_id in completed_doc_map:
                type_reviewed_docs.setdefault(dt, set()).add(d_id)

        by_document_type = [
            CorrectionDocumentTypeMetric(
                document_type=dt,
                reviewed_documents=len(type_reviewed_docs.get(dt, set())),
                documents_with_corrections=len(type_corrected_docs.get(dt, set())),
                correction_events=type_correction_events.get(dt, 0),
                correction_rate=_safe_rate(
                    len(type_corrected_docs.get(dt, set())),
                    len(type_reviewed_docs.get(dt, set())),
                ),
            )
            for dt in sorted(
                type_reviewed_docs.keys(),
                key=lambda dt: (-type_correction_events.get(dt, 0), -len(type_reviewed_docs.get(dt, set())), dt),
            )
        ]

        # 6. Breakdown by Review Reason
        reason_reviewed_docs: Dict[str, set] = {}
        reason_corrected_docs: Dict[str, set] = {}
        reason_correction_events: Dict[str, int] = {}

        for d_id in completed_doc_map:
            r = doc_reasons.get(d_id, "manual_review")
            reason_reviewed_docs.setdefault(r, set()).add(d_id)

        for d_id, evs in corr_by_doc.items():
            r = doc_reasons.get(d_id, "manual_review")
            reason_corrected_docs.setdefault(r, set()).add(d_id)
            reason_correction_events[r] = reason_correction_events.get(r, 0) + len(evs)
            if d_id in completed_doc_map:
                reason_reviewed_docs.setdefault(r, set()).add(d_id)

        by_reason = [
            CorrectionReasonMetric(
                reason=r,
                reviewed_documents=len(reason_reviewed_docs.get(r, set())),
                documents_with_corrections=len(reason_corrected_docs.get(r, set())),
                correction_events=reason_correction_events.get(r, 0),
                correction_rate=_safe_rate(
                    len(reason_corrected_docs.get(r, set())),
                    len(reason_reviewed_docs.get(r, set())),
                ),
            )
            for r in sorted(
                reason_reviewed_docs.keys(),
                key=lambda r: (-reason_correction_events.get(r, 0), -len(reason_reviewed_docs.get(r, set())), r),
            )
        ]

        # 7. Reason x Field cross-tabulation
        reason_field_events: Dict[Tuple[str, str], int] = {}
        reason_field_docs: Dict[Tuple[str, str], set] = {}

        for ev in corr_events:
            r = doc_reasons.get(ev.document_id, "manual_review")
            fn = ev.field_name or "unknown"
            key = (r, fn)
            reason_field_events[key] = reason_field_events.get(key, 0) + 1
            reason_field_docs.setdefault(key, set()).add(ev.document_id)

        by_reason_field = [
            CorrectionReasonFieldMetric(
                review_reason=r,
                field=fn,
                correction_count=cnt,
                affected_documents=len(reason_field_docs.get((r, fn), set())),
            )
            for (r, fn), cnt in sorted(
                reason_field_events.items(),
                key=lambda x: (-x[1], x[0][0], x[0][1]),
            )
        ]

        return ProductionFeedbackResponse(
            summary=summary,
            funnel=funnel,
            by_field=by_field,
            by_document_type=by_document_type,
            by_reason=by_reason,
            by_reason_field=by_reason_field,
        )

    async def _get_recent_documents(self, limit: int = 10) -> List[DocumentResponse]:
        """Retrieve most recently active documents for real-time Control Center visibility."""
        stmt = (
            select(Document)
            .order_by(Document.updated_at.desc(), Document.created_at.desc())
            .limit(limit)
        )
        docs = (await self.session.execute(stmt)).scalars().all()
        return [
            DocumentResponse(
                id=d.id,
                filename=d.filename,
                document_type=d.document_type,
                status=d.status,
                decision=d.decision,
                confidence=round(float(d.confidence), 4) if d.confidence is not None else None,
                review_reason=d.review_reason,
                review_details=d.review_details_json if isinstance(d.review_details_json, dict) else None,
                created_at=d.created_at,
                updated_at=d.updated_at,
                processing_started_at=d.processing_started_at,
                processing_finished_at=d.processing_finished_at,
                processing_duration_ms=d.processing_duration_ms,
                review_started_at=d.review_started_at,
                review_finished_at=d.review_finished_at,
                review_duration_ms=d.review_duration_ms,
                review_wait_duration_ms=d.review_wait_duration_ms,
            )
            for d in docs
        ]

    async def _get_quality_and_confidence(
        self,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        period: Optional[str] = "30d",
    ) -> ControlCenterQuality:
        """Aggregate production quality, OCR/KIE confidence, and domain validation pass rates.

        Strictly observes production database signals on a 0.0..1.0 scale without fabricating
        research benchmark metrics (CER, F1).
        """
        start, end = self._parse_date_bounds(date_from, date_to, period)

        # 1. Average Image Quality Score (DocumentQuality.quality_score)
        q_stmt = (
            select(func.avg(DocumentQuality.quality_score))
            .join(Document, DocumentQuality.document_id == Document.id)
        )
        if start:
            q_stmt = q_stmt.where((Document.processing_finished_at >= start) | (Document.created_at >= start))
        if end:
            q_stmt = q_stmt.where((Document.processing_finished_at <= end) | (Document.created_at <= end))
        q_res = (await self.session.execute(q_stmt)).scalar()
        avg_quality = round(float(q_res), 4) if q_res is not None else None

        # 2. Average Composite Document Confidence (Document.confidence)
        c_stmt = select(func.avg(Document.confidence)).where(Document.confidence.isnot(None))
        if start:
            c_stmt = c_stmt.where((Document.processing_finished_at >= start) | (Document.created_at >= start))
        if end:
            c_stmt = c_stmt.where((Document.processing_finished_at <= end) | (Document.created_at <= end))
        c_res = (await self.session.execute(c_stmt)).scalar()
        avg_doc_conf = round(float(c_res), 4) if c_res is not None else None

        # 3. Average KIE Field Confidence (ExtractedField.confidence)
        f_stmt = (
            select(func.avg(ExtractedField.confidence))
            .join(Document, ExtractedField.document_id == Document.id)
            .where(ExtractedField.confidence.isnot(None))
        )
        if start:
            f_stmt = f_stmt.where((Document.processing_finished_at >= start) | (Document.created_at >= start))
        if end:
            f_stmt = f_stmt.where((Document.processing_finished_at <= end) | (Document.created_at <= end))
        f_res = (await self.session.execute(f_stmt)).scalar()
        avg_kie_conf = round(float(f_res), 4) if f_res is not None else None

        # 4. Average OCR Token Confidence (from DocumentPage.tokens_json)
        p_stmt = (
            select(DocumentPage.tokens_json)
            .join(Document, DocumentPage.document_id == Document.id)
            .where(DocumentPage.tokens_json.isnot(None))
        )
        if start:
            p_stmt = p_stmt.where((Document.processing_finished_at >= start) | (Document.created_at >= start))
        if end:
            p_stmt = p_stmt.where((Document.processing_finished_at <= end) | (Document.created_at <= end))
        pages = (await self.session.execute(p_stmt)).scalars().all()
        all_token_confs: List[float] = []
        for tj in pages:
            if tj and isinstance(tj, list):
                for tok in tj:
                    if isinstance(tok, dict) and tok.get("confidence") is not None:
                        try:
                            all_token_confs.append(float(tok["confidence"]))
                        except (ValueError, TypeError):
                            pass
        avg_ocr_conf = round(sum(all_token_confs) / len(all_token_confs), 4) if all_token_confs else None

        # 5. Document-Level Validation Pass Rate (documents_passed_validation / documents_validated)
        val_docs_stmt = select(Document).where(Document.decision.isnot(None))
        if start:
            val_docs_stmt = val_docs_stmt.where((Document.processing_finished_at >= start) | (Document.created_at >= start))
        if end:
            val_docs_stmt = val_docs_stmt.where((Document.processing_finished_at <= end) | (Document.created_at <= end))
        val_docs = (await self.session.execute(val_docs_stmt)).scalars().all()

        total_validated = len(val_docs)
        validation_failure_reasons = {
            ReviewReason.VALIDATION_FAILED.value,
            ReviewReason.MISSING_REQUIRED_FIELD.value,
            ReviewReason.INVALID_DATE.value,
            ReviewReason.INVALID_NUMBER.value,
            ReviewReason.EMPTY_VALUE.value,
            ReviewReason.DOCUMENT_INTEGRITY.value,
        }
        failed_val_count = 0
        passed_val_count = 0
        for d in val_docs:
            is_val_fail = False
            if d.decision == AutomationDecision.MANUAL_REVIEW.value:
                if d.review_reason in validation_failure_reasons:
                    is_val_fail = True
                elif d.review_details_json and isinstance(d.review_details_json, dict):
                    if d.review_details_json.get("validation_errors"):
                        is_val_fail = True
            if is_val_fail:
                failed_val_count += 1
            else:
                passed_val_count += 1

        val_pass_rate = round(float(passed_val_count) / float(total_validated), 4) if total_validated > 0 else None

        return ControlCenterQuality(
            average_quality_score=avg_quality,
            average_document_confidence=avg_doc_conf,
            average_ocr_confidence=avg_ocr_conf,
            average_kie_confidence=avg_kie_conf,
            validation_pass_rate=val_pass_rate,
            total_documents_validated=total_validated,
            passed_validation_count=passed_val_count,
            failed_validation_count=failed_val_count,
        )

    async def get_control_center(
        self,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        period: Optional[str] = "30d",
    ) -> ControlCenterResponse:
        """Consolidate production analytics, live queue backlog, and quality telemetry."""
        # 1. Reuse existing canonical services (Single source of truth)
        overview = await self.get_overview(date_from=date_from, date_to=date_to, period=period)
        feedback = await self.get_production_feedback(date_from=date_from, date_to=date_to, period=period)
        decision_reasons = await self.get_decision_reasons_analytics(date_from=date_from, date_to=date_to, period=period)
        queue = await self.get_queue()  # Real-time / Current operational state
        trends = await self.get_automation_trends(date_from=date_from, date_to=date_to, period=period)
        recent_docs = await self._get_recent_documents(limit=10)
        quality = await self._get_quality_and_confidence(date_from=date_from, date_to=date_to, period=period)

        # 2. Build ControlCenterPeriod
        p_obj = ControlCenterPeriod(
            period=period or "30d",
            date_from=date_from,
            date_to=date_to,
        )

        # 3. Build ControlCenterSummary (normalized 0.0..1.0 confidence, canonical rates from overview & feedback)
        norm_confidence = round(float(overview.kpis.average_confidence) / 100.0, 4) if overview.kpis.average_confidence else 0.0

        summary = ControlCenterSummary(
            total_documents=overview.kpis.total_documents,
            processed_count=overview.kpis.processed_count,
            automatic_count=overview.kpis.automatic_count,
            manual_review_count=overview.kpis.manual_review_count,
            completed_manual_count=overview.kpis.completed_manual_count,
            error_count=overview.kpis.error_count,
            automation_rate=overview.kpis.automation_rate,
            manual_review_rate=overview.kpis.manual_review_rate,
            correction_rate=feedback.summary.correction_rate,
            no_change_review_rate=feedback.summary.no_change_review_rate,
            average_confidence=norm_confidence,
            average_processing_time_ms=overview.kpis.average_processing_time_ms,
            average_review_time_ms=overview.kpis.average_review_time_ms,
        )

        return ControlCenterResponse(
            period=p_obj,
            summary=summary,
            queue=queue,
            funnel=feedback.funnel,
            quality=quality,
            review_reasons=decision_reasons.by_reason,
            feedback=feedback,
            trends=trends.items,
            recent_documents=recent_docs,
        )



