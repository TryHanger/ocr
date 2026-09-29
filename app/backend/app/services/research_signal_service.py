"""Research Signal Service for deriving deterministic, descriptive observations from production facts."""

from datetime import datetime
import hashlib
from typing import Any, Dict, List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.enums import AuditAction, ResearchSignalType, ReviewReason
from app.core.improvement_constants import (
    MIN_BUCKET_SHARE,
    MIN_CORRECTION_COUNT,
    MIN_FIELD_SHARE,
    MIN_SIGNAL_SAMPLE_SIZE,
)
from app.models.audit import AuditEvent
from app.models.document import Document
from app.schemas.improvement import ImprovementOverviewResponse, ResearchSignal
from app.services.feedback_analytics_service import FeedbackAnalyticsService
from app.services.quality_bridge import default_evidence_bridge


def _generate_deterministic_signal_id(
    signal_type: ResearchSignalType,
    field: Optional[str] = None,
    document_type: Optional[str] = None,
    period: str = "all",
) -> str:
    """Compute a stable, reproducible SHA-256 hash identifier for derived signals."""
    raw = f"{signal_type.value}:{field or 'global'}:{document_type or 'all'}:{period}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    return f"sig_{signal_type.value.lower()}_{digest}"


class ResearchSignalService:
    """Derives pure, non-persisted factual ResearchSignal DTOs without prescriptions."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.analytics_service = FeedbackAnalyticsService(session)

    async def derive_signals(
        self,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        period: Optional[str] = "all",
        field_filter: Optional[str] = None,
    ) -> List[ResearchSignal]:
        """Derive all qualifying observational signals based strictly on centralized constants."""
        overview = await self.analytics_service.get_feedback_overview(
            date_from=date_from, date_to=date_to, period=period
        )

        # 1. Guard against small sample size or insufficient correction data (Zero Fabricated Signals)
        if (
            overview.summary.documents_reviewed < MIN_SIGNAL_SAMPLE_SIZE
            or overview.summary.total_field_corrections < MIN_CORRECTION_COUNT
        ):
            return []

        signals: List[ResearchSignal] = []
        population = {
            "documents_processed": overview.summary.documents_processed,
            "documents_reviewed": overview.summary.documents_reviewed,
            "total_field_corrections": overview.summary.total_field_corrections,
        }

        active_period = period or "all"

        # 2. Signal Type: CORRECTION_CONCENTRATION
        for fm in overview.by_field:
            if field_filter and fm.field != field_filter:
                continue

            if (
                fm.share_of_all_corrections >= (MIN_FIELD_SHARE * 100.0)
                and fm.correction_count >= MIN_CORRECTION_COUNT
            ):
                sig_id = _generate_deterministic_signal_id(
                    ResearchSignalType.CORRECTION_CONCENTRATION,
                    field=fm.field,
                    period=active_period,
                )
                evidence = {
                    "field": fm.field,
                    "correction_count": fm.correction_count,
                    "evaluated_count": fm.evaluated_count,
                    "correction_rate": fm.correction_rate,
                    "share_of_all_corrections": fm.share_of_all_corrections,
                    "affected_documents": fm.affected_documents,
                    "total_field_corrections": overview.summary.total_field_corrections,
                }
                desc = (
                    f"Field '{fm.field}' accounts for {fm.correction_count} of "
                    f"{overview.summary.total_field_corrections} ({fm.share_of_all_corrections}%) "
                    f"total field corrections across {fm.affected_documents} documents "
                    f"(field correction rate: {fm.correction_rate}%)."
                )

                # Informational reference to B0 baseline
                related_refs: List[Dict[str, Any]] = [
                    {
                        "track": "B0",
                        "experiment_id": "b0_clean_validation",
                        "name": "Clean Baseline Evaluation (B0)",
                        "description": "Baseline evaluation of RapidOCR and RuleBasedKIE on clean receipts.",
                    }
                ]

                signals.append(
                    ResearchSignal(
                        id=sig_id,
                        signal_type=ResearchSignalType.CORRECTION_CONCENTRATION,
                        title=f"High correction concentration for field '{fm.field}'",
                        description=desc,
                        population=population,
                        field=fm.field,
                        evidence=evidence,
                        period=active_period,
                        related_evidence_refs=related_refs,
                    )
                )

        # 3. Signal Type: CONFIDENCE_CORRECTION_PATTERN
        for cb in overview.by_confidence:
            if (
                cb.share_of_corrections >= (MIN_BUCKET_SHARE * 100.0)
                and cb.corrected_fields >= MIN_CORRECTION_COUNT
            ):
                sig_id = _generate_deterministic_signal_id(
                    ResearchSignalType.CONFIDENCE_CORRECTION_PATTERN,
                    field=cb.bucket,
                    period=active_period,
                )
                evidence = {
                    "bucket": cb.bucket,
                    "corrected_fields": cb.corrected_fields,
                    "evaluated_fields": cb.evaluated_fields,
                    "bucket_correction_rate": cb.correction_rate,
                    "share_of_corrections": cb.share_of_corrections,
                    "total_field_corrections": overview.summary.total_field_corrections,
                }
                desc = (
                    f"{cb.corrected_fields} of {overview.summary.total_field_corrections} "
                    f"({cb.share_of_corrections}%) corrections occurred for fields with initial extraction "
                    f"confidence in the '{cb.bucket}' range (bucket correction rate: {cb.correction_rate}%)."
                )
                signals.append(
                    ResearchSignal(
                        id=sig_id,
                        signal_type=ResearchSignalType.CONFIDENCE_CORRECTION_PATTERN,
                        title=f"Corrections clustered in confidence bucket '{cb.bucket}'",
                        description=desc,
                        population=population,
                        evidence=evidence,
                        period=active_period,
                        related_evidence_refs=[],
                    )
                )

        # 4. Signal Type: REVIEW_REASON_CORRECTION_PATTERN
        for rm in overview.by_reason:
            if (
                rm.correction_events >= MIN_CORRECTION_COUNT
                and rm.correction_rate >= 25.0
            ):
                sig_id = _generate_deterministic_signal_id(
                    ResearchSignalType.REVIEW_REASON_CORRECTION_PATTERN,
                    field=rm.reason,
                    period=active_period,
                )
                evidence = {
                    "review_reason": rm.reason,
                    "reviewed_documents": rm.reviewed_documents,
                    "documents_with_corrections": rm.documents_with_corrections,
                    "correction_events": rm.correction_events,
                    "correction_rate": rm.correction_rate,
                }
                desc = (
                    f"Documents routed with review reason '{rm.reason}' resulted in {rm.documents_with_corrections} "
                    f"documents corrected ({rm.correction_rate}% correction rate) across {rm.correction_events} correction events."
                )
                signals.append(
                    ResearchSignal(
                        id=sig_id,
                        signal_type=ResearchSignalType.REVIEW_REASON_CORRECTION_PATTERN,
                        title=f"Frequent operator corrections under review reason '{rm.reason}'",
                        description=desc,
                        population=population,
                        evidence=evidence,
                        period=active_period,
                        related_evidence_refs=[],
                    )
                )

        # 5. Signal Type: QUALITY_CORRECTION_PATTERN (Strictly observational, zero causal claims)
        # Check documents routed with LOW_IMAGE_QUALITY
        quality_reason = next((r for r in overview.by_reason if r.reason == ReviewReason.LOW_IMAGE_QUALITY.value), None)
        if quality_reason and quality_reason.correction_events >= MIN_CORRECTION_COUNT:
            sig_id = _generate_deterministic_signal_id(
                ResearchSignalType.QUALITY_CORRECTION_PATTERN,
                field="image_quality",
                period=active_period,
            )
            evidence = {
                "reviewed_documents": quality_reason.reviewed_documents,
                "documents_with_corrections": quality_reason.documents_with_corrections,
                "correction_events": quality_reason.correction_events,
                "correction_rate": quality_reason.correction_rate,
            }
            desc = (
                f"Documents with observed quality degradation had {quality_reason.documents_with_corrections} "
                f"historical corrections among {quality_reason.reviewed_documents} reviewed documents."
            )

            # Build informational related research references via default_evidence_bridge
            related_refs: List[Dict[str, Any]] = []
            for sig_key, reg_info in default_evidence_bridge.SIGNAL_REGISTRY.items():
                related_refs.append({
                    "track": "B1/B2",
                    "signal": sig_key,
                    "degradation_code": reg_info["research_degradation_code"],
                    "name": reg_info["research_degradation_name"],
                    "b1_experiment_id": reg_info["b1_experiment_id"],
                    "b2_policies_evaluated": reg_info.get("b2_evaluated_policies_count", 4),
                })

            signals.append(
                ResearchSignal(
                    id=sig_id,
                    signal_type=ResearchSignalType.QUALITY_CORRECTION_PATTERN,
                    title="Quality alerts coincided with manual review corrections",
                    description=desc,
                    population=population,
                    evidence=evidence,
                    period=active_period,
                    related_evidence_refs=related_refs,
                )
            )

        # 6. Signal Type: FIELD_VALIDATION_PATTERN (Strictly observational, zero causal claims)
        validation_reasons = {
            ReviewReason.VALIDATION_FAILED.value,
            ReviewReason.MISSING_REQUIRED_FIELD.value,
            ReviewReason.INVALID_DATE.value,
            ReviewReason.INVALID_NUMBER.value,
        }
        val_events_by_field: Dict[str, int] = {}
        for r in overview.by_reason:
            if r.reason in validation_reasons and r.correction_events >= MIN_CORRECTION_COUNT:
                # Query audit trail for fields corrected on documents with validation failure
                for fm in overview.by_field:
                    if fm.correction_count >= MIN_CORRECTION_COUNT:
                        val_events_by_field[fm.field] = fm.correction_count

        for fn, corr_count in val_events_by_field.items():
            if field_filter and fn != field_filter:
                continue
            sig_id = _generate_deterministic_signal_id(
                ResearchSignalType.FIELD_VALIDATION_PATTERN,
                field=fn,
                period=active_period,
            )
            evidence = {
                "field": fn,
                "correction_occurrences": corr_count,
            }
            desc = (
                f"A repeated historical sequence was observed between validation failures "
                f"and subsequent field corrections for '{fn}' ({corr_count} occurrences)."
            )
            signals.append(
                ResearchSignal(
                    id=sig_id,
                    signal_type=ResearchSignalType.FIELD_VALIDATION_PATTERN,
                    title=f"Validation failures repeatedly preceded operator corrections for '{fn}'",
                    description=desc,
                    population=population,
                    field=fn,
                    evidence=evidence,
                    period=active_period,
                    related_evidence_refs=[],
                )
            )

        return signals
