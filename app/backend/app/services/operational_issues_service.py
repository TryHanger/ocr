"""Operational Issues & Action Resolution Service.

Provides a pure derived/read-model that projects actionable OperationalIssue DTOs
and declarative operator guidance from existing production models.

Guiding Principles:
1. Zero Database Persistence:
   - OperationalIssue is purely dynamically derived on-the-fly.
   - Never creates database tables or secondary state storage.
2. Safe Declarative Actions:
   - ActionRegistry outputs declarative navigation and trigger descriptors.
   - Derivation NEVER mutates documents, retries jobs, or applies corrections.
3. Deterministic Severity:
   - Rules derive severity strictly from operational status (CRITICAL, HIGH, MEDIUM, LOW, INFO).
   - No opaque AI priority scores or fabricated probabilities.
4. Tri-Layer Evidence:
   - Production: factual observed values vs configured thresholds.
   - Historical: descriptive counts from real audit trails (or None if unavailable).
   - Research: informational reference to B1/B2 experiments without automated prescription.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import AuditAction, DocumentStatus, ReviewReason
from app.models.audit import AuditEvent
from app.models.document import Document
from app.schemas.decision import DecisionExplanation, DecisionReason
from app.schemas.operations import (
    ActionType,
    DocumentOperationsResponse,
    IssueCategory,
    IssueEvidence,
    IssueSeverity,
    OperationalAction,
    OperationalIssue,
    OperationsBacklogSummary,
    ResolutionStatus,
)
from app.services.explainability_service import ExplainabilityService
from app.services.quality_bridge import default_evidence_bridge


class ActionRegistry:
    """Centralized registry generating safe, declarative operator actions."""

    @classmethod
    def get_actions_for_issue(
        cls,
        code: str,
        category: IssueCategory,
        document_id: str,
        field_name: Optional[str] = None,
        condition_code: Optional[str] = None,
        can_review: bool = True,
        can_retry: bool = False,
        status: ResolutionStatus = ResolutionStatus.OPEN,
    ) -> List[OperationalAction]:
        """Produce allowed declarative actions tailored to an issue's context."""
        actions: List[OperationalAction] = []

        # 1. Field-specific actions
        if field_name:
            actions.append(
                OperationalAction(
                    id=f"act-field-{field_name}",
                    type=ActionType.OPEN_FIELD,
                    label=f"Inspect Field '{field_name}'",
                    description=f"Select and examine the extraction and bounding box for field '{field_name}'.",
                    target="fields",
                    params={"document_id": document_id, "field_name": field_name},
                )
            )

        # 2. Validation view
        if category == IssueCategory.VALIDATION:
            actions.append(
                OperationalAction(
                    id="act-validation",
                    type=ActionType.OPEN_VALIDATION,
                    label="View Validation Details",
                    description="Open schema validation panel to view required fields and format constraints.",
                    target="fields",
                    params={"document_id": document_id, "field_name": field_name} if field_name else {"document_id": document_id},
                )
            )

        # 3. Quality view
        if category == IssueCategory.QUALITY:
            actions.append(
                OperationalAction(
                    id="act-quality",
                    type=ActionType.OPEN_QUALITY_EVIDENCE,
                    label="View Image Quality",
                    description="Open image quality diagnostics covering blur, noise, resolution, and contrast.",
                    target="quality",
                    params={"document_id": document_id},
                )
            )

        # 4. Research Evidence view (strictly informational reference)
        if condition_code:
            track = "B2" if condition_code in ("D1", "D3", "D5", "D6") else "B1"
            actions.append(
                OperationalAction(
                    id=f"act-research-{condition_code.lower()}",
                    type=ActionType.OPEN_RESEARCH_EVIDENCE,
                    label=f"View Research Evidence ({condition_code})",
                    description=f"Open {track} research explorer for benchmark evidence on {condition_code}.",
                    target="research",
                    params={"document_id": document_id, "track": track, "condition_code": condition_code},
                )
            )

        # 5. Review document (primary HITL workflow)
        if can_review and status != ResolutionStatus.RESOLVED:
            actions.append(
                OperationalAction(
                    id="act-review",
                    type=ActionType.OPEN_REVIEW,
                    label="Review Document",
                    description="Open manual review controls to approve, correct, or reject this document.",
                    target="review",
                    params={"document_id": document_id},
                )
            )

        # 6. Retry processing (explicit operator action only for genuine execution/pipeline failures)
        is_pipeline_issue = category == IssueCategory.PROCESSING or code in (
            ReviewReason.PIPELINE_FAILURE.value,
            ReviewReason.OCR_ERROR.value,
            ReviewReason.KIE_ERROR.value,
            ReviewReason.SYSTEM_ERROR.value,
        )
        if can_retry and is_pipeline_issue:
            actions.append(
                OperationalAction(
                    id="act-retry",
                    type=ActionType.RETRY_PROCESSING,
                    label="Retry Processing",
                    description="Schedule document re-processing through the automated pipeline.",
                    target="pipeline",
                    params={"document_id": document_id},
                )
            )

        # 7. Document viewer fallback
        actions.append(
            OperationalAction(
                id="act-document",
                type=ActionType.OPEN_DOCUMENT,
                label="View Document",
                description="View original raster image and full OCR token overlays.",
                target="document",
                params={"document_id": document_id},
            )
        )

        return actions


class OperationalIssuesService:
    """Derives operational issues dynamically from existing production entities."""

    @classmethod
    def derive_resolution_status(cls, doc: Document) -> ResolutionStatus:
        """Compute the canonical derived resolution status.

        Semantics:
        - RESOLVED: Terminal completed states (completed_automatic, completed_manual) or review finished.
        - IN_REVIEW: Active operator review underway (manual_review with review_started_at set).
        - OPEN: Pending manual intervention (manual_review or error, not yet picked up).
        """
        if doc.status in (DocumentStatus.COMPLETED_AUTOMATIC.value, DocumentStatus.COMPLETED_MANUAL.value):
            return ResolutionStatus.RESOLVED
        if doc.review_finished_at is not None:
            return ResolutionStatus.RESOLVED

        if doc.status == DocumentStatus.MANUAL_REVIEW.value:
            if doc.review_started_at is not None:
                return ResolutionStatus.IN_REVIEW
            return ResolutionStatus.OPEN

        if doc.status == DocumentStatus.ERROR.value:
            return ResolutionStatus.OPEN

        # Pending processing / uploaded
        return ResolutionStatus.OPEN

    @classmethod
    def _find_event_timestamp(
        cls,
        doc: Document,
        action: Optional[str] = None,
    ) -> Optional[datetime]:
        """Extract historical timestamp from primary event without generating current time."""
        if hasattr(doc, "audit_events") and doc.audit_events:
            if action:
                for ev in doc.audit_events:
                    if ev.action == action:
                        return ev.created_at
            # First audit event fallback
            return doc.audit_events[0].created_at

        if doc.processing_finished_at:
            return doc.processing_finished_at
        if doc.created_at:
            return doc.created_at
        return None

    @classmethod
    def derive_document_issues(
        cls,
        doc: Document,
        historical_summary: Optional[Dict[str, Any]] = None,
    ) -> DocumentOperationsResponse:
        """Project all actionable operational issues from the document's production state.

        Args:
            doc: Production Document entity with relationships loaded.
            historical_summary: Optional factual count of similar reviews from audit trail.

        Returns:
            DocumentOperationsResponse containing derived issues and declarative actions.
        """
        res_status = cls.derive_resolution_status(doc)
        can_retry = doc.status in (DocumentStatus.ERROR.value, DocumentStatus.MANUAL_REVIEW.value)
        can_review = doc.status == DocumentStatus.MANUAL_REVIEW.value

        issues: List[OperationalIssue] = []

        # 1. Processing pipeline failure check
        if doc.status == DocumentStatus.ERROR.value:
            job_err_msg = None
            if hasattr(doc, "jobs") and doc.jobs:
                latest_job = doc.jobs[0]
                job_err_msg = latest_job.error_message

            err_code = doc.review_reason if (doc.review_reason and doc.review_reason != ReviewReason.NONE.value) else ReviewReason.PIPELINE_FAILURE.value
            created_ts = cls._find_event_timestamp(doc, AuditAction.PROCESSING_FAILED.value)

            issues.append(
                OperationalIssue(
                    id=f"iss-{doc.id}-proc",
                    document_id=doc.id,
                    category=IssueCategory.PROCESSING,
                    severity=IssueSeverity.CRITICAL,
                    code=err_code,
                    title="Document Processing Pipeline Failure",
                    description=job_err_msg or f"Processing terminated due to pipeline failure ({err_code}).",
                    source="processing_pipeline",
                    status=res_status,
                    evidence=IssueEvidence(
                        production=job_err_msg or f"Processing failed with error code: {err_code}",
                        historical=None,
                        research=None,
                    ),
                    available_actions=ActionRegistry.get_actions_for_issue(
                        code=err_code,
                        category=IssueCategory.PROCESSING,
                        document_id=doc.id,
                        can_review=False,
                        can_retry=True,
                        status=res_status,
                    ),
                    created_at=created_ts,
                )
            )

        # 2. Derive issues from decision explanation & reasons (read-oriented, no duplicated heavy validation)
        explanation = ExplainabilityService.explain_document(doc)

        # Inspect research evidence correspondence from existing quality adapter
        research_ref = None
        if hasattr(doc, "quality") and doc.quality:
            research_ref = default_evidence_bridge.get_evidence_for_quality(doc.quality)

        decision_eval_ts = cls._find_event_timestamp(doc, AuditAction.DECISION_EVALUATED.value)

        for idx, reason in enumerate(explanation.reasons):
            # Map canonical reason category to typed IssueCategory
            cat_str = reason.category.lower() if reason.category else "validation"
            if cat_str == "confidence":
                category = IssueCategory.CONFIDENCE
            elif cat_str == "quality":
                category = IssueCategory.QUALITY
            elif cat_str == "validation":
                category = IssueCategory.VALIDATION
            elif cat_str == "pipeline":
                category = IssueCategory.PROCESSING
            else:
                category = IssueCategory.MANUAL_REVIEW

            # Deterministic severity rule
            if res_status == ResolutionStatus.RESOLVED:
                severity = IssueSeverity.INFO
            elif category == IssueCategory.PROCESSING:
                severity = IssueSeverity.CRITICAL
            elif category == IssueCategory.VALIDATION:
                severity = IssueSeverity.HIGH
            elif category == IssueCategory.CONFIDENCE:
                severity = IssueSeverity.HIGH if reason.blocking else IssueSeverity.MEDIUM
            elif category == IssueCategory.QUALITY:
                severity = IssueSeverity.MEDIUM
            else:
                severity = IssueSeverity.MEDIUM

            # Title formulation
            title_prefix = {
                IssueCategory.CONFIDENCE: "Low Field Extraction Confidence",
                IssueCategory.VALIDATION: "Domain Validation Rule Violation",
                IssueCategory.QUALITY: "Image Quality Degradation Detected",
                IssueCategory.PROCESSING: "Pipeline Execution Issue",
                IssueCategory.MANUAL_REVIEW: "Manual Review Routing Reason",
            }.get(category, "Operational Issue")

            issue_title = f"{title_prefix}: {reason.field}" if reason.field else title_prefix

            # Tri-layer evidence assembly
            prod_ev = None
            if reason.observed_value is not None and reason.threshold is not None:
                prod_ev = f"Observed value ({reason.observed_value}) failed configured policy threshold ({reason.threshold})."
            else:
                prod_ev = reason.message

            hist_ev = None
            if historical_summary and reason.code in historical_summary:
                stats = historical_summary[reason.code]
                hist_ev = (
                    f"{stats['total']} similar manual review(s) recorded: "
                    f"{stats['corrections']} contained correction(s), "
                    f"{stats['no_corrections']} had no changes."
                )

            res_ev = None
            cond_code = None
            if category == IssueCategory.QUALITY and research_ref:
                cond_code = research_ref.research_degradation_code
                res_ev = (
                    f"Research evidence available: {research_ref.research_degradation_name} "
                    f"({research_ref.research_degradation_code}). B1 degradation curves and B2 "
                    f"preprocessing benchmarks are available for operator inspection."
                )

            actions = ActionRegistry.get_actions_for_issue(
                code=reason.code,
                category=category,
                document_id=doc.id,
                field_name=reason.field,
                condition_code=cond_code,
                can_review=can_review,
                can_retry=can_retry,
                status=res_status,
            )

            issues.append(
                OperationalIssue(
                    id=f"iss-{doc.id}-{idx}",
                    document_id=doc.id,
                    category=category,
                    severity=severity,
                    code=reason.code,
                    title=issue_title,
                    description=reason.message,
                    observed_value=reason.observed_value,
                    threshold=reason.threshold,
                    field=reason.field,
                    source=(
                        "confidence_engine"
                        if category == IssueCategory.CONFIDENCE
                        else "document_validator"
                        if category == IssueCategory.VALIDATION
                        else "quality_analyzer"
                        if category == IssueCategory.QUALITY
                        else "decision_engine"
                    ),
                    status=res_status,
                    evidence=IssueEvidence(
                        production=prod_ev,
                        historical=hist_ev,
                        research=res_ev,
                    ),
                    available_actions=actions,
                    created_at=decision_eval_ts,
                )
            )

        # 3. Overall summary formulation
        total_issues = len(issues)
        blocking_issues = sum(
            1 for iss in issues if iss.severity in (IssueSeverity.CRITICAL, IssueSeverity.HIGH) and iss.status != ResolutionStatus.RESOLVED
        )

        if total_issues == 0:
            summary = "Zero operational issues detected. Document processed straight-through."
        elif res_status == ResolutionStatus.RESOLVED:
            summary = "Document processing completed and resolved. Operational review is satisfied."
        elif blocking_issues > 0:
            summary = f"{blocking_issues} blocking operational issue(s) require manual operator review and resolution."
        else:
            summary = f"{total_issues} non-blocking operational advisory issue(s) identified."


        return DocumentOperationsResponse(
            document_id=doc.id,
            overall_status=res_status,
            total_issues=total_issues,
            blocking_issues=blocking_issues,
            issues=issues,
            summary=summary,
        )

    @classmethod
    async def get_backlog_summary(cls, session: AsyncSession) -> OperationsBacklogSummary:
        """Compute live projection of operational queue backlog without secondary persistence."""
        # 1. Total open (review not yet started)
        open_stmt = select(func.count(Document.id)).where(
            Document.status == DocumentStatus.MANUAL_REVIEW.value,
            Document.review_started_at.is_(None),
        )
        total_open = (await session.execute(open_stmt)).scalar() or 0

        # 2. Total in review (actively being reviewed)
        in_review_stmt = select(func.count(Document.id)).where(
            Document.status == DocumentStatus.MANUAL_REVIEW.value,
            Document.review_started_at.isnot(None),
            Document.review_finished_at.is_(None),
        )
        total_in_review = (await session.execute(in_review_stmt)).scalar() or 0

        # 3. Grouping by canonical reason for all active review queue items
        active_review_stmt = (
            select(Document.review_reason, func.count(Document.id))
            .where(
                Document.status == DocumentStatus.MANUAL_REVIEW.value,
                Document.review_reason != ReviewReason.NONE.value,
            )
            .group_by(Document.review_reason)
        )
        reason_rows = (await session.execute(active_review_stmt)).all()

        by_reason: Dict[str, int] = {}
        by_category: Dict[str, int] = {}

        for r_name, r_cnt in reason_rows:
            r_str = str(r_name)
            by_reason[r_str] = r_cnt
            cat_name = ExplainabilityService.get_category_for_code(r_str)
            by_category[cat_name] = by_category.get(cat_name, 0) + r_cnt

        return OperationsBacklogSummary(
            total_open=total_open,
            total_in_review=total_in_review,
            by_category=by_category,
            by_reason=by_reason,
        )
