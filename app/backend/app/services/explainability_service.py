"""Decision & Automation Explainability Service.

Provides factual, structured, and auditable explanations for production automation decisions.

Guiding Principles:
1. Strict Semantic Boundaries:
   - Describe existing production decisions; never alter decision outcomes.
   - Do NOT duplicate or invent decision thresholds; all thresholds come from production
     configuration or persisted evidence.
   - B1/B2 research evidence remains strictly informational and isolated.
2. Historical Immutability:
   - When a document is reviewed or corrected (COMPLETED_MANUAL), its historical decision
     explanation is retrieved from the DECISION_EVALUATED audit event, never reconstructed
     from mutated current fields.
   - For legacy DECISION_EVALUATED records lacking review_details, use persisted reason and explanation
     without reconstructing missing reasons from current document state.
3. Canonical Reason Preservation:
   - primary_reason strictly preserves the canonical review_reason without inventing priority ordering.
4. Evidence Fields:
   - observed_value and threshold are optional evidence fields, populated only when meaningful
     production evidence exists.
"""

import json
from typing import Any, Dict, List, Optional, Union
from app.core.enums import AuditAction, AutomationDecision, DocumentStatus, ReviewReason
from app.models.document import Document
from app.schemas.decision import DecisionExplanation, DecisionReason


# Mapping canonical review reason codes to semantic categories
REASON_CATEGORIES: Dict[str, str] = {
    ReviewReason.LOW_CONFIDENCE.value: "confidence",
    ReviewReason.LOW_IMAGE_QUALITY.value: "quality",
    ReviewReason.MISSING_REQUIRED_FIELD.value: "validation",
    ReviewReason.VALIDATION_FAILED.value: "validation",
    ReviewReason.INVALID_NUMBER.value: "validation",
    ReviewReason.INVALID_DATE.value: "validation",
    ReviewReason.EMPTY_VALUE.value: "validation",
    ReviewReason.DOCUMENT_INTEGRITY.value: "validation",
    ReviewReason.OCR_ERROR.value: "pipeline",
    ReviewReason.KIE_ERROR.value: "pipeline",
    ReviewReason.PIPELINE_FAILURE.value: "pipeline",
    ReviewReason.SYSTEM_ERROR.value: "pipeline",
    ReviewReason.NONE.value: "none",
}


class ExplainabilityService:
    """Service generating structured, factual explanations for document automation decisions."""

    @classmethod
    def get_category_for_code(cls, code: str) -> str:
        """Map canonical reason code to its semantic category."""
        return REASON_CATEGORIES.get(code, "validation")

    @classmethod
    def explain_document(cls, doc: Document) -> DecisionExplanation:
        """Derive the complete structured decision explanation for a document.

        Args:
            doc: Production Document entity with relationships loaded.

        Returns:
            DecisionExplanation DTO.
        """
        # Determine actual production decision
        decision = doc.decision or (
            AutomationDecision.AUTOMATIC.value
            if doc.status == DocumentStatus.COMPLETED_AUTOMATIC.value
            else AutomationDecision.MANUAL_REVIEW.value
            if doc.status in (DocumentStatus.MANUAL_REVIEW.value, DocumentStatus.COMPLETED_MANUAL.value)
            else "pending"
            if doc.status in (DocumentStatus.UPLOADED.value, DocumentStatus.PROCESSING.value)
            else "rejected"
        )

        # 1. Automatic Acceptance
        if decision == AutomationDecision.AUTOMATIC.value:
            return DecisionExplanation(
                decision=AutomationDecision.AUTOMATIC.value,
                primary_reason=None,
                reasons=[],
                positive_criteria=[
                    "Quality gate passed (image quality score met minimum threshold)",
                    "Domain schema validation succeeded without errors",
                    "All mandatory fields extracted with confidence above required thresholds",
                ],
                automatic_eligible=True,
                summary="Document met all criteria for automatic straight-through processing.",
            )

        # 2. Pending processing
        if decision == "pending":
            return DecisionExplanation(
                decision="pending",
                primary_reason=None,
                reasons=[],
                positive_criteria=[],
                automatic_eligible=False,
                summary="Document processing in progress; automation decision pending.",
            )

        # 3. Manual Review / Rejection
        # Check if document was subsequently reviewed/approved (COMPLETED_MANUAL)
        # In COMPLETED_MANUAL, doc.review_reason is reset to NONE and doc.review_details_json is None.
        # We MUST look up the historical DECISION_EVALUATED audit event rather than current mutated state.
        historical_eval_event = None
        if hasattr(doc, "audit_events") and doc.audit_events:
            for ev in doc.audit_events:
                if ev.action == AuditAction.DECISION_EVALUATED.value:
                    historical_eval_event = ev
                    break

        primary_reason: Optional[str] = None
        historical_details: Optional[Dict[str, Any]] = None
        historical_explanation_text: Optional[str] = None

        if doc.status == DocumentStatus.COMPLETED_MANUAL.value and historical_eval_event:
            meta = historical_eval_event.metadata_json or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            ev_reason = meta.get("reason")
            if ev_reason and ev_reason != ReviewReason.NONE.value:
                primary_reason = ev_reason
            historical_details = meta.get("review_details")
            historical_explanation_text = meta.get("explanation")
        else:
            if doc.review_reason and doc.review_reason != ReviewReason.NONE.value:
                primary_reason = doc.review_reason
            elif historical_eval_event:
                meta = historical_eval_event.metadata_json or {}
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except Exception:
                        meta = {}
                ev_reason = meta.get("reason")
                if ev_reason and ev_reason != ReviewReason.NONE.value:
                    primary_reason = ev_reason
                historical_details = meta.get("review_details")
                historical_explanation_text = meta.get("explanation")

        # Source of truth for structured reasons:
        # 1. doc.review_details_json (if currently present on document)
        # 2. historical_details from DECISION_EVALUATED audit event
        active_details = doc.review_details_json or historical_details
        if isinstance(active_details, str):
            try:
                active_details = json.loads(active_details)
            except Exception:
                active_details = None

        reasons_list: List[DecisionReason] = []

        if active_details and isinstance(active_details, dict):
            # Parse structured reasons list
            raw_reasons = active_details.get("reasons") or []
            for r in raw_reasons:
                if isinstance(r, dict):
                    code = r.get("code") or primary_reason or ReviewReason.VALIDATION_FAILED.value
                    cat = cls.get_category_for_code(code)
                    msg = r.get("message") or f"{code.replace('_', ' ').capitalize()}."
                    field_name = r.get("field")
                    conf = r.get("confidence")
                    thresh = r.get("threshold")
                    reasons_list.append(
                        DecisionReason(
                            code=code,
                            category=cat,
                            message=msg,
                            blocking=True,
                            field=field_name,
                            observed_value=conf,
                            threshold=thresh,
                        )
                    )
                elif isinstance(r, str):
                    parts = r.split(":")
                    raw_code = parts[0]
                    field_name = parts[1] if len(parts) > 1 else None
                    code = ReviewReason.LOW_CONFIDENCE.value if raw_code == "low_field_confidence" else raw_code
                    cat = cls.get_category_for_code(code)
                    field_info = (
                        active_details.get("fields", {}).get(field_name, {})
                        if (field_name and isinstance(active_details.get("fields"), dict))
                        else {}
                    )
                    conf = field_info.get("confidence") if isinstance(field_info, dict) else None
                    thresh = field_info.get("threshold") if isinstance(field_info, dict) else None
                    msg = (
                        field_info.get("reason")
                        if isinstance(field_info, dict) and field_info.get("reason")
                        else f"Condition {code.replace('_', ' ')} triggered"
                        + (f" on field '{field_name}'." if field_name else ".")
                    )
                    reasons_list.append(
                        DecisionReason(
                            code=code,
                            category=cat,
                            message=msg,
                            blocking=True,
                            field=field_name,
                            observed_value=conf,
                            threshold=thresh,
                        )
                    )


            # Also check validation_errors list if reasons was empty
            if not reasons_list and active_details.get("validation_errors"):
                for err in active_details.get("validation_errors", []):
                    if isinstance(err, dict):
                        code = err.get("code") or ReviewReason.VALIDATION_FAILED.value
                        reasons_list.append(
                            DecisionReason(
                                code=code,
                                category=cls.get_category_for_code(code),
                                message=err.get("message", "Validation error."),
                                blocking=True,
                                field=err.get("field"),
                                observed_value=None,
                                threshold=None,
                            )
                        )

        # Fallback for legacy documents lacking review_details:
        # Use strictly the persisted primary_reason and historical_explanation_text.
        # DO NOT reconstruct or recalculate missing reasons from current document fields!
        if not reasons_list and primary_reason and primary_reason != ReviewReason.NONE.value:
            cat = cls.get_category_for_code(primary_reason)
            msg = historical_explanation_text or f"Document routed to manual review: {primary_reason.replace('_', ' ')}."
            reasons_list.append(
                DecisionReason(
                    code=primary_reason,
                    category=cat,
                    message=msg,
                    blocking=True,
                    field=None,
                    observed_value=None,
                    threshold=None,
                )
            )

        # Synthesize concise explanation summary
        if reasons_list:
            if primary_reason:
                summary = f"Manual review required due to {primary_reason.replace('_', ' ')} ({len(reasons_list)} blocking condition(s))."
            else:
                summary = f"Manual review required ({len(reasons_list)} blocking condition(s))."
        elif primary_reason:
            summary = f"Manual review required due to {primary_reason.replace('_', ' ')}."
        else:
            summary = "Manual review required by operational policy."

        return DecisionExplanation(
            decision=decision,
            primary_reason=primary_reason,
            reasons=reasons_list,
            automatic_eligible=False,
            summary=summary,
        )
