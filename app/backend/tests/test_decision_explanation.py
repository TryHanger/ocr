"""Tests for Decision & Automation Explainability (MVP-6)."""

from datetime import datetime, timezone
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import AutomationDecision, DocumentStatus, DocumentType, ReviewReason
from app.models.audit import AuditEvent
from app.models.document import Document
from app.schemas.kie import NormalizedFieldDTO
from app.services.decision_engine import DecisionEngine
from app.services.explainability_service import ExplainabilityService


def test_automatic_decision_explanation():
    """Verify explain_document for an automatically approved document."""
    doc = Document(
        id="doc_auto_1",
        filename="clean_receipt.jpg",
        document_type=DocumentType.RECEIPT.value,
        status=DocumentStatus.COMPLETED_AUTOMATIC.value,
        decision=AutomationDecision.AUTOMATIC.value,
        review_reason=ReviewReason.NONE.value,
        review_details_json=None,
        audit_events=[],
    )

    explanation = ExplainabilityService.explain_document(doc)
    assert explanation.decision == "automatic"
    assert explanation.automatic_eligible is True
    assert explanation.primary_reason is None
    assert len(explanation.reasons) == 0
    assert "criteria for automatic" in explanation.summary
    assert len(explanation.positive_criteria) > 0


def test_manual_review_decision_explanation():
    """Verify field-level evidence and low confidence explanation."""
    details = {
        "reasons": [
            {
                "code": "low_confidence",
                "field": "total",
                "confidence": 0.60,
                "threshold": 0.95,
                "message": "Field 'total' confidence 0.60 below threshold 0.95",
            }
        ],
        "fields": {
            "total": {
                "status": "review",
                "confidence": 0.60,
                "threshold": 0.95,
                "reason": "field confidence 0.60 below threshold 0.95",
            }
        },
    }
    doc = Document(
        id="doc_manual_1",
        filename="faded_receipt.jpg",
        document_type=DocumentType.RECEIPT.value,
        status=DocumentStatus.MANUAL_REVIEW.value,
        decision=AutomationDecision.MANUAL_REVIEW.value,
        review_reason=ReviewReason.LOW_CONFIDENCE.value,
        review_details_json=details,
        audit_events=[],
    )

    explanation = ExplainabilityService.explain_document(doc)
    assert explanation.decision == "manual_review"
    assert explanation.automatic_eligible is False
    assert explanation.primary_reason == "low_confidence"
    assert len(explanation.reasons) >= 1
    reason = explanation.reasons[0]
    assert reason.field == "total"
    assert reason.observed_value == 0.60
    assert reason.threshold == 0.95
    assert reason.category == "confidence"


def test_multiple_blocking_reasons_preserved():
    """Verify that multiple failing conditions (validation + confidence) are all preserved."""
    details = {
        "reasons": [
            {
                "code": "missing_required_field",
                "field": "date",
                "confidence": None,
                "threshold": None,
                "message": "Missing required field: date",
            },
            {
                "code": "low_confidence",
                "field": "total",
                "confidence": 0.55,
                "threshold": 0.95,
                "message": "Field 'total' confidence 0.55 below threshold 0.95",
            }
        ],
        "validation_errors": [
            {"field": "date", "issue": "Missing required field: date"}
        ],
        "fields": {
            "total": {
                "status": "review",
                "confidence": 0.55,
                "threshold": 0.95,
                "reason": "field confidence 0.55 below threshold 0.95",
            }
        },
    }
    doc = Document(
        id="doc_multi_1",
        filename="bad_receipt.jpg",
        document_type=DocumentType.RECEIPT.value,
        status=DocumentStatus.MANUAL_REVIEW.value,
        decision=AutomationDecision.MANUAL_REVIEW.value,
        review_reason=ReviewReason.MISSING_REQUIRED_FIELD.value,
        review_details_json=details,
        audit_events=[],
    )

    explanation = ExplainabilityService.explain_document(doc)
    assert explanation.decision == "manual_review"
    assert explanation.primary_reason == "missing_required_field"
    # Should have both missing_required_field and low_confidence reasons
    codes = [r.code for r in explanation.reasons]
    assert any("missing" in c for c in codes)
    assert any("confidence" in c for c in codes)


def test_historical_decision_preserved_after_approval():
    """When a document is reviewed and approved (COMPLETED_MANUAL), its review_details_json

    is cleared from the document row, but explainability MUST read from the immutable
    DECISION_EVALUATED audit event rather than attempting to reconstruct or showing empty.
    """
    historical_review_details = {
        "reasons": [
            {
                "code": "low_confidence",
                "field": "total",
                "confidence": 0.62,
                "threshold": 0.95,
                "message": "Field 'total' confidence 0.62 below threshold 0.95",
            }
        ],
        "fields": {
            "total": {
                "status": "review",
                "confidence": 0.62,
                "threshold": 0.95,
                "reason": "field confidence 0.62 below threshold 0.95",
            }
        },
    }
    audit_event = AuditEvent(
        id="evt_1",
        document_id="doc_completed_1",
        actor="system",
        action="decision_evaluated",
        created_at=datetime(2026, 9, 28, 12, 0, 0, tzinfo=timezone.utc),
        metadata_json={
            "decision": "manual_review",
            "reason": "low_confidence",
            "explanation": "Document routed to manual review due to low confidence.",
            "review_details": historical_review_details,
        },
    )

    doc = Document(
        id="doc_completed_1",
        filename="approved_receipt.jpg",
        document_type=DocumentType.RECEIPT.value,
        status=DocumentStatus.COMPLETED_MANUAL.value,
        decision=AutomationDecision.MANUAL_REVIEW.value,
        review_reason=ReviewReason.NONE.value,  # Cleared upon review approval
        review_details_json=None,  # Cleared upon review approval
        audit_events=[audit_event],
    )

    explanation = ExplainabilityService.explain_document(doc)
    # Must retrieve the historical decision details from the audit event!
    assert explanation.decision == "manual_review"
    assert explanation.primary_reason == "low_confidence"
    assert len(explanation.reasons) == 1
    assert explanation.reasons[0].field == "total"
    assert explanation.reasons[0].observed_value == 0.62


def test_legacy_audit_event_fallback():
    """Legacy DECISION_EVALUATED event without review_details must not fabricate reasons."""
    audit_event = AuditEvent(
        id="evt_legacy",
        document_id="doc_legacy_1",
        actor="system",
        action="decision_evaluated",
        created_at=datetime(2026, 9, 28, 10, 0, 0, tzinfo=timezone.utc),
        metadata_json={
            "decision": "manual_review",
            "reason": "low_confidence",
            "explanation": "Document routed to manual review (legacy)",
        },
    )

    doc = Document(
        id="doc_legacy_1",
        filename="legacy.jpg",
        document_type=DocumentType.RECEIPT.value,
        status=DocumentStatus.COMPLETED_MANUAL.value,
        decision=AutomationDecision.MANUAL_REVIEW.value,
        review_reason=ReviewReason.NONE.value,
        review_details_json=None,
        audit_events=[audit_event],
    )

    explanation = ExplainabilityService.explain_document(doc)
    assert explanation.decision == "manual_review"
    assert explanation.primary_reason == "low_confidence"
    assert len(explanation.reasons) == 1
    assert explanation.reasons[0].code == "low_confidence"
    # Should not fabricate field or thresholds when absent in legacy records
    assert explanation.reasons[0].field is None
    assert explanation.reasons[0].threshold is None


def test_no_threshold_duplication():
    """Decision explanation thresholds must match production policy from DecisionEngine."""
    policy = DecisionEngine.get_policy("receipt")
    expected_threshold = policy["thresholds"]["total"]

    fields = {
        "company": NormalizedFieldDTO(field_name="company", value="ABC SDN BHD", confidence=0.95),
        "date": NormalizedFieldDTO(field_name="date", value="2026-09-28", confidence=0.92),
        "total": NormalizedFieldDTO(field_name="total", value="150.00", confidence=0.60),
        "address": NormalizedFieldDTO(field_name="address", value="Kuala Lumpur", confidence=0.88),
    }
    decision, reason, exp, details = DecisionEngine.evaluate("receipt", fields, None, 0.70)
    assert decision == AutomationDecision.MANUAL_REVIEW
    assert details["fields"]["total"]["threshold"] == expected_threshold

    doc = Document(
        id="doc_test_policy",
        filename="receipt.jpg",
        document_type=DocumentType.RECEIPT.value,
        status=DocumentStatus.MANUAL_REVIEW.value,
        decision=decision.value,
        review_reason=reason.value,
        review_details_json=details,
        audit_events=[],
    )
    explanation = ExplainabilityService.explain_document(doc)
    reason_obj = next(r for r in explanation.reasons if r.field == "total")
    assert reason_obj.threshold == expected_threshold


def test_no_research_evidence_in_decision_reasons():
    """Explainability must NOT include research benchmark artifacts or B1/B2 metrics in decision reasons."""
    doc = Document(
        id="doc_clean",
        filename="clean.jpg",
        document_type=DocumentType.RECEIPT.value,
        status=DocumentStatus.COMPLETED_AUTOMATIC.value,
        decision=AutomationDecision.AUTOMATIC.value,
        review_reason=ReviewReason.NONE.value,
        review_details_json=None,
        audit_events=[],
    )
    explanation = ExplainabilityService.explain_document(doc)
    explanation_dict = explanation.model_dump()
    reasons_text = str(explanation_dict).lower()
    assert "b1_" not in reasons_text
    assert "b2_" not in reasons_text
    assert "degradation" not in reasons_text
    assert "baseline" not in reasons_text


@pytest.mark.asyncio
async def test_decision_reasons_analytics_endpoint(client: AsyncClient, db_session: AsyncSession):
    """Test GET /api/v1/analytics/decision-reasons with empty and populated DB."""
    # 1. Test empty dataset
    resp = await client.get("/api/v1/analytics/decision-reasons")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_decisions"] == 0
    assert data["manual_review_count"] == 0
    assert data["automatic_count"] == 0
    assert data["manual_review_rate"] == 0.0
    assert data["by_reason"] == []
    assert data["by_category"] == []
    assert data["by_field"] == []

    # 2. Add sample DECISION_EVALUATED audit events
    now = datetime.now(timezone.utc)
    ev1 = AuditEvent(
        id="ev_1",
        document_id="d1",
        actor="system",
        action="decision_evaluated",
        created_at=now,
        metadata_json={
            "decision": "automatic",
            "reason": None,
            "review_details": None,
        },
    )
    ev2 = AuditEvent(
        id="ev_2",
        document_id="d2",
        actor="system",
        action="decision_evaluated",
        created_at=now,
        metadata_json={
            "decision": "manual_review",
            "reason": "low_confidence",
            "review_details": {
                "reasons": [
                    {
                        "code": "low_confidence",
                        "field": "total",
                        "confidence": 0.7,
                        "threshold": 0.95,
                        "message": "Field total low confidence",
                    }
                ],
                "fields": {
                    "total": {"status": "review", "confidence": 0.7}
                },
            },
        },
    )
    ev3 = AuditEvent(
        id="ev_3",
        document_id="d3",
        actor="system",
        action="decision_evaluated",
        created_at=now,
        metadata_json={
            "decision": "manual_review",
            "reason": "missing_required_field",
            "review_details": {
                "reasons": [
                    {
                        "code": "missing_required_field",
                        "field": "date",
                        "confidence": None,
                        "threshold": None,
                        "message": "Missing date",
                    }
                ],
                "validation_errors": [{"field": "date"}],
            },
        },
    )
    db_session.add_all([ev1, ev2, ev3])
    await db_session.commit()

    resp = await client.get("/api/v1/analytics/decision-reasons")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_decisions"] == 3
    assert data["automatic_count"] == 1
    assert data["manual_review_count"] == 2
    assert round(data["manual_review_rate"], 1) == 66.7
    assert len(data["by_reason"]) >= 2
    assert len(data["by_category"]) >= 2
    assert len(data["by_field"]) >= 2

    # Check category distribution
    cats = {c["category"]: c for c in data["by_category"]}
    assert "confidence" in cats
    assert "validation" in cats
    assert cats["confidence"]["count"] == 1
    assert cats["validation"]["count"] == 1

    # Check field distribution
    fields = {f["field"]: f for f in data["by_field"]}
    assert "total" in fields
    assert "date" in fields


@pytest.mark.asyncio
async def test_document_detail_api_decision_explanation(client: AsyncClient, db_session: AsyncSession):
    """Test GET /api/v1/documents/{id} includes decision_explanation in payload."""
    details = {
        "reasons": [
            {
                "code": "low_confidence",
                "field": "total",
                "confidence": 0.50,
                "threshold": 0.95,
                "message": "field confidence 0.50 below threshold 0.95",
            }
        ],
        "fields": {
            "total": {
                "status": "review",
                "confidence": 0.50,
                "threshold": 0.95,
                "reason": "field confidence 0.50 below threshold 0.95",
            }
        },
    }
    doc = Document(
        id="doc_api_test_1",
        filename="invoice_sample.pdf",
        file_path="/tmp/test.pdf",
        file_size_bytes=1024,
        mime_type="application/pdf",
        document_type=DocumentType.RECEIPT.value,
        status=DocumentStatus.MANUAL_REVIEW.value,
        decision=AutomationDecision.MANUAL_REVIEW.value,
        review_reason=ReviewReason.LOW_CONFIDENCE.value,
        review_details_json=details,
    )
    db_session.add(doc)
    await db_session.commit()

    resp = await client.get(f"/api/v1/documents/{doc.id}")
    assert resp.status_code == 200
    data = resp.json()
    assert "decision_explanation" in data
    explanation = data["decision_explanation"]
    assert explanation is not None
    assert explanation["decision"] == "manual_review"
    assert explanation["primary_reason"] == "low_confidence"
    assert len(explanation["reasons"]) == 1
    assert explanation["reasons"][0]["field"] == "total"
    assert explanation["reasons"][0]["observed_value"] == 0.50
    assert explanation["reasons"][0]["threshold"] == 0.95
