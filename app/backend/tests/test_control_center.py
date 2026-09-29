"""Focused backend tests for MVP-8: Document AI Quality & Performance Control Center."""

from datetime import datetime, timedelta, timezone
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import AuditAction, AutomationDecision, DocumentStatus, DocumentType, ReviewReason
from app.models.audit import AuditEvent
from app.models.document import Document
from app.models.field import ExtractedField
from app.models.page import DocumentPage
from app.models.quality import DocumentQuality
from app.services.analytics_service import AnalyticsService


@pytest.mark.asyncio
async def test_control_center_empty_dataset_and_division_safety(client: AsyncClient, db_session: AsyncSession):
    """1. Empty dataset, 5. Zero-safe percentages: returns valid payload without division by zero errors."""
    resp = await client.get("/api/v1/analytics/control-center")
    assert resp.status_code == 200
    data = resp.json()

    # Period
    assert data["period"]["period"] == "30d"

    # Summary
    summary = data["summary"]
    assert summary["total_documents"] == 0
    assert summary["processed_count"] == 0
    assert summary["automatic_count"] == 0
    assert summary["manual_review_count"] == 0
    assert summary["completed_manual_count"] == 0
    assert summary["error_count"] == 0
    assert summary["automation_rate"] == 0.0
    assert summary["manual_review_rate"] == 0.0
    assert summary["correction_rate"] == 0.0
    assert summary["no_change_review_rate"] == 0.0
    assert summary["average_confidence"] == 0.0
    assert summary["average_processing_time_ms"] == 0.0
    assert summary["average_review_time_ms"] == 0.0

    # Queue (Real-time live state)
    queue = data["queue"]
    assert queue["manual_review"] == 0
    assert queue["processing"] == 0
    assert queue["errors"] == 0

    # Funnel
    funnel = data["funnel"]
    assert funnel["processed"] == 0
    assert funnel["automatic"] == 0
    assert funnel["manual_review"] == 0
    assert funnel["completed_manual_review"] == 0
    assert funnel["with_corrections"] == 0
    assert funnel["without_corrections"] == 0

    # Quality
    quality = data["quality"]
    assert quality["average_quality_score"] is None
    assert quality["average_document_confidence"] is None
    assert quality["average_ocr_confidence"] is None
    assert quality["average_kie_confidence"] is None
    assert quality["validation_pass_rate"] is None
    assert quality["total_documents_validated"] == 0
    assert quality["passed_validation_count"] == 0
    assert quality["failed_validation_count"] == 0

    # Lists
    assert data["review_reasons"] == []
    assert len(data["trends"]) >= 30  # Continuous daily calendar timeline
    assert all(t["processed_count"] == 0 and t["automation_rate"] == 0.0 for t in data["trends"])
    assert data["recent_documents"] == []


@pytest.mark.asyncio
async def test_control_center_populated_dataset_and_canonical_consistency(
    client: AsyncClient, db_session: AsyncSession
):
    """2. Populated dataset, 11-14. Consistency checks with underlying canonical services."""
    now = datetime.now(timezone.utc)

    # 1. Automatic doc
    doc1 = Document(
        id="cc_doc_auto_1",
        filename="rec_auto.jpg",
        file_path="/tmp/rec_auto.jpg",
        file_size_bytes=1000,
        mime_type="image/jpeg",
        document_type=DocumentType.RECEIPT.value,
        status=DocumentStatus.COMPLETED_AUTOMATIC.value,
        decision=AutomationDecision.AUTOMATIC.value,
        confidence=0.95,
        review_reason=ReviewReason.NONE.value,
        processing_finished_at=now - timedelta(hours=2),
        processing_duration_ms=1200.0,
        created_at=now - timedelta(hours=2),
        updated_at=now - timedelta(hours=2),
    )
    # 2. Completed manual review doc with corrections
    doc2 = Document(
        id="cc_doc_rev_corr",
        filename="rec_rev_corr.jpg",
        file_path="/tmp/rec_rev_corr.jpg",
        file_size_bytes=1200,
        mime_type="image/jpeg",
        document_type=DocumentType.RECEIPT.value,
        status=DocumentStatus.COMPLETED_MANUAL.value,
        decision=AutomationDecision.MANUAL_REVIEW.value,
        confidence=0.82,
        review_reason=ReviewReason.NONE.value,
        processing_finished_at=now - timedelta(hours=3),
        processing_duration_ms=1500.0,
        review_finished_at=now - timedelta(hours=1),
        review_duration_ms=25000.0,
        created_at=now - timedelta(hours=3),
        updated_at=now - timedelta(hours=1),
    )
    # 3. Completed manual review doc WITHOUT corrections (no-change)
    doc3 = Document(
        id="cc_doc_rev_clean",
        filename="rec_rev_clean.jpg",
        file_path="/tmp/rec_rev_clean.jpg",
        file_size_bytes=1100,
        mime_type="image/jpeg",
        document_type=DocumentType.RECEIPT.value,
        status=DocumentStatus.COMPLETED_MANUAL.value,
        decision=AutomationDecision.MANUAL_REVIEW.value,
        confidence=0.88,
        review_reason=ReviewReason.NONE.value,
        processing_finished_at=now - timedelta(hours=4),
        processing_duration_ms=1400.0,
        review_finished_at=now - timedelta(hours=2),
        review_duration_ms=18000.0,
        created_at=now - timedelta(hours=4),
        updated_at=now - timedelta(hours=2),
    )
    # 4. Currently in Manual Review queue (live backlog)
    doc4 = Document(
        id="cc_doc_in_queue",
        filename="rec_queue.jpg",
        file_path="/tmp/rec_queue.jpg",
        file_size_bytes=1300,
        mime_type="image/jpeg",
        document_type=DocumentType.RECEIPT.value,
        status=DocumentStatus.MANUAL_REVIEW.value,
        decision=AutomationDecision.MANUAL_REVIEW.value,
        confidence=0.75,
        review_reason=ReviewReason.LOW_CONFIDENCE.value,
        processing_finished_at=now - timedelta(minutes=30),
        processing_duration_ms=1600.0,
        created_at=now - timedelta(minutes=30),
        updated_at=now - timedelta(minutes=30),
    )

    db_session.add_all([doc1, doc2, doc3, doc4])

    # Audit events for decisions
    ev1 = AuditEvent(
        id="cc_ev_dec_1",
        document_id=doc1.id,
        actor="system",
        action=AuditAction.DECISION_EVALUATED.value,
        created_at=doc1.processing_finished_at,
        metadata_json={"decision": "automatic", "reason": "none"},
    )
    ev2 = AuditEvent(
        id="cc_ev_dec_2",
        document_id=doc2.id,
        actor="system",
        action=AuditAction.DECISION_EVALUATED.value,
        created_at=doc2.processing_finished_at,
        metadata_json={
            "decision": "manual_review",
            "reason": "low_confidence",
            "review_details": {
                "reasons": [{"code": "low_confidence", "field": "total", "message": "Low confidence"}]
            },
        },
    )
    ev3 = AuditEvent(
        id="cc_ev_dec_3",
        document_id=doc3.id,
        actor="system",
        action=AuditAction.DECISION_EVALUATED.value,
        created_at=doc3.processing_finished_at,
        metadata_json={
            "decision": "manual_review",
            "reason": "low_confidence",
            "review_details": {
                "reasons": [{"code": "low_confidence", "field": "date", "message": "Low confidence"}]
            },
        },
    )
    ev4 = AuditEvent(
        id="cc_ev_dec_4",
        document_id=doc4.id,
        actor="system",
        action=AuditAction.DECISION_EVALUATED.value,
        created_at=doc4.processing_finished_at,
        metadata_json={
            "decision": "manual_review",
            "reason": "low_confidence",
            "review_details": {
                "reasons": [{"code": "low_confidence", "field": "company", "message": "Low confidence"}]
            },
        },
    )

    # Correction event for doc2
    corr_ev = AuditEvent(
        id="cc_ev_corr_1",
        document_id=doc2.id,
        actor="operator",
        action=AuditAction.FIELD_CORRECTED.value,
        field_name="total",
        old_value="12.00",
        new_value="15.00",
        created_at=now - timedelta(hours=1),
    )

    db_session.add_all([ev1, ev2, ev3, ev4, corr_ev])
    await db_session.commit()

    # Compare Control Center output with direct service calls
    service = AnalyticsService(db_session)
    overview_expected = await service.get_overview(period="30d")
    feedback_expected = await service.get_production_feedback(period="30d")
    reasons_expected = await service.get_decision_reasons_analytics(period="30d")
    queue_expected = await service.get_queue()

    resp = await client.get("/api/v1/analytics/control-center?period=30d")
    assert resp.status_code == 200
    data = resp.json()

    # Consistency 11: Automation rate equals canonical overview automation rate
    assert data["summary"]["automation_rate"] == overview_expected.kpis.automation_rate
    assert data["summary"]["processed_count"] == overview_expected.kpis.processed_count
    assert data["summary"]["total_documents"] == overview_expected.kpis.total_documents

    # Consistency 12: Correction rate equals MVP-7 feedback correction rate
    assert data["summary"]["correction_rate"] == feedback_expected.summary.correction_rate
    assert data["summary"]["no_change_review_rate"] == feedback_expected.summary.no_change_review_rate
    # Exactly 1 out of 2 completed reviews was corrected = 50.0%
    assert data["summary"]["correction_rate"] == 50.0
    assert data["summary"]["no_change_review_rate"] == 50.0

    # Consistency 13: Review reasons match canonical DecisionReasonsAnalytics
    assert len(data["review_reasons"]) == len(reasons_expected.by_reason)
    if data["review_reasons"]:
        assert data["review_reasons"][0]["code"] == reasons_expected.by_reason[0].code

    # Consistency 14: Queue matches existing queue analytics
    assert data["queue"]["manual_review"] == queue_expected.manual_review
    assert data["queue"]["manual_review"] == 1  # doc4 is in manual_review
    assert data["queue"]["processing"] == queue_expected.processing

    # Normalized confidence in 0..1 scale
    assert 0.0 <= data["summary"]["average_confidence"] <= 1.0


@pytest.mark.asyncio
async def test_control_center_quality_and_validation_signals(client: AsyncClient, db_session: AsyncSession):
    """8. Quality overview: tests aggregation of quality, OCR tokens, KIE fields, and document validation pass rate."""
    now = datetime.now(timezone.utc)

    # Document 1: Valid receipt with quality, page tokens, and extracted fields
    doc1 = Document(
        id="cc_q_doc_1",
        filename="q_rec1.jpg",
        file_path="/tmp/q_rec1.jpg",
        file_size_bytes=1000,
        mime_type="image/jpeg",
        document_type=DocumentType.RECEIPT.value,
        status=DocumentStatus.COMPLETED_AUTOMATIC.value,
        decision=AutomationDecision.AUTOMATIC.value,
        confidence=0.92,
        processing_finished_at=now - timedelta(hours=1),
        created_at=now - timedelta(hours=1),
        updated_at=now - timedelta(hours=1),
    )
    # Document 2: Failed validation receipt (missing required field)
    doc2 = Document(
        id="cc_q_doc_2",
        filename="q_rec2.jpg",
        file_path="/tmp/q_rec2.jpg",
        file_size_bytes=1000,
        mime_type="image/jpeg",
        document_type=DocumentType.RECEIPT.value,
        status=DocumentStatus.MANUAL_REVIEW.value,
        decision=AutomationDecision.MANUAL_REVIEW.value,
        review_reason=ReviewReason.MISSING_REQUIRED_FIELD.value,
        confidence=0.70,
        processing_finished_at=now - timedelta(hours=2),
        created_at=now - timedelta(hours=2),
        updated_at=now - timedelta(hours=2),
    )

    db_session.add_all([doc1, doc2])

    # Quality records
    q1 = DocumentQuality(
        id="cc_qual_1",
        document_id=doc1.id,
        quality_score=0.95,
        blur_score=250.0,
        contrast_score=60.0,
        noise_score=1.5,
        profile_summary="Normal",
    )
    q2 = DocumentQuality(
        id="cc_qual_2",
        document_id=doc2.id,
        quality_score=0.85,
        blur_score=180.0,
        contrast_score=45.0,
        noise_score=3.5,
        profile_summary="Slight Blur",
    )
    db_session.add_all([q1, q2])

    # Page with OCR tokens
    page1 = DocumentPage(
        id="cc_page_1",
        document_id=doc1.id,
        page_number=1,
        width=800,
        height=1000,
        image_path="/tmp/q_rec1.jpg",
        tokens_json=[
            {"text": "Total", "confidence": 0.98},
            {"text": "$10.00", "confidence": 0.92},
        ],
    )
    db_session.add(page1)

    # Extracted fields
    f1 = ExtractedField(
        id="cc_f_1",
        document_id=doc1.id,
        field_name="total",
        value="10.00",
        confidence=0.94,
        validation_status="VALID",
    )
    f2 = ExtractedField(
        id="cc_f_2",
        document_id=doc1.id,
        field_name="date",
        value="2026-09-28",
        confidence=0.90,
        validation_status="VALID",
    )
    db_session.add_all([f1, f2])

    await db_session.commit()

    resp = await client.get("/api/v1/analytics/control-center?period=30d")
    assert resp.status_code == 200
    data = resp.json()

    quality = data["quality"]
    # Image Quality avg: (0.95 + 0.85) / 2 = 0.90
    assert quality["average_quality_score"] == 0.9
    # OCR token conf: (0.98 + 0.92) / 2 = 0.95
    assert quality["average_ocr_confidence"] == 0.95
    # KIE field conf: (0.94 + 0.90) / 2 = 0.92
    assert quality["average_kie_confidence"] == 0.92
    # Document Validation: 2 documents reached decision; doc1 passed, doc2 failed with missing_required_field
    assert quality["total_documents_validated"] == 2
    assert quality["passed_validation_count"] == 1
    assert quality["failed_validation_count"] == 1
    assert quality["validation_pass_rate"] == 0.5


@pytest.mark.asyncio
async def test_control_center_recent_documents_returned(client: AsyncClient, db_session: AsyncSession):
    """7. Recent documents: returns top recent documents ordered by updated_at descending."""
    now = datetime.now(timezone.utc)
    for i in range(1, 6):
        doc = Document(
            id=f"cc_recent_{i}",
            filename=f"doc_{i}.pdf",
            file_path=f"/tmp/doc_{i}.pdf",
            file_size_bytes=1000 * i,
            mime_type="application/pdf",
            document_type=DocumentType.RECEIPT.value,
            status=DocumentStatus.COMPLETED_AUTOMATIC.value,
            decision=AutomationDecision.AUTOMATIC.value,
            confidence=0.85 + (i * 0.02),
            processing_finished_at=now - timedelta(minutes=i * 10),
            created_at=now - timedelta(minutes=i * 10),
            updated_at=now - timedelta(minutes=i * 10),
        )
        db_session.add(doc)
    await db_session.commit()

    resp = await client.get("/api/v1/analytics/control-center")
    assert resp.status_code == 200
    data = resp.json()

    recent = data["recent_documents"]
    assert len(recent) >= 5
    # Most recently updated document first
    assert recent[0]["id"] == "cc_recent_1"


@pytest.mark.asyncio
async def test_control_center_period_filtering_and_custom_bounds(client: AsyncClient, db_session: AsyncSession):
    """3. Period filtering, 4. Custom period, 6. Current queue remains real-time live backlog."""
    now = datetime.now(timezone.utc)

    # Old doc processed 45 days ago
    doc_old = Document(
        id="cc_doc_old",
        filename="old.jpg",
        file_path="/tmp/old.jpg",
        file_size_bytes=1000,
        mime_type="image/jpeg",
        document_type=DocumentType.RECEIPT.value,
        status=DocumentStatus.COMPLETED_AUTOMATIC.value,
        decision=AutomationDecision.AUTOMATIC.value,
        confidence=0.90,
        processing_finished_at=now - timedelta(days=45),
        created_at=now - timedelta(days=45),
        updated_at=now - timedelta(days=45),
    )
    # Recent doc processed today
    doc_new = Document(
        id="cc_doc_new",
        filename="new.jpg",
        file_path="/tmp/new.jpg",
        file_size_bytes=1000,
        mime_type="image/jpeg",
        document_type=DocumentType.RECEIPT.value,
        status=DocumentStatus.COMPLETED_AUTOMATIC.value,
        decision=AutomationDecision.AUTOMATIC.value,
        confidence=0.95,
        processing_finished_at=now - timedelta(hours=2),
        created_at=now - timedelta(hours=2),
        updated_at=now - timedelta(hours=2),
    )
    # Doc currently in queue
    doc_queue = Document(
        id="cc_doc_queue_live",
        filename="queue.jpg",
        file_path="/tmp/queue.jpg",
        file_size_bytes=1000,
        mime_type="image/jpeg",
        document_type=DocumentType.RECEIPT.value,
        status=DocumentStatus.MANUAL_REVIEW.value,
        created_at=now,
        updated_at=now,
    )
    db_session.add_all([doc_old, doc_new, doc_queue])
    await db_session.commit()

    # Query 7d: should see only doc_new in processed count
    r7d = await client.get("/api/v1/analytics/control-center?period=7d")
    assert r7d.status_code == 200
    d7d = r7d.json()
    assert d7d["summary"]["processed_count"] == 1
    # Live queue must still report the active manual review doc
    assert d7d["queue"]["manual_review"] == 1

    # Query custom date range covering 40-50 days ago
    dt_from = (now - timedelta(days=50)).strftime("%Y-%m-%dT00:00:00Z")
    dt_to = (now - timedelta(days=40)).strftime("%Y-%m-%dT23:59:59Z")
    rcustom = await client.get(f"/api/v1/analytics/control-center?period=custom&from={dt_from}&to={dt_to}")
    assert rcustom.status_code == 200
    dcustom = rcustom.json()
    assert dcustom["summary"]["processed_count"] == 1
    # Live queue is STILL live current state!
    assert dcustom["queue"]["manual_review"] == 1
