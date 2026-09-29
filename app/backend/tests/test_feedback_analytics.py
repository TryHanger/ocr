"""Focused backend tests for MVP-7: Feedback & Continuous Improvement."""

from datetime import datetime, timedelta, timezone
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import AuditAction, AutomationDecision, DocumentStatus, DocumentType, ReviewReason
from app.models.audit import AuditEvent
from app.models.document import Document
from app.services.analytics_service import AnalyticsService


@pytest.mark.asyncio
async def test_empty_dataset_and_division_safety(client: AsyncClient, db_session: AsyncSession):
    """1. No reviews, 18. Division safety: empty database returns zero for all rates without NaN/Inf."""
    resp = await client.get("/api/v1/analytics/feedback")
    assert resp.status_code == 200
    data = resp.json()

    # Summary
    summary = data["summary"]
    assert summary["total_correction_events"] == 0
    assert summary["documents_with_corrections"] == 0
    assert summary["completed_manual_review_documents"] == 0
    assert summary["correction_rate"] == 0.0
    assert summary["no_change_review_rate"] == 0.0
    assert summary["avg_corrections_per_corrected_document"] == 0.0

    # Funnel
    funnel = data["funnel"]
    assert funnel["processed"] == 0
    assert funnel["automatic"] == 0
    assert funnel["manual_review"] == 0
    assert funnel["completed_manual_review"] == 0
    assert funnel["with_corrections"] == 0
    assert funnel["without_corrections"] == 0
    assert funnel["automatic_rate"] == 0.0
    assert funnel["manual_review_rate"] == 0.0
    assert funnel["correction_rate"] == 0.0
    assert funnel["no_change_review_rate"] == 0.0

    # Lists
    assert data["by_field"] == []
    assert data["by_document_type"] == []
    assert data["by_reason"] == []
    assert data["by_reason_field"] == []


@pytest.mark.asyncio
async def test_reviews_without_corrections(client: AsyncClient, db_session: AsyncSession):
    """2. Reviews without corrections (100% no-change review rate)."""
    now = datetime.now(timezone.utc)
    doc1 = Document(
        id="doc_no_corr_1",
        filename="rec1.jpg",
        file_path="/tmp/rec1.jpg",
        file_size_bytes=1000,
        mime_type="image/jpeg",
        document_type=DocumentType.RECEIPT.value,
        status=DocumentStatus.COMPLETED_MANUAL.value,
        decision=AutomationDecision.MANUAL_REVIEW.value,
        review_reason=ReviewReason.NONE.value,
        processing_finished_at=now,
        review_finished_at=now,
    )
    # DECISION_EVALUATED audit event indicating it was routed for low confidence
    ev_dec = AuditEvent(
        id="ev_dec_1",
        document_id="doc_no_corr_1",
        actor="system",
        action=AuditAction.DECISION_EVALUATED.value,
        created_at=now,
        metadata_json={"decision": "manual_review", "reason": "low_confidence"},
    )
    db_session.add_all([doc1, ev_dec])
    await db_session.commit()

    resp = await client.get("/api/v1/analytics/feedback")
    assert resp.status_code == 200
    data = resp.json()

    summary = data["summary"]
    assert summary["total_correction_events"] == 0
    assert summary["documents_with_corrections"] == 0
    assert summary["completed_manual_review_documents"] == 1
    assert summary["correction_rate"] == 0.0
    assert summary["no_change_review_rate"] == 100.0
    assert summary["avg_corrections_per_corrected_document"] == 0.0

    funnel = data["funnel"]
    assert funnel["completed_manual_review"] == 1
    assert funnel["with_corrections"] == 0
    assert funnel["without_corrections"] == 1
    assert funnel["correction_rate"] == 0.0
    assert funnel["no_change_review_rate"] == 100.0


@pytest.mark.asyncio
async def test_mixed_reviews_field_aggregations_and_matrix(client: AsyncClient, db_session: AsyncSession):
    """Tests 3 (with corrections), 4 (mixed), 5 (multi corrections to same field),

    6 (multi fields on same doc), 7 (multi docs), 8 (multi doc types), 9 (zero corr type),
    10 (multi reasons), 11 (zero corr reason), 12/13 (reason x field matrix without duplicate counting).
    """
    now = datetime.now(timezone.utc)

    # Document A: Receipt, Low Confidence -> 2 corrections (total corrected twice!)
    doc_a = Document(
        id="doc_a",
        filename="rec_a.jpg",
        file_path="/tmp/a.jpg",
        file_size_bytes=1000,
        mime_type="image/jpeg",
        document_type=DocumentType.RECEIPT.value,
        status=DocumentStatus.COMPLETED_MANUAL.value,
        decision=AutomationDecision.MANUAL_REVIEW.value,
        processing_finished_at=now,
        review_finished_at=now,
    )
    ev_dec_a = AuditEvent(
        id="ev_dec_a",
        document_id="doc_a",
        actor="system",
        action=AuditAction.DECISION_EVALUATED.value,
        created_at=now,
        metadata_json={"decision": "manual_review", "reason": "low_confidence"},
    )
    # Correction 1 to total: 100 -> 110
    ev_c1 = AuditEvent(
        id="ev_c1",
        document_id="doc_a",
        actor="operator:alice",
        action=AuditAction.FIELD_CORRECTED.value,
        field_name="total",
        old_value="100.00",
        new_value="110.00",
        created_at=now,
    )
    # Correction 2 to total: 110 -> 120
    ev_c2 = AuditEvent(
        id="ev_c2",
        document_id="doc_a",
        actor="operator:alice",
        action=AuditAction.FIELD_CORRECTED.value,
        field_name="total",
        old_value="110.00",
        new_value="120.00",
        created_at=now,
    )

    # Document B: Invoice, Missing Required Field -> 2 corrections on different fields (date, company)
    doc_b = Document(
        id="doc_b",
        filename="inv_b.pdf",
        file_path="/tmp/b.pdf",
        file_size_bytes=2000,
        mime_type="application/pdf",
        document_type=DocumentType.INVOICE.value,
        status=DocumentStatus.COMPLETED_MANUAL.value,
        decision=AutomationDecision.MANUAL_REVIEW.value,
        processing_finished_at=now,
        review_finished_at=now,
    )
    ev_dec_b = AuditEvent(
        id="ev_dec_b",
        document_id="doc_b",
        actor="system",
        action=AuditAction.DECISION_EVALUATED.value,
        created_at=now,
        metadata_json={"decision": "manual_review", "reason": "missing_required_field"},
    )
    ev_c3 = AuditEvent(
        id="ev_c3",
        document_id="doc_b",
        actor="operator:bob",
        action=AuditAction.FIELD_CORRECTED.value,
        field_name="date",
        old_value=None,
        new_value="2026-09-28",
        created_at=now,
    )
    ev_c4 = AuditEvent(
        id="ev_c4",
        document_id="doc_b",
        actor="operator:bob",
        action=AuditAction.FIELD_CORRECTED.value,
        field_name="company",
        old_value="OLD CORP",
        new_value="NEW CORP",
        created_at=now,
    )

    # Document C: Act, Low Image Quality -> No corrections!
    doc_c = Document(
        id="doc_c",
        filename="act_c.pdf",
        file_path="/tmp/c.pdf",
        file_size_bytes=1500,
        mime_type="application/pdf",
        document_type=DocumentType.ACT.value,
        status=DocumentStatus.COMPLETED_MANUAL.value,
        decision=AutomationDecision.MANUAL_REVIEW.value,
        processing_finished_at=now,
        review_finished_at=now,
    )
    ev_dec_c = AuditEvent(
        id="ev_dec_c",
        document_id="doc_c",
        actor="system",
        action=AuditAction.DECISION_EVALUATED.value,
        created_at=now,
        metadata_json={"decision": "manual_review", "reason": "low_image_quality"},
    )

    # Document D: Contract -> Automatic pass (not in manual reviews)
    doc_d = Document(
        id="doc_d",
        filename="contract_d.pdf",
        file_path="/tmp/d.pdf",
        file_size_bytes=3000,
        mime_type="application/pdf",
        document_type=DocumentType.CONTRACT.value,
        status=DocumentStatus.COMPLETED_AUTOMATIC.value,
        decision=AutomationDecision.AUTOMATIC.value,
        processing_finished_at=now,
    )
    ev_dec_d = AuditEvent(
        id="ev_dec_d",
        document_id="doc_d",
        actor="system",
        action=AuditAction.DECISION_EVALUATED.value,
        created_at=now,
        metadata_json={"decision": "automatic", "reason": None},
    )

    db_session.add_all([
        doc_a, ev_dec_a, ev_c1, ev_c2,
        doc_b, ev_dec_b, ev_c3, ev_c4,
        doc_c, ev_dec_c,
        doc_d, ev_dec_d,
    ])
    await db_session.commit()

    resp = await client.get("/api/v1/analytics/feedback")
    assert resp.status_code == 200
    data = resp.json()

    # 1. Summary assertions
    # Total completed manual reviews: doc_a, doc_b, doc_c = 3
    # Documents with corrections: doc_a, doc_b = 2
    # Total correction events: 4 (2 on doc_a, 2 on doc_b)
    summary = data["summary"]
    assert summary["completed_manual_review_documents"] == 3
    assert summary["documents_with_corrections"] == 2
    assert summary["total_correction_events"] == 4
    # correction_rate: 2 / 3 * 100 = 66.7%
    assert summary["correction_rate"] == 66.7
    # no_change_review_rate: 1 / 3 * 100 = 33.3%
    assert summary["no_change_review_rate"] == 33.3
    # avg_corrections_per_corrected_document: 4 / 2 = 2.0
    assert summary["avg_corrections_per_corrected_document"] == 2.0

    # 2. Funnel assertions (Tests 14, 15, 16)
    funnel = data["funnel"]
    assert funnel["processed"] == 4  # doc_a, doc_b, doc_c, doc_d
    assert funnel["automatic"] == 1  # doc_d
    assert funnel["manual_review"] == 3  # doc_a, doc_b, doc_c
    assert funnel["completed_manual_review"] == 3
    assert funnel["with_corrections"] == 2
    assert funnel["without_corrections"] == 1
    assert funnel["automatic_rate"] == 25.0
    assert funnel["manual_review_rate"] == 75.0
    assert funnel["correction_rate"] == 66.7
    assert funnel["no_change_review_rate"] == 33.3

    # 3. Field aggregation (Tests 5, 6, 7)
    # total: 2 events, 1 doc, share = 2/4 = 50.0%
    # date: 1 event, 1 doc, share = 1/4 = 25.0%
    # company: 1 event, 1 doc, share = 1/4 = 25.0%
    fields = {f["field"]: f for f in data["by_field"]}
    assert "total" in fields
    assert fields["total"]["correction_count"] == 2
    assert fields["total"]["affected_documents"] == 1
    assert fields["total"]["share"] == 50.0

    assert "date" in fields
    assert fields["date"]["correction_count"] == 1
    assert fields["date"]["affected_documents"] == 1
    assert fields["date"]["share"] == 25.0

    assert "company" in fields
    assert fields["company"]["correction_count"] == 1
    assert fields["company"]["affected_documents"] == 1
    assert fields["company"]["share"] == 25.0

    # 4. Document Type aggregation (Tests 8, 9)
    # receipt: 1 reviewed, 1 corrected, 2 events, rate = 100.0%
    # invoice: 1 reviewed, 1 corrected, 2 events, rate = 100.0%
    # act: 1 reviewed, 0 corrected, 0 events, rate = 0.0% (Zero correction document type!)
    dtypes = {d["document_type"]: d for d in data["by_document_type"]}
    assert "receipt" in dtypes
    assert dtypes["receipt"]["reviewed_documents"] == 1
    assert dtypes["receipt"]["documents_with_corrections"] == 1
    assert dtypes["receipt"]["correction_events"] == 2
    assert dtypes["receipt"]["correction_rate"] == 100.0

    assert "invoice" in dtypes
    assert dtypes["invoice"]["reviewed_documents"] == 1
    assert dtypes["invoice"]["documents_with_corrections"] == 1
    assert dtypes["invoice"]["correction_events"] == 2
    assert dtypes["invoice"]["correction_rate"] == 100.0

    assert "act" in dtypes
    assert dtypes["act"]["reviewed_documents"] == 1
    assert dtypes["act"]["documents_with_corrections"] == 0
    assert dtypes["act"]["correction_events"] == 0
    assert dtypes["act"]["correction_rate"] == 0.0

    # 5. Review Reason aggregation (Tests 10, 11)
    # low_confidence: 1 reviewed, 1 corrected, 2 events, rate = 100.0%
    # missing_required_field: 1 reviewed, 1 corrected, 2 events, rate = 100.0%
    # low_image_quality: 1 reviewed, 0 corrected, 0 events, rate = 0.0% (Zero correction reason!)
    reasons = {r["reason"]: r for r in data["by_reason"]}
    assert "low_confidence" in reasons
    assert reasons["low_confidence"]["reviewed_documents"] == 1
    assert reasons["low_confidence"]["documents_with_corrections"] == 1
    assert reasons["low_confidence"]["correction_events"] == 2
    assert reasons["low_confidence"]["correction_rate"] == 100.0

    assert "missing_required_field" in reasons
    assert reasons["missing_required_field"]["reviewed_documents"] == 1
    assert reasons["missing_required_field"]["documents_with_corrections"] == 1
    assert reasons["missing_required_field"]["correction_events"] == 2
    assert reasons["missing_required_field"]["correction_rate"] == 100.0

    assert "low_image_quality" in reasons
    assert reasons["low_image_quality"]["reviewed_documents"] == 1
    assert reasons["low_image_quality"]["documents_with_corrections"] == 0
    assert reasons["low_image_quality"]["correction_events"] == 0
    assert reasons["low_image_quality"]["correction_rate"] == 0.0

    # 6. Reason x Field matrix (Tests 12, 13)
    # (low_confidence, total): 2 events, 1 doc
    # (missing_required_field, date): 1 event, 1 doc
    # (missing_required_field, company): 1 event, 1 doc
    matrix = {(rf["review_reason"], rf["field"]): rf for rf in data["by_reason_field"]}
    assert ("low_confidence", "total") in matrix
    assert matrix[("low_confidence", "total")]["correction_count"] == 2
    assert matrix[("low_confidence", "total")]["affected_documents"] == 1

    assert ("missing_required_field", "date") in matrix
    assert matrix[("missing_required_field", "date")]["correction_count"] == 1
    assert matrix[("missing_required_field", "date")]["affected_documents"] == 1

    assert ("missing_required_field", "company") in matrix
    assert matrix[("missing_required_field", "company")]["correction_count"] == 1
    assert matrix[("missing_required_field", "company")]["affected_documents"] == 1


@pytest.mark.asyncio
async def test_historical_integrity_audit_values_preserved(client: AsyncClient, db_session: AsyncSession):
    """17. Historical integrity: values come from AuditEvent rather than mutated current document values."""
    now = datetime.now(timezone.utc)
    doc = Document(
        id="doc_hist_1",
        filename="invoice_mutated.pdf",
        file_path="/tmp/hist.pdf",
        file_size_bytes=1200,
        mime_type="application/pdf",
        document_type=DocumentType.INVOICE.value,
        status=DocumentStatus.COMPLETED_MANUAL.value,
        decision=AutomationDecision.MANUAL_REVIEW.value,
        processing_finished_at=now,
        review_finished_at=now,
    )
    # In audit history: old_value was "100.00", new_value was "150.00"
    ev_c = AuditEvent(
        id="ev_hist_corr",
        document_id="doc_hist_1",
        actor="operator:charlie",
        action=AuditAction.FIELD_CORRECTED.value,
        field_name="total",
        old_value="100.00",
        new_value="150.00",
        created_at=now,
    )
    db_session.add_all([doc, ev_c])
    await db_session.commit()

    # Query document detail endpoint
    resp = await client.get(f"/api/v1/documents/{doc.id}")
    assert resp.status_code == 200
    doc_data = resp.json()
    assert "field_corrections" in doc_data
    assert len(doc_data["field_corrections"]) == 1
    fc = doc_data["field_corrections"][0]
    assert fc["field"] == "total"
    assert fc["previous_value"] == "100.00"
    assert fc["final_value"] == "150.00"
    assert fc["actor"] == "operator:charlie"


@pytest.mark.asyncio
async def test_period_filtering_semantics(client: AsyncClient, db_session: AsyncSession):
    """Tests 19 (today), 20 (7d/30d), 21 (custom period)."""
    now = datetime.now(timezone.utc)
    two_days_ago = now - timedelta(days=2)
    twenty_days_ago = now - timedelta(days=20)
    forty_days_ago = now - timedelta(days=40)

    # Document 1: Today
    doc_today = Document(
        id="doc_today",
        filename="today.jpg",
        file_path="/tmp/t.jpg",
        file_size_bytes=1000,
        mime_type="image/jpeg",
        document_type=DocumentType.RECEIPT.value,
        status=DocumentStatus.COMPLETED_MANUAL.value,
        decision=AutomationDecision.MANUAL_REVIEW.value,
        processing_finished_at=now,
        review_finished_at=now,
    )
    ev_today = AuditEvent(
        id="ev_today",
        document_id="doc_today",
        actor="operator:t",
        action=AuditAction.FIELD_CORRECTED.value,
        field_name="total",
        old_value="10",
        new_value="20",
        created_at=now,
    )

    # Document 2: 2 days ago (within 7d, 30d, not today)
    doc_2d = Document(
        id="doc_2d",
        filename="2d.jpg",
        file_path="/tmp/2d.jpg",
        file_size_bytes=1000,
        mime_type="image/jpeg",
        document_type=DocumentType.RECEIPT.value,
        status=DocumentStatus.COMPLETED_MANUAL.value,
        decision=AutomationDecision.MANUAL_REVIEW.value,
        processing_finished_at=two_days_ago,
        review_finished_at=two_days_ago,
    )
    ev_2d = AuditEvent(
        id="ev_2d",
        document_id="doc_2d",
        actor="operator:t",
        action=AuditAction.FIELD_CORRECTED.value,
        field_name="date",
        old_value="old",
        new_value="new",
        created_at=two_days_ago,
    )

    # Document 3: 20 days ago (within 30d, not 7d or today)
    doc_20d = Document(
        id="doc_20d",
        filename="20d.jpg",
        file_path="/tmp/20d.jpg",
        file_size_bytes=1000,
        mime_type="image/jpeg",
        document_type=DocumentType.RECEIPT.value,
        status=DocumentStatus.COMPLETED_MANUAL.value,
        decision=AutomationDecision.MANUAL_REVIEW.value,
        processing_finished_at=twenty_days_ago,
        review_finished_at=twenty_days_ago,
    )
    ev_20d = AuditEvent(
        id="ev_20d",
        document_id="doc_20d",
        actor="operator:t",
        action=AuditAction.FIELD_CORRECTED.value,
        field_name="company",
        old_value="c1",
        new_value="c2",
        created_at=twenty_days_ago,
    )

    db_session.add_all([
        doc_today, ev_today,
        doc_2d, ev_2d,
        doc_20d, ev_20d,
    ])
    await db_session.commit()

    # 1. Test period=today
    resp_today = await client.get("/api/v1/analytics/feedback?period=today")
    assert resp_today.status_code == 200
    data_today = resp_today.json()
    assert data_today["summary"]["total_correction_events"] == 1
    assert data_today["summary"]["documents_with_corrections"] == 1

    # 2. Test period=7d
    resp_7d = await client.get("/api/v1/analytics/feedback?period=7d")
    assert resp_7d.status_code == 200
    data_7d = resp_7d.json()
    assert data_7d["summary"]["total_correction_events"] == 2
    assert data_7d["summary"]["documents_with_corrections"] == 2

    # 3. Test period=30d
    resp_30d = await client.get("/api/v1/analytics/feedback?period=30d")
    assert resp_30d.status_code == 200
    data_30d = resp_30d.json()
    assert data_30d["summary"]["total_correction_events"] == 3
    assert data_30d["summary"]["documents_with_corrections"] == 3

    # 4. Test custom bounds: between 3 days ago and 1 day ago (should capture only doc_2d)
    dt_from = (two_days_ago - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    dt_to = (two_days_ago + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    resp_custom = await client.get(f"/api/v1/analytics/feedback?from={dt_from}&to={dt_to}")
    assert resp_custom.status_code == 200
    data_custom = resp_custom.json()
    assert data_custom["summary"]["total_correction_events"] == 1
    assert data_custom["summary"]["documents_with_corrections"] == 1
    assert data_custom["by_field"][0]["field"] == "date"
