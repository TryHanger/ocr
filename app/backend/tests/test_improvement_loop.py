"""Comprehensive backend tests for MVP-10: Document AI Improvement Loop."""

from datetime import datetime, timezone
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import AuditAction, AutomationDecision, DocumentStatus, DocumentType, ResearchQuestionStatus, ResearchSignalType, ReviewReason
from app.core.improvement_constants import MIN_CORRECTION_COUNT, MIN_SIGNAL_SAMPLE_SIZE
from app.models.audit import AuditEvent
from app.models.document import Document
from app.models.field import ExtractedField
from app.models.research_question import ResearchQuestion
from app.schemas.improvement import ResearchSignal


@pytest.mark.asyncio
async def test_feedback_overview_empty_safety(client: AsyncClient, db_session: AsyncSession):
    """1. Empty database returns zero for all rates without NaN/Inf or division errors."""
    resp = await client.get("/api/v1/improvement/overview")
    assert resp.status_code == 200
    data = resp.json()

    summary = data["summary"]
    assert summary["documents_processed"] == 0
    assert summary["documents_reviewed"] == 0
    assert summary["documents_corrected"] == 0
    assert summary["total_field_corrections"] == 0
    assert summary["correction_rate"] == 0.0
    assert summary["no_change_review_rate"] == 0.0

    assert data["by_field"] == []
    assert data["by_confidence"] != []  # Buckets are defined with 0 counts
    for b in data["by_confidence"]:
        assert b["evaluated_fields"] == 0
        assert b["corrected_fields"] == 0
        assert b["correction_rate"] == 0.0
        assert b["share_of_corrections"] == 0.0

    assert data["by_reason"] == []
    assert data["by_document_type"] == []
    assert data["active_signals_count"] == 0
    assert data["open_questions_count"] == 0


@pytest.mark.asyncio
async def test_reviewed_vs_corrected_distinction(client: AsyncClient, db_session: AsyncSession):
    """2. Operator reviewed and approved without edits: reviewed=True, corrected=False."""
    now = datetime.now(timezone.utc)
    doc = Document(
        id="doc_reviewed_no_edit",
        filename="rec_clean.jpg",
        file_path="/tmp/rec_clean.jpg",
        file_size_bytes=1024,
        mime_type="image/jpeg",
        document_type=DocumentType.RECEIPT.value,
        status=DocumentStatus.COMPLETED_MANUAL.value,
        decision=AutomationDecision.MANUAL_REVIEW.value,
        review_reason=ReviewReason.NONE.value,
        processing_finished_at=now,
        review_finished_at=now,
    )
    f1 = ExtractedField(
        id="f_no_corr_1",
        document_id="doc_reviewed_no_edit",
        field_name="total",
        value="50.00",
        confidence=0.92,
        validation_status="VALID",
    )
    db_session.add_all([doc, f1])
    await db_session.commit()

    resp = await client.get("/api/v1/improvement/overview")
    assert resp.status_code == 200
    data = resp.json()
    summary = data["summary"]

    assert summary["documents_processed"] == 1
    assert summary["documents_reviewed"] == 1
    assert summary["documents_corrected"] == 0
    assert summary["total_field_corrections"] == 0
    assert summary["correction_rate"] == 0.0
    assert summary["no_change_review_rate"] == 100.0


@pytest.mark.asyncio
async def test_dual_metrics_and_evaluated_fields_denominator(client: AsyncClient, db_session: AsyncSession):
    """3. Dual metrics on fields (rate + share) & strict evaluated denominator (valid confidence only)."""
    now = datetime.now(timezone.utc)

    # Document 1: evaluated total with confidence 0.65 -> corrected
    doc1 = Document(
        id="doc_dual_1",
        filename="rec1.jpg",
        file_path="/tmp/rec1.jpg",
        file_size_bytes=1000,
        mime_type="image/jpeg",
        document_type=DocumentType.RECEIPT.value,
        status=DocumentStatus.COMPLETED_MANUAL.value,
        decision=AutomationDecision.MANUAL_REVIEW.value,
        processing_finished_at=now,
    )
    f1_1 = ExtractedField(
        id="f_d1_total",
        document_id="doc_dual_1",
        field_name="total",
        value="100.00",
        corrected_value="120.00",
        confidence=0.65,
    )
    f1_2 = ExtractedField(
        id="f_d1_date",
        document_id="doc_dual_1",
        field_name="date",
        value="2026-01-01",
        confidence=0.95,
    )
    # Field without confidence (should NOT be in evaluated denominator)
    f1_3 = ExtractedField(
        id="f_d1_no_conf",
        document_id="doc_dual_1",
        field_name="notes",
        value="memo",
        confidence=None,
    )

    ev_corr1 = AuditEvent(
        id="ev_c_1",
        document_id="doc_dual_1",
        actor="operator:alice",
        action=AuditAction.FIELD_CORRECTED.value,
        field_name="total",
        old_value="100.00",
        new_value="120.00",
        created_at=now,
    )

    # Document 2: evaluated total with confidence 0.60 -> corrected again
    doc2 = Document(
        id="doc_dual_2",
        filename="rec2.jpg",
        file_path="/tmp/rec2.jpg",
        file_size_bytes=1000,
        mime_type="image/jpeg",
        document_type=DocumentType.RECEIPT.value,
        status=DocumentStatus.COMPLETED_MANUAL.value,
        decision=AutomationDecision.MANUAL_REVIEW.value,
        processing_finished_at=now,
    )
    f2_1 = ExtractedField(
        id="f_d2_total",
        document_id="doc_dual_2",
        field_name="total",
        value="40.00",
        corrected_value="45.00",
        confidence=0.60,
    )
    ev_corr2 = AuditEvent(
        id="ev_c_2",
        document_id="doc_dual_2",
        actor="operator:bob",
        action=AuditAction.FIELD_CORRECTED.value,
        field_name="total",
        old_value="40.00",
        new_value="45.00",
        created_at=now,
    )

    db_session.add_all([doc1, f1_1, f1_2, f1_3, ev_corr1, doc2, f2_1, ev_corr2])
    await db_session.commit()

    resp = await client.get("/api/v1/improvement/overview")
    assert resp.status_code == 200
    data = resp.json()

    # Total corrections: 2 (both on 'total')
    by_field = data["by_field"]
    total_metric = next(f for f in by_field if f["field"] == "total")
    assert total_metric["correction_count"] == 2
    assert total_metric["evaluated_count"] == 2
    assert total_metric["correction_rate"] == 100.0
    assert total_metric["share_of_all_corrections"] == 100.0
    assert total_metric["affected_documents"] == 2

    # Confidence bucket 0.50–0.70 has both total corrections
    bucket_60 = next(b for b in data["by_confidence"] if b["bucket"] == "0.50–0.70")
    assert bucket_60["evaluated_fields"] == 2
    assert bucket_60["corrected_fields"] == 2
    assert bucket_60["correction_rate"] == 100.0


@pytest.mark.asyncio
async def test_no_signals_without_sufficient_data(client: AsyncClient, db_session: AsyncSession):
    """4. Threshold constraints prevent generating signals when reviewed < 5 or corrections < 2."""
    now = datetime.now(timezone.utc)
    # Only 2 reviewed documents (below MIN_SIGNAL_SAMPLE_SIZE = 5)
    doc1 = Document(
        id="doc_low_n_1",
        filename="rec1.jpg",
        file_path="/tmp/rec1.jpg",
        file_size_bytes=1000,
        mime_type="image/jpeg",
        document_type=DocumentType.RECEIPT.value,
        status=DocumentStatus.COMPLETED_MANUAL.value,
        processing_finished_at=now,
    )
    doc2 = Document(
        id="doc_low_n_2",
        filename="rec2.jpg",
        file_path="/tmp/rec2.jpg",
        file_size_bytes=1000,
        mime_type="image/jpeg",
        document_type=DocumentType.RECEIPT.value,
        status=DocumentStatus.COMPLETED_MANUAL.value,
        processing_finished_at=now,
    )
    ev_c = AuditEvent(
        id="ev_c_low",
        document_id="doc_low_n_1",
        actor="operator",
        action=AuditAction.FIELD_CORRECTED.value,
        field_name="total",
        created_at=now,
    )
    db_session.add_all([doc1, doc2, ev_c])
    await db_session.commit()

    resp = await client.get("/api/v1/improvement/signals")
    assert resp.status_code == 200
    signals = resp.json()
    assert signals == []  # Strictly suppressed without sufficient sample size


@pytest.mark.asyncio
async def test_deterministic_signals_and_negative_invariants(client: AsyncClient, db_session: AsyncSession):
    """5. When sufficient data exists, signals are derived with deterministic hash IDs.

    Verifies strictly observational text (NO prescriptive or causal words).
    """
    now = datetime.now(timezone.utc)

    # Create 6 reviewed documents with corrections to meet MIN_SIGNAL_SAMPLE_SIZE (>=5) and MIN_CORRECTION_COUNT (>=2)
    docs = []
    audits = []
    fields = []
    for i in range(6):
        d_id = f"doc_sig_pop_{i}"
        d = Document(
            id=d_id,
            filename=f"rec_{i}.jpg",
            file_path=f"/tmp/rec_{i}.jpg",
            file_size_bytes=1000,
            mime_type="image/jpeg",
            document_type=DocumentType.RECEIPT.value,
            status=DocumentStatus.COMPLETED_MANUAL.value,
            decision=AutomationDecision.MANUAL_REVIEW.value,
            review_reason=ReviewReason.LOW_CONFIDENCE.value,
            processing_finished_at=now,
        )
        docs.append(d)

        # Field total in confidence bucket 0.50–0.70
        f = ExtractedField(
            id=f"f_sig_{i}",
            document_id=d_id,
            field_name="total",
            value="10.00",
            corrected_value="12.00",
            confidence=0.60,
        )
        fields.append(f)

        ev_c = AuditEvent(
            id=f"ev_corr_sig_{i}",
            document_id=d_id,
            actor="operator:alice",
            action=AuditAction.FIELD_CORRECTED.value,
            field_name="total",
            old_value="10.00",
            new_value="12.00",
            created_at=now,
        )
        audits.append(ev_c)

        ev_dec = AuditEvent(
            id=f"ev_dec_sig_{i}",
            document_id=d_id,
            actor="system",
            action=AuditAction.DECISION_EVALUATED.value,
            metadata_json={"decision": "manual_review", "reason": "low_confidence"},
            created_at=now,
        )
        audits.append(ev_dec)

    db_session.add_all(docs + fields + audits)
    await db_session.commit()

    # Query signals twice to test determinism
    resp1 = await client.get("/api/v1/improvement/signals")
    resp2 = await client.get("/api/v1/improvement/signals")
    assert resp1.status_code == 200
    assert resp2.status_code == 200

    sigs1 = resp1.json()
    sigs2 = resp2.json()

    assert len(sigs1) > 0
    assert sigs1 == sigs2  # Exact reproducible match

    # Verify ID structure: sig_{type}_{hash}
    for s in sigs1:
        assert s["id"].startswith("sig_")
        assert len(s["id"].split("_")) >= 3

        # Negative checks on signal wording: Observation != Recommendation
        text = (s["title"] + " " + s["description"]).lower()
        prohibited_words = [
            "use clahe",
            "retrain model",
            "switch model",
            "change threshold",
            "causes",
            "risk score",
            "recommended solution",
        ]
        for pw in prohibited_words:
            assert pw not in text, f"Prescriptive or causal phrase '{pw}' found in signal: {text}"

        # Evidence dictionary must have explicit numbers
        ev = s["evidence"]
        assert isinstance(ev, dict)
        assert len(ev) > 0


@pytest.mark.asyncio
async def test_research_question_lifecycle_and_pagination(client: AsyncClient, db_session: AsyncSession):
    """6. Research question creation from signal, pagination, and status PATCH transitions."""
    # 1. Create a question linked to a signal
    create_payload = {
        "title": "Investigate low confidence clustering on receipt totals",
        "description": "Observed that total field corrections heavily cluster in 0.50–0.70 confidence bucket.",
        "source_signal_id": "sig_confidence_correction_pattern_abc1234567",
        "field": "total",
        "document_type": "receipt",
        "related_evidence_refs": [
            {
                "track": "B0",
                "experiment_id": "b0_clean_validation",
                "name": "Clean Baseline Evaluation (B0)",
            }
        ],
        "created_by": "researcher_elena",
    }
    create_resp = await client.post("/api/v1/improvement/questions", json=create_payload)
    assert create_resp.status_code == 201
    q = create_resp.json()
    q_id = q["id"]
    assert q["status"] == "OPEN"
    assert q["title"] == create_payload["title"]
    assert q["source_signal_id"] == create_payload["source_signal_id"]

    # 2. Paginated list
    list_resp = await client.get("/api/v1/improvement/questions?limit=10&offset=0")
    assert list_resp.status_code == 200
    list_data = list_resp.json()
    assert list_data["total"] >= 1
    assert any(item["id"] == q_id for item in list_data["items"])

    # 3. Status PATCH transition: OPEN -> IN_PROGRESS
    patch1 = await client.patch(
        f"/api/v1/improvement/questions/{q_id}",
        json={"status": ResearchQuestionStatus.IN_PROGRESS.value},
    )
    assert patch1.status_code == 200
    assert patch1.json()["status"] == "IN_PROGRESS"

    # 4. Status PATCH transition: IN_PROGRESS -> EXPERIMENT_AVAILABLE
    patch2 = await client.patch(
        f"/api/v1/improvement/questions/{q_id}",
        json={"status": ResearchQuestionStatus.EXPERIMENT_AVAILABLE.value},
    )
    assert patch2.status_code == 200
    assert patch2.json()["status"] == "EXPERIMENT_AVAILABLE"

    # 5. Status PATCH transition: EXPERIMENT_AVAILABLE -> CLOSED (soft lifecycle end-state)
    patch3 = await client.patch(
        f"/api/v1/improvement/questions/{q_id}",
        json={"status": ResearchQuestionStatus.CLOSED.value},
    )
    assert patch3.status_code == 200
    assert patch3.json()["status"] == "CLOSED"

    # 6. Verify detail enrichment with research evidence
    detail_resp = await client.get(f"/api/v1/improvement/questions/{q_id}")
    assert detail_resp.status_code == 200
    detail_data = detail_resp.json()
    assert detail_data["id"] == q_id
    assert detail_data["status"] == "CLOSED"
    if detail_data.get("enriched_evidence"):
        assert "b0_clean_validation" in detail_data["enriched_evidence"]


@pytest.mark.asyncio
async def test_research_isolation_and_no_src_imports():
    """7. Verifies research isolation: product layer never imports src/, data/, or experiments/."""
    import inspect
    import app.services.feedback_analytics_service as fas
    import app.services.research_signal_service as rss
    import app.services.research_question_service as rqs
    import app.api.v1.endpoints.improvement as ep

    for mod in [fas, rss, rqs, ep]:
        src_code = inspect.getsource(mod)
        assert "from src." not in src_code
        assert "import src." not in src_code
        assert "import experiments." not in src_code
        assert "from experiments." not in src_code
