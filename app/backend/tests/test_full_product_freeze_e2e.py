"""Full Product Freeze End-to-End Verification Test.

Covers the complete Document AI Control Center lifecycle:
1. Document Ingestion (upload)
2. Automated Pipeline Execution
3. Quality Analysis
4. OCR & KIE Extraction
5. Confidence Evaluation
6. Decision Engine & Audit Trail
7. HITL Manual Review Workflow
8. Human Field Correction & Audit
9. Operational Issue Derivation (MVP-9)
10. Feedback Analytics & Derived Research Signals (MVP-10)
11. Persistent Research Question Lifecycle & Transitions (MVP-10)
"""

from pathlib import Path
import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.enums import AuditAction
from app.models.audit import AuditEvent
from app.models.document import Document
from app.models.field import ExtractedField
from app.models.job import ProcessingJob
from app.models.research_question import ResearchQuestion
from app.services.pipeline_service import PipelineService


@pytest.mark.asyncio
async def test_full_document_ai_e2e_freeze_journey(client: AsyncClient, db_session: AsyncSession):
    # Locate real test image
    repo_root = Path(settings.RESEARCH_ROOT_PATH).resolve()
    real_img_path = repo_root / "data" / "SROIE2019" / "test" / "img" / "X00016469670.jpg"

    if not real_img_path.exists():
        pytest.skip(f"Real test image not found at {real_img_path}")

    with open(real_img_path, "rb") as f:
        img_bytes = f.read()

    # 1. Document Upload / Ingestion
    files = {"file": ("test_receipt.jpg", img_bytes, "image/jpeg")}
    upload_res = await client.post("/api/v1/documents", files=files, data={"document_type": "receipt"})
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["id"]
    assert doc_id is not None

    # 2. Automated Pipeline Processing
    job_stmt = select(ProcessingJob).where(ProcessingJob.document_id == doc_id)
    job = (await db_session.execute(job_stmt)).scalar_one()

    pipeline = PipelineService(db_session)
    success = await pipeline.execute_job(job.id)
    assert success is True

    # 3. Quality Analysis Verification
    doc_res = await client.get(f"/api/v1/documents/{doc_id}")
    assert doc_res.status_code == 200
    doc_data = doc_res.json()

    assert doc_data["quality"] is not None
    assert "quality_score" in doc_data["quality"]
    assert "blur_score" in doc_data["quality"]
    assert doc_data["quality"]["quality_score"] > 0.0

    # 4. OCR & KIE Extraction
    assert len(doc_data["pages"]) >= 1
    page = doc_data["pages"][0]
    assert len(page["tokens"]) > 0
    assert page["full_text"] != ""

    assert len(doc_data["fields"]) > 0
    assert "total" in doc_data["fields"]

    # 5. Confidence Evaluation
    assert doc_data["confidence"] is not None
    assert 0.0 <= doc_data["confidence"] <= 100.0

    # 6. Decision Engine & Audit Trail
    assert doc_data["decision"] in ("automatic", "manual_review")
    assert doc_data["status"] in ("completed_automatic", "manual_review")

    audit_res = await client.get(f"/api/v1/documents/{doc_id}")
    audit_events = audit_res.json().get("audit_events", [])
    assert len(audit_events) > 0
    decision_events = [
        e for e in audit_events if e["action"] in ("DECISION_EVALUATED", "decision_evaluated")
    ]
    assert len(decision_events) >= 1

    # 7. HITL Manual Review Workflow
    doc = await db_session.get(Document, doc_id)
    doc.status = "manual_review"
    doc.decision = "manual_review"
    await db_session.commit()

    # 8. Human Field Correction & Audit
    correction_res = await client.patch(
        f"/api/v1/documents/{doc_id}/fields/total",
        json={
            "field_name": "total",
            "corrected_value": "125.50",
            "reviewer_name": "senior_reviewer_1",
            "notes": "Manual total correction",
        },
    )
    assert correction_res.status_code == 200
    assert correction_res.json()["value"] == "125.50"
    assert correction_res.json()["is_corrected"] is True

    complete_res = await client.post(
        f"/api/v1/documents/{doc_id}/review/complete",
        json={"reviewer_name": "senior_reviewer_1", "notes": "Completed manual review"},
    )
    assert complete_res.status_code == 200
    assert complete_res.json()["status"] == "completed_manual"

    # 9. Operational Issue Derivation (MVP-9)
    issues_res = await client.get(f"/api/v1/operations/documents/{doc_id}/issues")
    assert issues_res.status_code == 200
    issues_data = issues_res.json()
    assert "issues" in issues_data
    assert len(issues_data["issues"]) > 0
    assert "available_actions" in issues_data["issues"][0]
    assert len(issues_data["issues"][0]["available_actions"]) > 0
    assert issues_data["document_id"] == doc_id

    # 10. Feedback Analytics & Derived Research Signals (MVP-10)
    for i in range(5):
        d = Document(
            filename=f"hist_sroie_{i}.jpg",
            file_path=f"/tmp/hist_sroie_{i}.jpg",
            file_size_bytes=1024,
            mime_type="image/jpeg",
            document_type="receipt",
            status="completed_manual",
            decision="manual_review",
            confidence=0.72,
        )
        db_session.add(d)
        await db_session.flush()

        f = ExtractedField(
            document_id=d.id,
            field_name="total",
            value="10.00",
            corrected_value="15.00" if i < 3 else None,
            confidence=0.70,
        )
        db_session.add(f)

        if i < 3:
            ev = AuditEvent(
                document_id=d.id,
                action=AuditAction.FIELD_CORRECTED.value,
                actor="reviewer_test",
                field_name="total",
                old_value="10.00",
                new_value="15.00",
            )
            db_session.add(ev)
    await db_session.commit()

    overview_res = await client.get("/api/v1/improvement/overview?period=all")
    assert overview_res.status_code == 200
    overview_data = overview_res.json()
    assert overview_data["summary"]["total_field_corrections"] >= 4
    assert len(overview_data["by_field"]) > 0

    signals_res = await client.get("/api/v1/improvement/signals?period=all")
    assert signals_res.status_code == 200
    signals = signals_res.json()
    assert len(signals) > 0

    target_signal = signals[0]
    assert target_signal["id"].startswith("sig_")
    assert "evidence" in target_signal
    assert len(target_signal["related_evidence_refs"]) > 0

    # 11. Persistent Research Question Lifecycle & Transitions (MVP-10)
    create_q_payload = {
        "title": f"Why does field {target_signal.get('field', 'total')} require frequent manual correction?",
        "description": "Investigating OCR/KIE extraction error patterns observed in recent batches.",
        "source_signal_id": target_signal["id"],
        "field": target_signal.get("field", "total"),
        "document_type": "receipt",
        "related_evidence_refs": target_signal.get("related_evidence_refs", []),
        "created_by": "qa_lead",
    }
    create_q_res = await client.post("/api/v1/improvement/questions", json=create_q_payload)
    assert create_q_res.status_code == 201
    question = create_q_res.json()
    q_id = question["id"]
    assert question["status"] == "OPEN"

    detail_q_res = await client.get(f"/api/v1/improvement/questions/{q_id}")
    assert detail_q_res.status_code == 200
    assert detail_q_res.json()["enriched_evidence"] is not None

    patch_1_res = await client.patch(
        f"/api/v1/improvement/questions/{q_id}",
        json={"status": "IN_PROGRESS"},
    )
    assert patch_1_res.status_code == 200
    assert patch_1_res.json()["status"] == "IN_PROGRESS"

    patch_2_res = await client.patch(
        f"/api/v1/improvement/questions/{q_id}",
        json={"status": "EXPERIMENT_AVAILABLE"},
    )
    assert patch_2_res.status_code == 200
    assert patch_2_res.json()["status"] == "EXPERIMENT_AVAILABLE"

    patch_3_res = await client.patch(
        f"/api/v1/improvement/questions/{q_id}",
        json={"status": "CLOSED"},
    )
    assert patch_3_res.status_code == 200
    assert patch_3_res.json()["status"] == "CLOSED"

    persisted_q = await db_session.get(ResearchQuestion, q_id)
    assert persisted_q is not None
    assert (persisted_q.status == "CLOSED" or str(persisted_q.status) == "CLOSED")
