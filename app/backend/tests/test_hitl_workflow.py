"""Tests for Human-in-the-Loop (HITL) Manual Review workflow and validation gates."""

import io
from PIL import Image, ImageDraw
import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import AutomationDecision, DocumentStatus, ReviewReason
from app.models.document import Document
from app.models.job import ProcessingJob
from app.services.pipeline_service import PipelineService


def _create_receipt_with_bad_total() -> bytes:
    """Creates a receipt image where the total text is visibly invalid or missing."""
    img = Image.new("RGB", (600, 800), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((50, 50), "HITL MART SDN BHD", fill=(0, 0, 0))
    draw.text((50, 100), "DATE: 2026-09-28", fill=(0, 0, 0))
    # Note: total will not be recognized as a valid numeric amount
    draw.text((50, 600), "TOTAL: 12O.50", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_manual_review_reason_and_details(client: AsyncClient, db_session: AsyncSession):
    img_bytes = _create_receipt_with_bad_total()

    # 1. Ingest document
    files = {"file": ("hitl_doc.jpg", img_bytes, "image/jpeg")}
    upload_res = await client.post("/api/v1/documents", files=files, data={"document_type": "receipt"})
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["id"]

    # 2. Run pipeline
    pipeline = PipelineService(db_session)
    job_stmt = select(ProcessingJob).where(ProcessingJob.document_id == doc_id)
    job = (await db_session.execute(job_stmt)).scalar_one()
    await pipeline.execute_job(job.id)

    # 3. Check document status and review details
    detail_res = await client.get(f"/api/v1/documents/{doc_id}")
    assert detail_res.status_code == 200
    doc = detail_res.json()

    assert doc["status"] == DocumentStatus.MANUAL_REVIEW.value
    assert doc["review_reason"] in (
        ReviewReason.VALIDATION_FAILED.value,
        ReviewReason.INVALID_NUMBER.value,
        ReviewReason.MISSING_REQUIRED_FIELD.value,
        ReviewReason.LOW_CONFIDENCE.value,
    )
    assert doc["review_details"] is not None
    assert "reasons" in doc["review_details"]
    assert "validation_errors" in doc["review_details"]
    assert doc["review_started_at"] is not None


@pytest.mark.asyncio
async def test_field_correction_and_original_value_preservation(client: AsyncClient, db_session: AsyncSession):
    img_bytes = _create_receipt_with_bad_total()

    files = {"file": ("correction_test.jpg", img_bytes, "image/jpeg")}
    upload_res = await client.post("/api/v1/documents", files=files, data={"document_type": "receipt"})
    doc_id = upload_res.json()["id"]

    pipeline = PipelineService(db_session)
    job_stmt = select(ProcessingJob).where(ProcessingJob.document_id == doc_id)
    job = (await db_session.execute(job_stmt)).scalar_one()
    await pipeline.execute_job(job.id)

    # Correct company field
    patch_res = await client.patch(
        f"/api/v1/documents/{doc_id}/fields/company",
        json={
            "field_name": "company",
            "corrected_value": "HITL SUPERMARKET BHD",
            "reviewer_name": "operator_bob",
            "notes": "Expanded company name",
        },
    )
    assert patch_res.status_code == 200
    f = patch_res.json()
    assert f["value"] == "HITL SUPERMARKET BHD"
    assert f["corrected_value"] == "HITL SUPERMARKET BHD"
    assert f["original_value"] is not None
    assert f["is_corrected"] is True

    # Audit event must be logged
    detail_res = await client.get(f"/api/v1/documents/{doc_id}")
    audit_events = detail_res.json()["audit_events"]
    corr_events = [e for e in audit_events if e["action"] == "field_corrected"]
    assert len(corr_events) >= 1
    assert corr_events[0]["field_name"] == "company"
    assert corr_events[0]["new_value"] == "HITL SUPERMARKET BHD"
    assert "operator_bob" in corr_events[0]["actor"]


@pytest.mark.asyncio
async def test_review_completion_gate_and_revalidation(client: AsyncClient, db_session: AsyncSession):
    img_bytes = _create_receipt_with_bad_total()

    files = {"file": ("gate_test.jpg", img_bytes, "image/jpeg")}
    upload_res = await client.post("/api/v1/documents", files=files, data={"document_type": "receipt"})
    doc_id = upload_res.json()["id"]

    pipeline = PipelineService(db_session)
    job_stmt = select(ProcessingJob).where(ProcessingJob.document_id == doc_id)
    job = (await db_session.execute(job_stmt)).scalar_one()
    await pipeline.execute_job(job.id)

    # 1. Attempt to complete review with bad total -> MUST FAIL
    bad_patch = await client.patch(
        f"/api/v1/documents/{doc_id}/fields/total",
        json={"field_name": "total", "corrected_value": "12O.50", "reviewer_name": "bob"},
    )
    assert bad_patch.status_code == 200
    assert bad_patch.json()["validation_status"] == "INVALID"

    fail_complete = await client.post(
        f"/api/v1/documents/{doc_id}/review/complete",
        json={"reviewer_name": "bob", "notes": "Premature attempt"},
    )
    assert fail_complete.status_code == 400
    assert "Validation failed" in fail_complete.json()["detail"]

    # Verify document is still in MANUAL_REVIEW
    doc_res = await client.get(f"/api/v1/documents/{doc_id}")
    assert doc_res.json()["status"] == DocumentStatus.MANUAL_REVIEW.value
    assert doc_res.json()["review_finished_at"] is None

    # 2. Fix the invalid fields so all required fields are valid
    await client.patch(
        f"/api/v1/documents/{doc_id}/fields/company",
        json={"field_name": "company", "corrected_value": "HITL MART", "reviewer_name": "bob"},
    )
    await client.patch(
        f"/api/v1/documents/{doc_id}/fields/date",
        json={"field_name": "date", "corrected_value": "2026-09-28", "reviewer_name": "bob"},
    )
    ok_patch = await client.patch(
        f"/api/v1/documents/{doc_id}/fields/total",
        json={"field_name": "total", "corrected_value": "120.50", "reviewer_name": "bob"},
    )
    assert ok_patch.status_code == 200
    assert ok_patch.json()["validation_status"] == "VALID"

    # 3. Complete review now -> MUST SUCCEED
    ok_complete = await client.post(
        f"/api/v1/documents/{doc_id}/review/complete",
        json={"reviewer_name": "bob", "notes": "All fields verified"},
    )
    assert ok_complete.status_code == 200
    assert ok_complete.json()["status"] == DocumentStatus.COMPLETED_MANUAL.value

    # 4. Check Document record
    final_doc = (await client.get(f"/api/v1/documents/{doc_id}")).json()
    assert final_doc["status"] == DocumentStatus.COMPLETED_MANUAL.value
    assert final_doc["decision"] == AutomationDecision.MANUAL_REVIEW.value
    assert final_doc["review_finished_at"] is not None
    assert final_doc["review_duration_ms"] is not None
    assert final_doc["review_duration_ms"] >= 0.0
