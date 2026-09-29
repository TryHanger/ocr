"""Tests for ProcessingStep provenance and field correction audit history."""

import io
from PIL import Image, ImageDraw
import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.job import ProcessingJob
from app.models.step import ProcessingStep
from app.services.pipeline_service import PipelineService


def _create_slanted_image() -> bytes:
    """Creates a slightly rotated synthetic receipt to trigger deskew step."""
    img = Image.new("RGB", (600, 800), color=(250, 250, 250))
    draw = ImageDraw.Draw(img)
    draw.text((60, 60), "PROVENANCE MART SDN BHD", fill=(0, 0, 0))
    draw.text((60, 110), "DATE: 28/09/2026", fill=(0, 0, 0))
    draw.text((60, 600), "TOTAL: 99.50", fill=(0, 0, 0))
    # Rotate by 4 degrees to trigger deskew in preprocessing policy
    rotated = img.rotate(4.0, expand=False, fillcolor=(250, 250, 250))
    buf = io.BytesIO()
    rotated.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_provenance_recording_and_idempotent_retry(client: AsyncClient, db_session: AsyncSession):
    img_bytes = _create_slanted_image()

    # 1. Upload document
    files = {"file": ("provenance_test.jpg", img_bytes, "image/jpeg")}
    upload_res = await client.post("/api/v1/documents", files=files, data={"document_type": "receipt"})
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["id"]

    # 2. Execute pipeline
    pipeline = PipelineService(db_session)
    job_stmt = select(ProcessingJob).where(ProcessingJob.document_id == doc_id)
    job = (await db_session.execute(job_stmt)).scalar_one()

    ok = await pipeline.execute_job(job.id)
    assert ok is True

    # 3. Verify steps via Document Detail API
    detail_res = await client.get(f"/api/v1/documents/{doc_id}")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert "steps" in detail
    steps = detail["steps"]
    assert len(steps) >= 1
    # Check ordering and operations
    orders = [s["step_order"] for s in steps]
    assert orders == sorted(orders)
    for s in steps:
        assert s["duration_ms"] >= 0.0
        assert s["status"] in ("APPLIED", "completed", "failed")

    # 4. Test Idempotency: re-run the pipeline on the same job
    ok2 = await pipeline.execute_job(job.id)
    assert ok2 is True

    # Check steps count didn't duplicate
    steps_stmt = select(ProcessingStep).where(ProcessingStep.document_id == doc_id)
    steps_after = (await db_session.execute(steps_stmt)).scalars().all()
    assert len(steps_after) == len(steps)


@pytest.mark.asyncio
async def test_field_correction_preserves_provenance(client: AsyncClient, db_session: AsyncSession):
    img_bytes = _create_slanted_image()

    files = {"file": ("correction_test.jpg", img_bytes, "image/jpeg")}
    upload_res = await client.post("/api/v1/documents", files=files, data={"document_type": "receipt"})
    doc_id = upload_res.json()["id"]

    pipeline = PipelineService(db_session)
    job_stmt = select(ProcessingJob).where(ProcessingJob.document_id == doc_id)
    job = (await db_session.execute(job_stmt)).scalar_one()
    await pipeline.execute_job(job.id)

    # Fetch initial fields
    detail_res = await client.get(f"/api/v1/documents/{doc_id}")
    detail = detail_res.json()

    # Apply correction to total
    patch_res = await client.patch(
        f"/api/v1/documents/{doc_id}/fields/total",
        json={
            "field_name": "total",
            "corrected_value": "120.00",
            "reviewer_name": "operator_alice",
            "notes": "Fixed price typo",
        },
    )
    assert patch_res.status_code == 200
    field_data = patch_res.json()
    assert field_data["value"] == "120.00"
    assert field_data["corrected_value"] == "120.00"
    assert field_data["is_corrected"] is True

    # Verify audit event was stored
    doc_after = await client.get(f"/api/v1/documents/{doc_id}")
    audit_events = doc_after.json()["audit_events"]
    correction_event = [e for e in audit_events if e["action"] == "field_corrected"]
    assert len(correction_event) >= 1
    assert correction_event[0]["field_name"] == "total"
    assert correction_event[0]["new_value"] == "120.00"
    assert "operator_alice" in correction_event[0]["actor"]
