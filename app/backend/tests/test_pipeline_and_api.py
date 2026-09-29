"""Integration tests for pipeline execution, REST API, review, and analytics."""

import io
from PIL import Image, ImageDraw
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.job import ProcessingJob
from app.services.pipeline_service import PipelineService


def _create_synthetic_receipt_image() -> bytes:
    """Generate a clean synthetic receipt image with visible text."""
    img = Image.new("RGB", (600, 800), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((50, 50), "RESTORE STORE SDN BHD", fill=(0, 0, 0))
    draw.text((50, 100), "DATE: 25/09/2026", fill=(0, 0, 0))
    draw.text((50, 150), "JALAN AMPANG KUALA LUMPUR", fill=(0, 0, 0))
    draw.text((50, 600), "TOTAL: 154.00", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_system_endpoints(client: AsyncClient):
    # Health
    res = await client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

    # Ready
    res = await client.get("/ready")
    assert res.status_code == 200
    assert res.json()["status"] == "ready"

    # Version
    res = await client.get("/api/version")
    assert res.status_code == 200
    assert "version" in res.json()


@pytest.mark.asyncio
async def test_upload_pipeline_and_review_workflow(client: AsyncClient, db_session: AsyncSession):
    image_bytes = _create_synthetic_receipt_image()

    # 1. Upload document
    files = {"file": ("receipt_test.jpg", image_bytes, "image/jpeg")}
    upload_res = await client.post("/api/v1/documents", files=files, data={"document_type": "receipt"})
    assert upload_res.status_code == 201
    doc_data = upload_res.json()
    doc_id = doc_data["id"]
    assert doc_data["status"] == "uploaded"

    # 2. Run Pipeline Service on the job
    pipeline = PipelineService(db_session)
    # Fetch job ID from DB
    from sqlalchemy import select
    job_stmt = select(ProcessingJob).where(ProcessingJob.document_id == doc_id)
    job = (await db_session.execute(job_stmt)).scalar_one()

    success = await pipeline.execute_job(job.id)
    assert success is True

    # 3. Inspect document details via API
    detail_res = await client.get(f"/api/v1/documents/{doc_id}")
    assert detail_res.status_code == 200
    detail = detail_res.json()

    assert detail["status"] in ("completed_automatic", "manual_review")
    assert len(detail["pages"]) == 1
    assert detail["quality"] is not None
    assert detail["quality"]["quality_score"] > 0
    assert len(detail["audit_events"]) >= 3

    # 4. Review / Field Correction
    patch_res = await client.patch(
        f"/api/v1/documents/{doc_id}/fields/total",
        json={
            "field_name": "total",
            "corrected_value": "154.50",
            "reviewer_name": "ivanov",
            "notes": "Corrected cents",
        },
    )
    if "total" in detail["fields"]:
        assert patch_res.status_code == 200
        assert patch_res.json()["value"] == "154.50"

    # 5. Operator Approval
    approve_res = await client.post(
        f"/api/v1/documents/{doc_id}/approve",
        json={"reviewer_name": "ivanov", "notes": "Approved after verification"},
    )
    assert approve_res.status_code == 200
    assert approve_res.json()["status"] == "completed_manual"

    # 6. Verify Analytics
    analytics_res = await client.get("/api/v1/analytics/overview")
    assert analytics_res.status_code == 200
    kpis = analytics_res.json()["kpis"]
    assert kpis["total_documents"] >= 1
    assert kpis["processed_count"] >= 1
