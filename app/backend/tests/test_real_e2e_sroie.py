"""End-to-End integration test on a real SROIE document from data/SROIE2019/test/img/."""

from pathlib import Path
import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.document import Document
from app.models.job import ProcessingJob
from app.services.pipeline_service import PipelineService


@pytest.mark.asyncio
async def test_real_sroie_document_pipeline_e2e(client: AsyncClient, db_session: AsyncSession):
    # Locate real SROIE test image
    repo_root = Path(settings.RESEARCH_ROOT_PATH).resolve()
    real_img_path = repo_root / "data" / "SROIE2019" / "test" / "img" / "X00016469670.jpg"

    if not real_img_path.exists():
        pytest.skip(f"Real SROIE test image not found at {real_img_path}")

    with open(real_img_path, "rb") as f:
        img_bytes = f.read()

    # 1. Upload Real SROIE Document
    files = {"file": ("X00016469670.jpg", img_bytes, "image/jpeg")}
    upload_res = await client.post("/api/v1/documents", files=files, data={"document_type": "receipt"})
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["id"]

    # 2. Worker executes pipeline
    pipeline = PipelineService(db_session)
    job_stmt = select(ProcessingJob).where(ProcessingJob.document_id == doc_id)
    job = (await db_session.execute(job_stmt)).scalar_one()

    success = await pipeline.execute_job(job.id)
    assert success is True

    # 3. Retrieve final document details
    detail_res = await client.get(f"/api/v1/documents/{doc_id}")
    assert detail_res.status_code == 200
    doc_data = detail_res.json()

    # Verify state machine progressed past intake
    assert doc_data["status"] in ("completed_automatic", "manual_review")
    assert doc_data["confidence"] is not None
    assert doc_data["confidence"] > 0.0

    # Verify quality analysis was executed
    assert doc_data["quality"] is not None
    assert doc_data["quality"]["blur_score"] > 0.0
    assert doc_data["quality"]["quality_score"] > 0.0

    # Verify page and OCR tokens
    assert len(doc_data["pages"]) == 1
    page = doc_data["pages"][0]
    assert len(page["tokens"]) > 0
    assert page["full_text"] != ""

    # Verify KIE fields
    fields = doc_data["fields"]
    assert len(fields) > 0

    # Check that at least company or total was recognized on this real receipt
    assert "company" in fields or "total" in fields or "date" in fields

    # Verify audit events recorded
    assert len(doc_data["audit_events"]) >= 4
