"""Tests for Operational Telemetry, HITL Metrics, and Confidence Distributions."""

import io
from PIL import Image, ImageDraw
import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import DocumentStatus
from app.models.job import ProcessingJob
from app.services.pipeline_service import PipelineService


def _create_sample_receipt(text_total: str = "100.00") -> bytes:
    img = Image.new("RGB", (600, 800), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((50, 50), "TELEMETRY STORE SDN BHD", fill=(0, 0, 0))
    draw.text((50, 100), "DATE: 2026-09-28", fill=(0, 0, 0))
    draw.text((50, 600), f"TOTAL: {text_total}", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_telemetry_endpoints_and_aggregations(client: AsyncClient, db_session: AsyncSession):
    # 1. Upload and process a document
    img_bytes = _create_sample_receipt(text_total="250.00")
    files = {"file": ("telem_1.jpg", img_bytes, "image/jpeg")}
    upload_res = await client.post("/api/v1/documents", files=files, data={"document_type": "receipt"})
    doc_id = upload_res.json()["id"]

    pipeline = PipelineService(db_session)
    job_stmt = select(ProcessingJob).where(ProcessingJob.document_id == doc_id)
    job = (await db_session.execute(job_stmt)).scalar_one()
    await pipeline.execute_job(job.id)

    # 2. Check processing duration tracking
    doc_data = (await client.get(f"/api/v1/documents/{doc_id}")).json()
    assert doc_data["processing_started_at"] is not None
    assert doc_data["processing_finished_at"] is not None
    assert doc_data["processing_duration_ms"] is not None
    assert doc_data["processing_duration_ms"] > 0

    # 3. Apply corrections to ensure all required fields are valid
    await client.patch(
        f"/api/v1/documents/{doc_id}/fields/company",
        json={"field_name": "company", "corrected_value": "TELEMETRY STORE SDN BHD", "reviewer_name": "operator_test"},
    )
    await client.patch(
        f"/api/v1/documents/{doc_id}/fields/date",
        json={"field_name": "date", "corrected_value": "2026-09-28", "reviewer_name": "operator_test"},
    )
    await client.patch(
        f"/api/v1/documents/{doc_id}/fields/total",
        json={"field_name": "total", "corrected_value": "250.00", "reviewer_name": "operator_test"},
    )

    # 4. Complete review if in manual review
    if doc_data["status"] == DocumentStatus.MANUAL_REVIEW.value:
        comp_res = await client.post(
            f"/api/v1/documents/{doc_id}/review/complete",
            json={"reviewer_name": "operator_test"},
        )
        assert comp_res.status_code == 200
        assert comp_res.json()["status"] == DocumentStatus.COMPLETED_MANUAL.value

    # 5. Query GET /analytics/overview
    overview_res = await client.get("/api/v1/analytics/overview")
    assert overview_res.status_code == 200
    overview = overview_res.json()
    kpis = overview["kpis"]
    assert kpis["total_documents"] >= 1
    assert kpis["processed_count"] >= 1
    assert "automation_rate" in kpis
    assert "manual_review_rate" in kpis
    assert "failure_rate" in kpis
    assert "corrections_total" in kpis
    assert kpis["corrections_total"] >= 1

    # 6. Query GET /analytics/review-reasons
    reasons_res = await client.get("/api/v1/analytics/review-reasons")
    assert reasons_res.status_code == 200
    reasons_data = reasons_res.json()
    assert "total_reviews" in reasons_data
    assert isinstance(reasons_data["items"], list)

    # 7. Query GET /analytics/corrections
    corrections_res = await client.get("/api/v1/analytics/corrections")
    assert corrections_res.status_code == 200
    corr_data = corrections_res.json()
    assert corr_data["total_fields"] >= 1
    assert corr_data["total_corrections"] >= 1
    assert any(item["field"] == "total" for item in corr_data["items"])
    total_field_stat = [i for i in corr_data["items"] if i["field"] == "total"][0]
    assert total_field_stat["corrected_count"] >= 1
    assert total_field_stat["correction_rate"] > 0.0

    # 8. Query GET /analytics/confidence
    confidence_res = await client.get("/api/v1/analytics/confidence")
    assert confidence_res.status_code == 200
    conf_data = confidence_res.json()
    assert len(conf_data["distribution"]) == 5
    bucket_labels = [b["bucket"] for b in conf_data["distribution"]]
    assert "0.00–0.59" in bucket_labels
    assert "0.90–1.00" in bucket_labels
    assert conf_data["average_confidence"] > 0.0
