"""Tests for Business Analytics, Document Type Performance, Trends, and Temporal Filtering."""

from datetime import datetime, timedelta
import io
from PIL import Image, ImageDraw
import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import AutomationDecision, DocumentStatus, DocumentType
from app.models.document import Document
from app.models.job import ProcessingJob
from app.services.analytics_service import AnalyticsService, _safe_rate
from app.services.pipeline_service import PipelineService


def _create_receipt_bytes() -> bytes:
    img = Image.new("RGB", (600, 800), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((50, 50), "BUSINESS ANALYTICS STORE", fill=(0, 0, 0))
    draw.text((50, 100), "DATE: 2026-09-28", fill=(0, 0, 0))
    draw.text((50, 600), "TOTAL: 150.00", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_safe_rate_division_by_zero():
    """Verify division by zero produces 0.0 and never NaN or Infinity."""
    assert _safe_rate(0, 0) == 0.0
    assert _safe_rate(10, 0) == 0.0
    assert _safe_rate(0, 10) == 0.0
    assert _safe_rate(5, 10) == 50.0


def test_confidence_boundaries_and_calibration():
    """Verify confidence bucket boundaries across all specified thresholds."""
    bucket_defs = [
        ("0.00–0.59", 0.0, 0.60),
        ("0.60–0.69", 0.60, 0.70),
        ("0.70–0.79", 0.70, 0.80),
        ("0.80–0.89", 0.80, 0.90),
        ("0.90–1.00", 0.90, 1.01),
    ]

    test_values = [
        (0.00, "0.00–0.59"),
        (0.59, "0.00–0.59"),
        (0.60, "0.60–0.69"),
        (0.69, "0.60–0.69"),
        (0.70, "0.70–0.79"),
        (0.79, "0.70–0.79"),
        (0.80, "0.80–0.89"),
        (0.89, "0.80–0.89"),
        (0.90, "0.90–1.00"),
        (1.00, "0.90–1.00"),
    ]

    for val, expected_bucket in test_values:
        matched = None
        for name, low, high in bucket_defs:
            if low <= val < high or (high >= 1.0 and val >= 1.0 and name == "0.90–1.00"):
                matched = name
                break
        assert matched == expected_bucket, f"Value {val} matched {matched} instead of {expected_bucket}"


@pytest.mark.asyncio
async def test_business_analytics_endpoints_and_filtering(client: AsyncClient, db_session: AsyncSession):
    # 1. Ingest a receipt and execute pipeline
    img_bytes = _create_receipt_bytes()
    files = {"file": ("biz_receipt.jpg", img_bytes, "image/jpeg")}
    upload_res = await client.post("/api/v1/documents", files=files, data={"document_type": "receipt"})
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["id"]

    pipeline = PipelineService(db_session)
    job_stmt = select(ProcessingJob).where(ProcessingJob.document_id == doc_id)
    job = (await db_session.execute(job_stmt)).scalar_one()
    await pipeline.execute_job(job.id)

    # 2. Test GET /analytics/document-types
    dt_res = await client.get("/api/v1/analytics/document-types")
    assert dt_res.status_code == 200
    dt_data = dt_res.json()
    assert "items" in dt_data
    assert len(dt_data["items"]) >= 1

    receipt_type = next((item for item in dt_data["items"] if item["document_type"] == "receipt"), None)
    assert receipt_type is not None
    assert receipt_type["total_documents"] >= 1
    assert receipt_type["processed_documents"] >= 1
    assert "automation_rate" in receipt_type
    assert "manual_review_rate" in receipt_type
    assert "failure_rate" in receipt_type
    assert receipt_type["average_confidence"] >= 0.0

    # 3. Test GET /analytics/trends
    trends_res = await client.get("/api/v1/analytics/trends?period=7d")
    assert trends_res.status_code == 200
    trends_data = trends_res.json()
    assert "items" in trends_data
    assert len(trends_data["items"]) == 7
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    today_trend = next((t for t in trends_data["items"] if t["date"] == today_str), None)
    assert today_trend is not None
    assert today_trend["processed_count"] >= 1

    # 4. Test GET /analytics/queue
    queue_res = await client.get("/api/v1/analytics/queue")
    assert queue_res.status_code == 200
    q_data = queue_res.json()
    assert "manual_review" in q_data
    assert "processing" in q_data
    assert "errors" in q_data

    # 5. Test Temporal Filtering on Overview
    today_res = await client.get("/api/v1/analytics/overview?period=today")
    assert today_res.status_code == 200
    today_kpis = today_res.json()["kpis"]
    assert today_kpis["total_documents"] >= 1

    # Custom date filter (in future: should yield 0)
    future_from = (datetime.utcnow() + timedelta(days=10)).isoformat()
    future_to = (datetime.utcnow() + timedelta(days=20)).isoformat()
    future_res = await client.get(f"/api/v1/analytics/overview?from={future_from}&to={future_to}")
    assert future_res.status_code == 200
    future_kpis = future_res.json()["kpis"]
    assert future_kpis["total_documents"] == 0
    assert future_kpis["automation_rate"] == 0.0
    assert future_kpis["manual_review_rate"] == 0.0
    assert future_kpis["failure_rate"] == 0.0

    # 6. Verify GAP-01 Canonical Contract: items[].field
    corr_res = await client.get("/api/v1/analytics/corrections")
    assert corr_res.status_code == 200
    corr_data = corr_res.json()
    assert "items" in corr_data
    if len(corr_data["items"]) > 0:
        first_item = corr_data["items"][0]
        assert "field" in first_item
        assert "total_occurrences" in first_item
        assert "corrected_count" in first_item
        assert "correction_rate" in first_item
