"""Unit and integration tests for MVP-9 Operational Action & Resolution Center."""

from datetime import datetime
import json
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import AuditAction, AutomationDecision, DocumentStatus, ReviewReason
from app.main import app
from app.models.audit import AuditEvent
from app.models.document import Document
from app.models.field import ExtractedField
from app.models.job import ProcessingJob
from app.models.quality import DocumentQuality
from app.schemas.operations import (
    ActionType,
    IssueCategory,
    IssueSeverity,
    ResolutionStatus,
)
from app.services.operational_issues_service import (
    ActionRegistry,
    OperationalIssuesService,
)


@pytest.mark.asyncio
async def test_action_registry_declarative_actions():
    """Verify ActionRegistry produces safe declarative actions without mutations."""
    # Low confidence on a field
    actions = ActionRegistry.get_actions_for_issue(
        code=ReviewReason.LOW_CONFIDENCE.value,
        category=IssueCategory.CONFIDENCE,
        document_id="doc-123",
        field_name="total",
        can_review=True,
        can_retry=True,
        status=ResolutionStatus.OPEN,
    )

    action_types = [a.type for a in actions]
    assert ActionType.OPEN_FIELD in action_types
    assert ActionType.OPEN_REVIEW in action_types
    assert ActionType.OPEN_DOCUMENT in action_types
    # Retry must NOT be offered for low confidence in a deterministic pipeline
    assert ActionType.RETRY_PROCESSING not in action_types

    field_act = next(a for a in actions if a.type == ActionType.OPEN_FIELD)
    assert field_act.target == "fields"
    assert field_act.params.get("field_name") == "total"

    # Pipeline failure issue: retry MUST be offered
    pipe_actions = ActionRegistry.get_actions_for_issue(
        code=ReviewReason.PIPELINE_FAILURE.value,
        category=IssueCategory.PROCESSING,
        document_id="doc-123",
        can_review=False,
        can_retry=True,
        status=ResolutionStatus.OPEN,
    )
    pipe_types = [a.type for a in pipe_actions]
    assert ActionType.RETRY_PROCESSING in pipe_types
    assert ActionType.OPEN_DOCUMENT in pipe_types


    # Quality degradation with research condition
    q_actions = ActionRegistry.get_actions_for_issue(
        code=ReviewReason.LOW_IMAGE_QUALITY.value,
        category=IssueCategory.QUALITY,
        document_id="doc-123",
        condition_code="D1",
        can_review=True,
        can_retry=False,
        status=ResolutionStatus.OPEN,
    )
    q_types = [a.type for a in q_actions]
    assert ActionType.OPEN_QUALITY_EVIDENCE in q_types
    assert ActionType.OPEN_RESEARCH_EVIDENCE in q_types

    res_act = next(a for a in q_actions if a.type == ActionType.OPEN_RESEARCH_EVIDENCE)
    assert res_act.target == "research"
    assert res_act.params.get("track") == "B2"
    assert res_act.params.get("condition_code") == "D1"


@pytest.mark.asyncio
async def test_derive_resolution_status():
    """Verify deterministic derivation of resolution status."""
    now = datetime(2026, 9, 29, 8, 0, 0)

    # 1. Automatic straight-through -> RESOLVED
    doc_auto = Document(
        id="d1",
        filename="rec.jpg",
        file_path="/tmp/rec.jpg",
        file_size_bytes=100,
        mime_type="image/jpeg",
        status=DocumentStatus.COMPLETED_AUTOMATIC.value,
        created_at=now,
    )
    assert OperationalIssuesService.derive_resolution_status(doc_auto) == ResolutionStatus.RESOLVED

    # 2. Manual review, not yet picked up -> OPEN
    doc_open = Document(
        id="d2",
        filename="rec.jpg",
        file_path="/tmp/rec.jpg",
        file_size_bytes=100,
        mime_type="image/jpeg",
        status=DocumentStatus.MANUAL_REVIEW.value,
        review_started_at=None,
        created_at=now,
    )
    assert OperationalIssuesService.derive_resolution_status(doc_open) == ResolutionStatus.OPEN

    # 3. Manual review in progress -> IN_REVIEW
    doc_in_rev = Document(
        id="d3",
        filename="rec.jpg",
        file_path="/tmp/rec.jpg",
        file_size_bytes=100,
        mime_type="image/jpeg",
        status=DocumentStatus.MANUAL_REVIEW.value,
        review_started_at=now,
        review_finished_at=None,
        created_at=now,
    )
    assert OperationalIssuesService.derive_resolution_status(doc_in_rev) == ResolutionStatus.IN_REVIEW

    # 4. Completed manual review -> RESOLVED
    doc_manual_done = Document(
        id="d4",
        filename="rec.jpg",
        file_path="/tmp/rec.jpg",
        file_size_bytes=100,
        mime_type="image/jpeg",
        status=DocumentStatus.COMPLETED_MANUAL.value,
        review_started_at=now,
        review_finished_at=now,
        created_at=now,
    )
    assert OperationalIssuesService.derive_resolution_status(doc_manual_done) == ResolutionStatus.RESOLVED


@pytest.mark.asyncio
async def test_no_false_positives_for_automatic_document():
    """Verify that a valid automatic document generates zero blocking operational issues."""
    doc = Document(
        id="doc-clean",
        filename="clean.jpg",
        file_path="/tmp/clean.jpg",
        file_size_bytes=1024,
        mime_type="image/jpeg",
        status=DocumentStatus.COMPLETED_AUTOMATIC.value,
        decision=AutomationDecision.AUTOMATIC.value,
        confidence=0.95,
        review_reason=ReviewReason.NONE.value,
        created_at=datetime(2026, 9, 29, 8, 0, 0),
        processing_finished_at=datetime(2026, 9, 29, 8, 0, 1),
    )

    resp = OperationalIssuesService.derive_document_issues(doc)
    assert resp.overall_status == ResolutionStatus.RESOLVED
    assert resp.total_issues == 0
    assert resp.blocking_issues == 0
    assert "straight-through" in resp.summary.lower() or "zero operational issues" in resp.summary.lower()


@pytest.mark.asyncio
async def test_issue_derivation_from_review_reasons():
    """Verify issue derivation from review details and canonical reasons."""
    created_ts = datetime(2026, 9, 29, 8, 15, 0)
    eval_ts = datetime(2026, 9, 29, 8, 15, 2)

    doc = Document(
        id="doc-issues",
        filename="degraded.jpg",
        file_path="/tmp/degraded.jpg",
        file_size_bytes=2048,
        mime_type="image/jpeg",
        status=DocumentStatus.MANUAL_REVIEW.value,
        decision=AutomationDecision.MANUAL_REVIEW.value,
        confidence=0.65,
        review_reason=ReviewReason.LOW_CONFIDENCE.value,
        review_details_json={
            "reasons": [
                {
                    "code": ReviewReason.LOW_CONFIDENCE.value,
                    "field": "total",
                    "confidence": 0.65,
                    "threshold": 0.85,
                    "message": "Field 'total' extraction confidence (0.65) below threshold (0.85).",
                },
                {
                    "code": ReviewReason.MISSING_REQUIRED_FIELD.value,
                    "field": "date",
                    "message": "Mandatory field 'date' was not detected in document.",
                },
            ]
        },
        created_at=created_ts,
        processing_finished_at=eval_ts,
    )

    # Attach audit event to verify timestamp derivation
    doc.audit_events = [
        AuditEvent(
            id="ev-1",
            document_id=doc.id,
            action=AuditAction.DECISION_EVALUATED.value,
            actor="decision_engine",
            created_at=eval_ts,
            metadata_json={"reason": ReviewReason.LOW_CONFIDENCE.value},
        )
    ]

    hist_summary = {
        ReviewReason.LOW_CONFIDENCE.value: {
            "total": 15,
            "corrections": 9,
            "no_corrections": 6,
        }
    }

    resp = OperationalIssuesService.derive_document_issues(doc, historical_summary=hist_summary)

    assert resp.overall_status == ResolutionStatus.OPEN
    assert resp.total_issues == 2
    assert resp.blocking_issues == 2

    # Check Low Confidence issue
    iss_conf = next(i for i in resp.issues if i.code == ReviewReason.LOW_CONFIDENCE.value)
    assert iss_conf.category == IssueCategory.CONFIDENCE
    assert iss_conf.severity == IssueSeverity.HIGH
    assert iss_conf.field == "total"
    assert iss_conf.observed_value == 0.65
    assert iss_conf.threshold == 0.85
    assert iss_conf.created_at == eval_ts
    assert "15 similar manual review(s)" in iss_conf.evidence.historical
    assert "9 contained correction(s)" in iss_conf.evidence.historical

    # Check Missing Required Field issue
    iss_val = next(i for i in resp.issues if i.code == ReviewReason.MISSING_REQUIRED_FIELD.value)
    assert iss_val.category == IssueCategory.VALIDATION
    assert iss_val.severity == IssueSeverity.HIGH
    assert iss_val.field == "date"
    assert iss_val.created_at == eval_ts


@pytest.mark.asyncio
async def test_retry_safety_no_mutation():
    """Verify derive_document_issues never triggers retry or alters document state."""
    doc = Document(
        id="doc-safety",
        filename="failed.jpg",
        file_path="/tmp/failed.jpg",
        file_size_bytes=512,
        mime_type="image/jpeg",
        status=DocumentStatus.ERROR.value,
        review_reason=ReviewReason.PIPELINE_FAILURE.value,
        created_at=datetime(2026, 9, 29, 8, 0, 0),
    )
    doc.jobs = [
        ProcessingJob(
            id="job-1",
            document_id=doc.id,
            status="failed",
            error_message="Image decoding failed",
            created_at=datetime(2026, 9, 29, 8, 0, 0),
        )
    ]

    resp = OperationalIssuesService.derive_document_issues(doc)

    # Status must still be ERROR and exactly 1 job must exist
    assert doc.status == DocumentStatus.ERROR.value
    assert len(doc.jobs) == 1
    assert resp.issues[0].severity == IssueSeverity.CRITICAL
    assert any(a.type == ActionType.RETRY_PROCESSING for a in resp.issues[0].available_actions)


@pytest.mark.asyncio
async def test_research_isolation_no_forbidden_imports():
    """Verify research isolation in operational service."""
    import sys
    from app.services import operational_issues_service

    mod_dict = sys.modules[operational_issues_service.__name__].__dict__
    # Check that src. or experiments. are not imported in operational_issues_service
    for k, v in mod_dict.items():
        if hasattr(v, "__name__"):
            assert not v.__name__.startswith("src."), f"Forbidden research import: {v.__name__}"
            assert not v.__name__.startswith("experiments."), f"Forbidden research import: {v.__name__}"


@pytest.mark.asyncio
async def test_operations_api_endpoints(client: AsyncClient, db_session: AsyncSession):
    """Integration test for Operations REST API endpoints."""
    # 1. Ingest a test document in manual review state
    doc = Document(
        id="test-op-doc-1",
        filename="invoice_review.jpg",
        file_path="/tmp/inv.jpg",
        file_size_bytes=1024,
        mime_type="image/jpeg",
        document_type="invoice",
        status=DocumentStatus.MANUAL_REVIEW.value,
        decision=AutomationDecision.MANUAL_REVIEW.value,
        confidence=0.72,
        review_reason=ReviewReason.LOW_CONFIDENCE.value,
        review_details_json={
            "reasons": [
                {
                    "code": ReviewReason.LOW_CONFIDENCE.value,
                    "field": "total",
                    "confidence": 0.72,
                    "threshold": 0.85,
                    "message": "Field 'total' confidence 0.72 below threshold 0.85",
                }
            ]
        },
        created_at=datetime(2026, 9, 29, 8, 30, 0),
    )
    db_session.add(doc)
    await db_session.commit()

    # 1. GET /api/v1/operations/documents/{id}/issues
    resp = await client.get(f"/api/v1/operations/documents/{doc.id}/issues")
    assert resp.status_code == 200
    data = resp.json()
    assert data["document_id"] == doc.id
    assert data["overall_status"] == "open"
    assert data["total_issues"] == 1
    assert data["blocking_issues"] == 1
    assert data["issues"][0]["category"] == "confidence"
    assert data["issues"][0]["field"] == "total"

    # 2. GET /api/v1/operations/documents/{id}/actions
    actions_resp = await client.get(f"/api/v1/operations/documents/{doc.id}/actions")
    assert actions_resp.status_code == 200
    actions = actions_resp.json()
    action_types = [a["type"] for a in actions]
    assert "open_field" in action_types
    assert "open_review" in action_types

    # 3. GET /api/v1/operations/backlog
    backlog_resp = await client.get("/api/v1/operations/backlog")
    assert backlog_resp.status_code == 200
    backlog = backlog_resp.json()
    assert backlog["total_open"] >= 1
    assert ReviewReason.LOW_CONFIDENCE.value in backlog["by_reason"]
    assert "confidence" in backlog["by_category"]

