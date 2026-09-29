"""Tests for ConfidenceEngine and DecisionEngine."""

from app.core.enums import AutomationDecision, ReviewReason
from app.schemas.kie import NormalizedFieldDTO
from app.schemas.ocr import NormalizedTokenDTO
from app.schemas.quality import QualityResultDTO
from app.services.confidence_engine import ConfidenceEngine
from app.services.decision_engine import DecisionEngine


def test_confidence_calculations():
    tokens = [
        NormalizedTokenDTO(text="A", confidence=0.9, bbox=[0, 0, 1, 1]),
        NormalizedTokenDTO(text="B", confidence=0.8, bbox=[0, 0, 1, 1]),
    ]
    assert ConfidenceEngine.calculate_ocr_confidence(tokens) == 0.85

    fields = {
        "company": NormalizedFieldDTO(field_name="company", value="ABC", confidence=0.95),
        "total": NormalizedFieldDTO(field_name="total", value="10.00", confidence=0.90),
    }

    doc_conf = ConfidenceEngine.calculate_document_confidence(
        fields=fields,
        ocr_tokens=tokens,
        image_quality_score=0.8,
    )
    assert 0.85 <= doc_conf <= 0.95


def test_decision_engine_automatic():
    fields = {
        "company": NormalizedFieldDTO(field_name="company", value="ABC SDN BHD", confidence=0.95),
        "date": NormalizedFieldDTO(field_name="date", value="2026-09-28", confidence=0.92),
        "total": NormalizedFieldDTO(field_name="total", value="150.00", confidence=0.97),
        "address": NormalizedFieldDTO(field_name="address", value="Kuala Lumpur", confidence=0.88),
    }
    quality = QualityResultDTO(
        document_id="doc1",
        blur_score=180.0,
        contrast_score=45.0,
        noise_score=3.0,
        rotation_angle=0.2,
        resolution_dpi=300.0,
        quality_score=0.90,
        profile_summary="High Quality",
    )

    decision, reason, _, details = DecisionEngine.evaluate("receipt", fields, quality, 0.94)
    assert decision == AutomationDecision.AUTOMATIC
    assert reason == ReviewReason.NONE
    assert details is None


def test_decision_engine_missing_field():
    fields = {
        "company": NormalizedFieldDTO(field_name="company", value="ABC SDN BHD", confidence=0.95),
        # date missing
        "total": NormalizedFieldDTO(field_name="total", value="150.00", confidence=0.97),
    }
    decision, reason, _, details = DecisionEngine.evaluate("receipt", fields, None, 0.90)
    assert decision == AutomationDecision.MANUAL_REVIEW
    assert reason == ReviewReason.MISSING_REQUIRED_FIELD
    assert details is not None
    assert any(err["field"] == "date" for err in details["validation_errors"])


def test_decision_engine_low_confidence():
    fields = {
        "company": NormalizedFieldDTO(field_name="company", value="ABC SDN BHD", confidence=0.95),
        "date": NormalizedFieldDTO(field_name="date", value="2026-09-28", confidence=0.92),
        "total": NormalizedFieldDTO(field_name="total", value="150.00", confidence=0.60),  # below 0.95
    }
    decision, reason, _, details = DecisionEngine.evaluate("receipt", fields, None, 0.70)
    assert decision == AutomationDecision.MANUAL_REVIEW
    assert reason == ReviewReason.LOW_CONFIDENCE
    assert details is not None
    assert details["fields"]["total"]["status"] == "review"
