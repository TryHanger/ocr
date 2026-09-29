"""Tests for MVP-5.4 Operations ↔ Research Evidence Bridge."""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.quality import QualityResultDTO
from app.schemas.research import ResearchEvidenceReference
from app.services.quality_bridge import default_evidence_bridge, ResearchEvidenceBridge
from app.services.research_service import ResearchService


def test_mapping_registry_valid_signals():
    """Test 1: Verify production signals with verified research counterparts resolve correctly."""
    bridge = ResearchEvidenceBridge()

    # Rotation -> D6
    rot_map = bridge.get_mapping("rotation")
    assert rot_map is not None
    assert rot_map["research_degradation_code"] == "D6"
    assert rot_map["b1_experiment_id"] == "b1_rotation"
    assert rot_map["b1_available"] is True
    assert rot_map["b2_degradation_code"] == "D6"
    assert rot_map["b2_available"] is True
    assert rot_map["b2_evaluated_policies_count"] == 4

    # Noise -> D3
    noise_map = bridge.get_mapping("noise")
    assert noise_map is not None
    assert noise_map["research_degradation_code"] == "D3"
    assert noise_map["b1_experiment_id"] == "b1_gaussian_noise"
    assert noise_map["b1_available"] is True
    assert noise_map["b2_degradation_code"] == "D3"
    assert noise_map["b2_available"] is True

    # Blur -> D1
    blur_map = bridge.get_mapping("blur")
    assert blur_map is not None
    assert blur_map["research_degradation_code"] == "D1"
    assert blur_map["b1_experiment_id"] == "b1_gaussian_blur"
    assert blur_map["b1_available"] is True
    assert blur_map["b2_degradation_code"] == "D1"
    assert blur_map["b2_available"] is True

    # Resolution -> D5
    res_map = bridge.get_mapping("resolution")
    assert res_map is not None
    assert res_map["research_degradation_code"] == "D5"
    assert res_map["b1_experiment_id"] == "b1_downsampling"
    assert res_map["b1_available"] is True
    assert res_map["b2_degradation_code"] == "D5"
    assert res_map["b2_available"] is True


def test_no_unsupported_mappings():
    """Test 2: Verify signals without research evidence do not receive fabricated references."""
    bridge = ResearchEvidenceBridge()

    # Contrast exists in production QualityAnalyzer, but has NO standalone B1 degradation dimension
    assert bridge.get_mapping("contrast") is None
    assert bridge.create_reference("contrast", observed_value=22.5) is None

    # Unsupported / non-existent signals
    assert bridge.get_mapping("unknown_signal") is None
    assert bridge.get_mapping("color_jitter") is None
    assert bridge.get_mapping("perspective") is None  # Evaluated in B1, but not detected in production QualityAnalyzer


def test_evidence_dto_structure():
    """Test 3: Verify the Evidence Reference DTO structure and serialization."""
    bridge = ResearchEvidenceBridge()
    ref = bridge.create_reference("rotation", observed_value=4.2)
    assert ref is not None
    assert isinstance(ref, ResearchEvidenceReference)

    # Flat properties
    assert ref.production_signal == "rotation"
    assert ref.observed_value == 4.2
    assert ref.unit == "degrees"
    assert ref.research_degradation_code == "D6"
    assert ref.research_degradation_name == "Rotation & Skew Robustness"
    assert ref.b1_experiment_id == "b1_rotation"
    assert ref.b1_available is True
    assert ref.b2_degradation_code == "D6"
    assert ref.b2_available is True
    assert ref.b2_evaluated_policies_count == 4

    # Nested structures for client convenience
    assert ref.research_degradation is not None
    assert ref.research_degradation.code == "D6"
    assert ref.research_degradation.name == "Rotation & Skew Robustness"
    assert ref.b1 is not None
    assert ref.b1.experiment_id == "b1_rotation"
    assert ref.b1.available is True
    assert ref.b2 is not None
    assert ref.b2.available is True
    assert "4 compound preprocessing policies" in ref.b2.details


def test_no_severity_inference():
    """Test 4: Verify the API does NOT infer S1..S4 severity from arbitrary observed quality values."""
    bridge = ResearchEvidenceBridge()

    for test_angle in [0.5, 1.2, 4.2, 8.7, 15.0, -3.4]:
        ref = bridge.create_reference("rotation", observed_value=test_angle)
        assert ref is not None
        dto_dict = ref.model_dump()

        # DTO must NOT have any severity fields
        assert "severity" not in dto_dict
        assert "inferred_severity" not in dto_dict
        assert "target_severity" not in dto_dict

        # Value must remain strictly the observed continuous angle
        assert ref.observed_value == round(test_angle, 2)


def test_b1_registry_consistency():
    """Test 5: Verify valid mappings resolve to actual B1 experiments in ResearchService."""
    research_service = ResearchService()
    bridge = ResearchEvidenceBridge()

    for sig, mapping in bridge.SIGNAL_REGISTRY.items():
        b1_id = mapping["b1_experiment_id"]
        assert b1_id in research_service.registry, f"B1 experiment '{b1_id}' for signal '{sig}' not in registry."
        exp_entry = research_service.registry[b1_id]
        assert exp_entry["track"] == "B1"
        assert exp_entry["deg_code"] == mapping["research_degradation_code"]


def test_b2_degradation_coverage_semantics():
    """Test 6: Verify B2 correspondence points to degradation dimension coverage, not an isolated policy."""
    research_service = ResearchService()
    bridge = ResearchEvidenceBridge()

    # B2 registry has 4 compound policies
    b2_policies = [k for k, v in research_service.registry.items() if v.get("track") == "B2"]
    assert len(b2_policies) == 4

    for sig, mapping in bridge.SIGNAL_REGISTRY.items():
        b2_code = mapping["b2_degradation_code"]
        assert b2_code in ("D1", "D3", "D5", "D6"), f"Unknown B2 degradation code: {b2_code}"
        assert mapping["b2_available"] is True
        assert mapping["b2_evaluated_policies_count"] == 4

        ref = bridge.create_reference(sig, observed_value=1.0)
        assert ref is not None
        # B2 does NOT prescribe an individual policy experiment ID
        assert ref.b2.experiment_id is None
        assert ref.b2_degradation_code == b2_code


def test_existing_production_detection_semantics():
    """Test 7: Verify evidence is generated using existing production detection semantics."""
    bridge = ResearchEvidenceBridge()

    # Case A: Document with rotation >= 2.0 (production needs_deskew threshold)
    quality_rotated = QualityResultDTO(
        document_id="doc-rot",
        blur_score=180.0,  # Sharp
        contrast_score=45.0,  # Normal
        noise_score=2.0,  # Clean
        rotation_angle=4.2,  # >= 2.0 (needs_deskew)
        resolution_dpi=300.0,
        quality_score=0.82,
        profile_summary="High Quality",
        recommendations={"needs_deskew": True, "needs_clahe": False, "needs_denoise": False},
    )
    evidence_rot = bridge.get_evidence_for_quality(quality_rotated)
    assert len(evidence_rot) == 1
    assert evidence_rot[0].production_signal == "rotation"
    assert evidence_rot[0].research_degradation_code == "D6"
    assert evidence_rot[0].observed_value == 4.2

    # Case B: Document with sub-threshold rotation (e.g. 0.8°) - must NOT trigger detection
    quality_slight_rot = QualityResultDTO(
        document_id="doc-slight",
        blur_score=180.0,
        contrast_score=45.0,
        noise_score=2.0,
        rotation_angle=0.8,  # < 2.0
        resolution_dpi=300.0,
        quality_score=0.95,
        profile_summary="High Quality",
        recommendations={"needs_deskew": False, "needs_clahe": False, "needs_denoise": False},
    )
    evidence_slight = bridge.get_evidence_for_quality(quality_slight_rot)
    assert len(evidence_slight) == 0

    # Case C: Document with noise > 12.0 (production needs_denoise threshold)
    quality_noisy = QualityResultDTO(
        document_id="doc-noisy",
        blur_score=180.0,
        contrast_score=45.0,
        noise_score=16.5,  # > 12.0 (needs_denoise)
        rotation_angle=0.1,
        resolution_dpi=300.0,
        quality_score=0.60,
        profile_summary="Acceptable",
        recommendations={"needs_deskew": False, "needs_clahe": False, "needs_denoise": True},
    )
    evidence_noisy = bridge.get_evidence_for_quality(quality_noisy)
    assert len(evidence_noisy) == 1
    assert evidence_noisy[0].production_signal == "noise"
    assert evidence_noisy[0].research_degradation_code == "D3"

    # Case D: Document with blur < 30.0 (documented blur_factor = 0.0 threshold)
    quality_blurry = QualityResultDTO(
        document_id="doc-blurry",
        blur_score=18.0,  # < 30.0
        contrast_score=45.0,
        noise_score=2.0,
        rotation_angle=0.1,
        resolution_dpi=300.0,
        quality_score=0.45,
        profile_summary="Degraded",
    )
    evidence_blurry = bridge.get_evidence_for_quality(quality_blurry)
    assert len(evidence_blurry) == 1
    assert evidence_blurry[0].production_signal == "blur"
    assert evidence_blurry[0].research_degradation_code == "D1"

    # Case E: Clean pristine document -> empty evidence list
    quality_clean = QualityResultDTO(
        document_id="doc-clean",
        blur_score=220.0,
        contrast_score=50.0,
        noise_score=1.5,
        rotation_angle=0.0,
        resolution_dpi=300.0,
        quality_score=0.98,
        profile_summary="High Quality",
    )
    assert bridge.get_evidence_for_quality(quality_clean) == []

    # Case F: None quality -> empty evidence list
    assert bridge.get_evidence_for_quality(None) == []


@pytest.mark.asyncio
async def test_document_detail_endpoint_with_evidence(client: AsyncClient, db_session: AsyncSession):
    """Test 8: Verify GET /api/v1/documents/{id} returns typed research evidence when characteristics are detected."""
    import uuid
    from app.models.document import Document
    from app.models.quality import DocumentQuality

    doc_id = str(uuid.uuid4())
    doc = Document(
        id=doc_id,
        filename="test_degraded.jpg",
        file_path="mock/path.jpg",
        file_size_bytes=1024,
        mime_type="image/jpeg",
        status="completed_automatic",
    )
    db_session.add(doc)

    quality = DocumentQuality(
        id=str(uuid.uuid4()),
        document_id=doc_id,
        blur_score=150.0,
        contrast_score=40.0,
        noise_score=2.0,
        rotation_angle=4.2,  # Detected rotation
        resolution_dpi=300.0,
        quality_score=0.78,
        profile_summary="Acceptable",
        metrics_json={"recommendations": {"needs_deskew": True}},
    )
    db_session.add(quality)
    await db_session.commit()

    detail_res = await client.get(f"/api/v1/documents/{doc_id}")
    assert detail_res.status_code == 200
    detail = detail_res.json()

    assert "research_evidence" in detail
    assert isinstance(detail["research_evidence"], list)
    assert len(detail["research_evidence"]) >= 1

    rot_ev = next((e for e in detail["research_evidence"] if e["production_signal"] == "rotation"), None)
    assert rot_ev is not None
    assert rot_ev["research_degradation_code"] == "D6"
    assert rot_ev["research_degradation_name"] == "Rotation & Skew Robustness"
    assert rot_ev["observed_value"] == 4.2
    assert rot_ev["b1_available"] is True
    assert rot_ev["b1_experiment_id"] == "b1_rotation"
    assert rot_ev["b2_available"] is True
    assert rot_ev["b2_degradation_code"] == "D6"


@pytest.mark.asyncio
async def test_document_detail_endpoint_clean_doc(client: AsyncClient, db_session: AsyncSession):
    """Test 9: Verify clean document without detected degradations returns research_evidence == []."""
    import uuid
    from app.models.document import Document
    from app.models.quality import DocumentQuality

    doc_id = str(uuid.uuid4())
    doc = Document(
        id=doc_id,
        filename="test_clean.jpg",
        file_path="mock/clean.jpg",
        file_size_bytes=1024,
        mime_type="image/jpeg",
        status="completed_automatic",
    )
    db_session.add(doc)

    quality = DocumentQuality(
        id=str(uuid.uuid4()),
        document_id=doc_id,
        blur_score=250.0,  # Sharp
        contrast_score=50.0,
        noise_score=1.2,  # Clean
        rotation_angle=0.0,  # No rotation
        resolution_dpi=300.0,
        quality_score=0.98,
        profile_summary="High Quality",
        metrics_json={"recommendations": {"needs_deskew": False, "needs_denoise": False}},
    )
    db_session.add(quality)
    await db_session.commit()

    detail_res = await client.get(f"/api/v1/documents/{doc_id}")
    assert detail_res.status_code == 200
    detail = detail_res.json()

    assert "research_evidence" in detail
    assert detail["research_evidence"] == []

