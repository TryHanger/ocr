"""Tests for MVP-5.2 Degradation Explorer API, calculations, calibration parameters, and artifact immutability."""

import hashlib
from pathlib import Path
import pytest
from httpx import AsyncClient

from app.core.config import settings


@pytest.mark.asyncio
async def test_get_degradations_all_200(client: AsyncClient):
    """Test GET /api/v1/research/degradations returns all 8 degradation experiments and baseline control."""
    res = await client.get("/api/v1/research/degradations")
    assert res.status_code == 200
    data = res.json()

    assert "items" in data
    assert "total" in data
    assert data["total"] == 8
    assert len(data["items"]) == 8

    # Verify Control Condition
    assert "control_condition" in data
    ctrl = data["control_condition"]
    assert ctrl["condition_id"] == "D0_S0_P0"
    assert ctrl["severity"] == "Control"
    assert ctrl["cer"] == 0.3203
    assert ctrl["ned"] == 0.6816
    assert ctrl["wer"] == 0.4831
    assert ctrl["macro_f1"] == 0.4782
    assert ctrl["entity_hmean"] == 0.4979
    assert ctrl["delta_cer"] == 0.0
    assert ctrl["delta_ned"] == 0.0
    assert ctrl["delta_macro_f1"] == 0.0
    assert ctrl["delta_entity_hmean"] == 0.0
    assert ctrl["parameter"] == "none"

    # Verify canonical dataset is dynamic
    assert "126" in data["canonical_dataset"]


@pytest.mark.asyncio
async def test_degradation_items_condition_structure(client: AsyncClient):
    """Test each degradation item contains Control + S1..S4 with structured parameters and findings."""
    res = await client.get("/api/v1/research/degradations")
    assert res.status_code == 200
    items = res.json()["items"]

    expected_codes = {"D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"}
    found_codes = {item["code"] for item in items}
    assert found_codes == expected_codes

    for item in items:
        assert item["id"].startswith("b1_")
        assert item["sample_size"] == 126
        assert len(item["conditions"]) == 5  # Control, S1, S2, S3, S4

        severities = [c["severity"] for c in item["conditions"]]
        assert severities == ["Control", "S1", "S2", "S3", "S4"]

        # Ensure first condition is Control
        c0 = item["conditions"][0]
        assert c0["condition_id"] == "D0_S0_P0"
        assert c0["severity"] == "Control"
        assert c0["delta_cer"] == 0.0

        # Ensure S1..S4 have valid metrics and non-empty parameters
        for c in item["conditions"][1:]:
            assert c["cer"] >= 0.0
            assert c["ned"] >= 0.0
            assert c["macro_f1"] >= 0.0
            assert c["parameters"] is not None
            assert c["parameter"] is not None and c["parameter"] != "none"

        # Check findings
        assert len(item["findings"]) >= 2
        # Inflection finding must be present
        assert any("Largest observed CER change:" in f for f in item["findings"])


@pytest.mark.asyncio
async def test_filter_by_id_and_code(client: AsyncClient):
    """Test filtering by experiment ID, short name, and degradation code."""
    # 1. By full id: b1_gaussian_blur
    res1 = await client.get("/api/v1/research/degradations?degradation=b1_gaussian_blur")
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["total"] == 1
    assert data1["items"][0]["code"] == "D1"
    assert data1["items"][0]["name"] == "Gaussian Blur Robustness"

    # 2. By short id: gaussian_blur
    res2 = await client.get("/api/v1/research/degradations?degradation=gaussian_blur")
    assert res2.status_code == 200
    assert res2.json()["items"][0]["code"] == "D1"

    # 3. By code: D1
    res3 = await client.get("/api/v1/research/degradations?degradation=D1")
    assert res3.status_code == 200
    assert res3.json()["items"][0]["code"] == "D1"

    # 4. By case-insensitive code: d6 (rotation)
    res4 = await client.get("/api/v1/research/degradations?degradation=d6")
    assert res4.status_code == 200
    assert res4.json()["items"][0]["code"] == "D6"
    assert "Rotation" in res4.json()["items"][0]["name"]


@pytest.mark.asyncio
async def test_filter_not_found_and_traversal_protection(client: AsyncClient):
    """Test 404 response on unknown degradation and path traversal attempts."""
    # 1. Unknown degradation name
    res = await client.get("/api/v1/research/degradations?degradation=unknown_defect_99")
    assert res.status_code == 404
    detail = res.json()["detail"]
    assert "not found" in detail
    assert "D1" in detail

    # 2. Path traversal attempt
    res_trav = await client.get("/api/v1/research/degradations?degradation=../../etc/passwd")
    assert res_trav.status_code == 404


@pytest.mark.asyncio
async def test_metric_accuracy_and_calibration_params(client: AsyncClient):
    """Verify exact metrics against B1 CSV artifact and calibration parameters."""
    res = await client.get("/api/v1/research/degradations?degradation=b1_gaussian_blur")
    assert res.status_code == 200
    item = res.json()["items"][0]

    # Verify D1 conditions against known experimental outputs
    # D1_S1_P0: cer=0.3205, d_cer=0.0002, f1=0.4861
    s1 = item["conditions"][1]
    assert s1["condition_id"] == "D1_S1_P0"
    assert s1["cer"] == 0.3205
    assert s1["delta_cer"] == 0.0002
    assert s1["macro_f1"] == 0.4861
    assert s1["parameters"] == {"sigma": 1.0}
    assert s1["parameter"] == "sigma=1.0"

    # D1_S4_P0: cer=0.8608, d_cer=0.5405, f1=0.1171
    s4 = item["conditions"][4]
    assert s4["condition_id"] == "D1_S4_P0"
    assert s4["cer"] == 0.8608
    assert s4["delta_cer"] == 0.5405
    assert s4["macro_f1"] == 0.1171
    assert s4["parameters"] == {"sigma": 5.0}
    assert s4["parameter"] == "sigma=5.0"

    # Verify largest inflection finding contains S3 -> S4 transition
    inflection_f = next(f for f in item["findings"] if "Largest observed CER change:" in f)
    assert "S3 → S4" in inflection_f
    assert "+0.3405" in inflection_f


@pytest.mark.asyncio
async def test_research_artifacts_immutability():
    """Verify cryptographic SHA-256 hashes of B1 research files to guarantee read-only immutability."""
    root = Path(settings.RESEARCH_ROOT_PATH).resolve()

    expected_hashes = {
        "experiments/runs/b1_degraded_validation_n126_gpu/b1_conditions_detailed.csv": "81a6029e4542a199accd65f5fe8bfc0a2667821921a7d2c4f511742d31f11792",
        "experiments/runs/b1_degraded_validation_n126_gpu/summary.json": "cb28cbc3ec572ce085a2eb4e060784431fa7f2d291436c9ae7c7a1101e572c43",
        "experiments/calibration/calibration_report.json": "1e270733f585aee935aeadbd36026cb4804349a56ee2c6a815edb2d0ece2b34f",
    }

    for rel_path, expected_sha in expected_hashes.items():
        file_path = root / rel_path
        assert file_path.exists(), f"Artifact missing: {rel_path}"
        computed_sha = hashlib.sha256(file_path.read_bytes()).hexdigest()
        assert computed_sha == expected_sha, f"Artifact mutated! {rel_path}"
