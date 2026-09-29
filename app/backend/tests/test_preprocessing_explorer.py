"""Tests for MVP-5.3 Preprocessing Explorer API, calculations, baseline references, and artifact immutability."""

import csv
import hashlib
from pathlib import Path
import pytest
from httpx import AsyncClient

from app.core.config import settings


@pytest.mark.asyncio
async def test_get_preprocessing_all_200(client: AsyncClient):
    """Test GET /api/v1/research/preprocessing returns all 32 condition groups across 128 evaluations."""
    res = await client.get("/api/v1/research/preprocessing")
    assert res.status_code == 200
    data = res.json()

    assert "groups" in data
    assert "total_groups" in data
    assert data["total_groups"] == 32
    assert len(data["groups"]) == 32
    assert data["total_conditions_evaluated"] == 128
    assert "126" in data["canonical_dataset"]

    # Verify degradations list
    assert len(data["degradations"]) == 8
    deg_codes = {d["code"] for d in data["degradations"]}
    assert deg_codes == {"D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"}

    # Verify severities list
    assert data["severities"] == ["S1", "S2", "S3", "S4"]

    # Verify policies list
    assert len(data["policies"]) == 4
    policy_keys = {p["key"] for p in data["policies"]}
    assert policy_keys == {
        "p_standard_receipt_enhancement",
        "p_contrast_enhancement",
        "p_minimal_cleanup",
        "p_aggressive_binarization",
    }


@pytest.mark.asyncio
async def test_preprocessing_group_structure_and_baselines(client: AsyncClient):
    """Test every condition group contains clean control, degraded baseline, and 4 policy metrics."""
    res = await client.get("/api/v1/research/preprocessing")
    assert res.status_code == 200
    groups = res.json()["groups"]

    for g in groups:
        assert g["degradation_code"] in {"D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"}
        assert g["severity"] in {"S1", "S2", "S3", "S4"}
        assert g["ref_b1_condition_id"].startswith(g["degradation_code"])

        # Clean control baseline: Pristine RapidOCR / RuleBasedKIE
        ctrl = g["clean_control_metrics"]
        assert ctrl["cer"] == 0.3203
        assert ctrl["ned"] == 0.6816
        assert ctrl["wer"] == 0.4831
        assert ctrl["macro_f1"] == 0.4782
        assert ctrl["entity_hmean"] == 0.4979

        # Degraded baseline
        deg = g["degraded_metrics"]
        assert deg["cer"] >= 0.0
        assert deg["ned"] >= 0.0
        assert deg["macro_f1"] >= 0.0

        # Policies
        assert len(g["policies"]) == 4
        for pol in g["policies"]:
            assert pol["condition_id"].startswith(g["degradation_code"])
            assert pol["metrics"]["cer"] >= 0.0
            assert pol["metrics"]["ned"] >= 0.0
            assert pol["metrics"]["macro_f1"] >= 0.0
            assert pol["metrics"]["entity_hmean"] >= 0.0

        # Non-empty findings
        assert len(g["findings"]) >= 1


@pytest.mark.asyncio
async def test_metric_accuracy_against_frozen_b2_csv(client: AsyncClient):
    """Dynamically read frozen B2 CSV and verify API returns exact values without hardcoding."""
    root = Path(settings.RESEARCH_ROOT_PATH).resolve()
    csv_path = root / "experiments/runs/b2_preprocessing_validation_n126_gpu/b2_conditions_detailed.csv"
    assert csv_path.exists()

    with open(csv_path, "r", encoding="utf-8") as f:
        b2_rows = list(csv.DictReader(f))

    # Test Rotation S2 condition (D6_S2)
    res = await client.get("/api/v1/research/preprocessing?degradation=D6&severity=S2")
    assert res.status_code == 200
    data = res.json()
    assert data["total_groups"] == 1
    group = data["groups"][0]
    assert group["degradation_code"] == "D6"
    assert group["severity"] == "S2"

    for pol in group["policies"]:
        # Find matching row in CSV
        row = next(
            r for r in b2_rows
            if r["degradation"] == "rotation"
            and r["severity"] == "2"
            and r["preprocessing"] == pol["policy_key"]
        )
        expected_cer = round(float(row["cer_norm_mean"]), 4)
        expected_f1 = round(float(row["macro_f1_norm_mean"]), 4)
        expected_ned = round(float(row["ned_norm_mean"]), 4)
        expected_hmean = round(float(row["entity_hmean"]), 4)
        expected_d_cer_art = round(float(row["delta_cer_recovery_b1"]), 4)

        assert pol["metrics"]["cer"] == expected_cer
        assert pol["metrics"]["macro_f1"] == expected_f1
        assert pol["metrics"]["ned"] == expected_ned
        assert pol["metrics"]["entity_hmean"] == expected_hmean
        # Cross-check delta vs degraded against artifact pre-calculated field
        assert abs(pol["delta_cer_vs_degraded"] - expected_d_cer_art) <= 0.001


@pytest.mark.asyncio
async def test_relative_recovery_rate_and_guard(client: AsyncClient):
    """Verify strictly: (CER_deg - CER_prep) / (CER_deg - CER_ctrl) with guard for abs <= 0.005."""
    # 1. Rotation S2: Degraded CER is ~0.418, Control is 0.3203 -> Denom is ~0.0977 (> 0.005)
    res_rot = await client.get("/api/v1/research/preprocessing?degradation=rotation&severity=S2")
    assert res_rot.status_code == 200
    group_rot = res_rot.json()["groups"][0]
    c_ctrl = group_rot["clean_control_metrics"]["cer"]
    c_deg = group_rot["degraded_metrics"]["cer"]

    for pol in group_rot["policies"]:
        c_prep = pol["metrics"]["cer"]
        denom = c_deg - c_ctrl
        expected_rel_rec = round((c_deg - c_prep) / denom, 4)
        assert pol["relative_recovery_rate"] is not None
        assert pol["relative_recovery_rate"] == expected_rel_rec

    # Specifically for Standard Receipt Enhancement on Rotation S2:
    p_std = next(p for p in group_rot["policies"] if p["policy_key"] == "p_standard_receipt_enhancement")
    assert p_std["relative_recovery_rate"] > 0.90  # ~0.9918 (99.2% recovery)

    # For Aggressive Binarization on Rotation S2: relative recovery is negative (< 0 is valid and preserved)
    p_agg = next(p for p in group_rot["policies"] if p["policy_key"] == "p_aggressive_binarization")
    assert p_agg["relative_recovery_rate"] < 0.0  # -0.4749

    # 2. Guard test: Condition with tiny CER change (e.g. Gaussian Blur S1 where CER is 0.3205 vs 0.3203 -> denom 0.0002 <= 0.005)
    res_blur1 = await client.get("/api/v1/research/preprocessing?degradation=gaussian_blur&severity=S1")
    assert res_blur1.status_code == 200
    group_blur1 = res_blur1.json()["groups"][0]
    for pol in group_blur1["policies"]:
        assert pol["relative_recovery_rate"] is None  # Guard applied, null


@pytest.mark.asyncio
async def test_filtering_by_degradation_severity_and_policy(client: AsyncClient):
    """Test filtering by degradation aliases, severity aliases, and policy keys."""
    # 1. By code D6 vs name rotation
    res_d6 = await client.get("/api/v1/research/preprocessing?degradation=D6")
    res_rot = await client.get("/api/v1/research/preprocessing?degradation=rotation")
    assert res_d6.status_code == 200
    assert res_rot.status_code == 200
    assert res_d6.json()["total_groups"] == 4
    assert res_rot.json()["total_groups"] == 4
    assert res_d6.json()["groups"][0]["degradation_code"] == "D6"
    assert res_rot.json()["groups"][0]["degradation_code"] == "D6"

    # 2. By severity S3 vs 3
    res_s3 = await client.get("/api/v1/research/preprocessing?severity=S3")
    res_3 = await client.get("/api/v1/research/preprocessing?severity=3")
    assert res_s3.status_code == 200
    assert res_3.status_code == 200
    assert res_s3.json()["total_groups"] == 8
    assert res_3.json()["total_groups"] == 8
    assert all(g["severity"] == "S3" for g in res_s3.json()["groups"])

    # 3. By policy
    res_pol = await client.get("/api/v1/research/preprocessing?policy=p_standard_receipt_enhancement")
    assert res_pol.status_code == 200
    for g in res_pol.json()["groups"]:
        assert len(g["policies"]) == 1
        assert g["policies"][0]["policy_key"] == "p_standard_receipt_enhancement"

    # 4. Multi-filter combination: D6 + S2 + p_standard_receipt_enhancement
    res_comb = await client.get("/api/v1/research/preprocessing?degradation=D6&severity=S2&policy=p_standard_receipt_enhancement")
    assert res_comb.status_code == 200
    assert res_comb.json()["total_groups"] == 1
    assert len(res_comb.json()["groups"][0]["policies"]) == 1


@pytest.mark.asyncio
async def test_filtering_invalid_parameters_and_path_traversal(client: AsyncClient):
    """Test 404 responses on unknown degradation/severity/policy and path traversal attempts."""
    # 1. Unknown degradation
    res1 = await client.get("/api/v1/research/preprocessing?degradation=unknown_warp")
    assert res1.status_code == 404
    assert "not found" in res1.json()["detail"]

    # 2. Unknown severity
    res2 = await client.get("/api/v1/research/preprocessing?severity=S9")
    assert res2.status_code == 404

    # 3. Unknown policy
    res3 = await client.get("/api/v1/research/preprocessing?policy=nonexistent_filter")
    assert res3.status_code == 404

    # 4. Path traversal attempt
    res4 = await client.get("/api/v1/research/preprocessing?degradation=../../etc/shadow")
    assert res4.status_code == 404


@pytest.mark.asyncio
async def test_findings_are_strictly_computed_and_non_judgmental(client: AsyncClient):
    """Verify findings do not use subjective words (best, worst, optimal, recommended, stable)."""
    res = await client.get("/api/v1/research/preprocessing")
    assert res.status_code == 200
    groups = res.json()["groups"]

    forbidden_words = ["best", "worst", "optimal", "recommended", "stable"]

    for g in groups:
        for f in g["findings"]:
            f_lower = f.lower()
            for word in forbidden_words:
                assert f" {word} " not in f" {f_lower} ", f"Found forbidden subjective word '{word}' in finding: '{f}'"


@pytest.mark.asyncio
async def test_b2_research_artifacts_immutability():
    """Verify cryptographic SHA-256 hashes of B2 & B1 research files to guarantee read-only immutability."""
    root = Path(settings.RESEARCH_ROOT_PATH).resolve()

    expected_hashes = {
        "experiments/runs/b2_preprocessing_validation_n126_gpu/b2_conditions_detailed.csv": "3d1d535771207732421688fe0e1f51d950f05721832209bcbde4385f4ab307e2",
        "experiments/runs/b2_preprocessing_validation_n126_gpu/summary.json": "17ff010b9c9dbc5b7961111fa5e5933e28e50314c8f17c7cb1cfb47e77007663",
        "experiments/runs/b1_degraded_validation_n126_gpu/b1_conditions_detailed.csv": "81a6029e4542a199accd65f5fe8bfc0a2667821921a7d2c4f511742d31f11792",
        "experiments/runs/b1_degraded_validation_n126_gpu/summary.json": "cb28cbc3ec572ce085a2eb4e060784431fa7f2d291436c9ae7c7a1101e572c43",
    }

    for rel_path, expected_sha in expected_hashes.items():
        file_path = root / rel_path
        assert file_path.exists(), f"Artifact missing: {rel_path}"
        computed_sha = hashlib.sha256(file_path.read_bytes()).hexdigest()
        assert computed_sha == expected_sha, f"Artifact mutated! {rel_path}"
