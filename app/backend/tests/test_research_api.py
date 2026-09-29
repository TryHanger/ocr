"""Tests for Research Mode API endpoints, registry, and read-only artifact isolation."""

import hashlib
from pathlib import Path
import pytest
from httpx import AsyncClient

from app.core.config import settings


@pytest.mark.asyncio
async def test_research_overview_endpoint(client: AsyncClient):
    """Test GET /api/v1/research/overview returns verified tracks and metrics."""
    res = await client.get("/api/v1/research/overview")
    assert res.status_code == 200
    data = res.json()

    assert "tracks" in data
    assert len(data["tracks"]) == 3
    track_ids = [t["id"] for t in data["tracks"]]
    assert "B0" in track_ids
    assert "B1" in track_ids
    assert "B2" in track_ids

    assert data["total_experiments"] >= 14
    assert "canonical_dataset" in data
    assert "execution_provider" in data

    # Verify B0 baseline key metrics
    b0_track = next(t for t in data["tracks"] if t["id"] == "B0")
    assert b0_track["experiment_count"] == 2
    assert "cer_normalized" in b0_track["key_metrics"]
    assert b0_track["key_metrics"]["cer_normalized"] > 0.0


@pytest.mark.asyncio
async def test_research_experiments_list_and_filtering(client: AsyncClient):
    """Test GET /api/v1/research/experiments returns all registered experiments and filters by track."""
    # 1. All experiments
    res = await client.get("/api/v1/research/experiments")
    assert res.status_code == 200
    data = res.json()
    assert "items" in data
    assert data["total"] >= 14

    for exp in data["items"]:
        assert "id" in exp
        assert "track" in exp
        assert "name" in exp
        assert "status" in exp
        assert "source" in exp
        assert exp["status"] in ("COMPLETED", "UNAVAILABLE")

    # 2. Filter by Track B1
    res_b1 = await client.get("/api/v1/research/experiments?track=B1")
    assert res_b1.status_code == 200
    b1_data = res_b1.json()
    assert b1_data["total"] == 8
    assert all(e["track"] == "B1" for e in b1_data["items"])

    # 3. Filter by Track B2
    res_b2 = await client.get("/api/v1/research/experiments?track=B2")
    assert res_b2.status_code == 200
    b2_data = res_b2.json()
    assert b2_data["total"] == 4
    assert all(e["track"] == "B2" for e in b2_data["items"])

    # 4. Invalid track filter -> 400
    res_invalid = await client.get("/api/v1/research/experiments?track=UNKNOWN")
    assert res_invalid.status_code == 400


@pytest.mark.asyncio
async def test_research_experiment_details_and_findings(client: AsyncClient):
    """Test GET /api/v1/research/experiments/{id} for B0, B1, and B2."""
    # 1. Test B0 Clean Validation
    res_b0 = await client.get("/api/v1/research/experiments/b0_clean_validation")
    assert res_b0.status_code == 200
    b0 = res_b0.json()
    assert b0["id"] == "b0_clean_validation"
    assert b0["track"] == "B0"
    assert b0["sample_size"] == 126
    assert "source" in b0
    assert len(b0["conditions"]) >= 1
    assert len(b0["findings"]) >= 1
    assert b0["ocr_metrics"] is not None
    assert b0["kie_metrics"] is not None
    assert b0["ocr_metrics"]["cer_normalized_mean"] > 0.0

    # 2. Test B0 CPU vs GPU Comparison
    res_gpu = await client.get("/api/v1/research/experiments/b0_cpu_vs_gpu")
    assert res_gpu.status_code == 200
    gpu = res_gpu.json()
    assert gpu["id"] == "b0_cpu_vs_gpu"
    assert "speedup" in gpu["metrics_summary"]
    assert any("speedup" in f.lower() for f in gpu["findings"])

    # 3. Test B1 Rotation Degradation
    res_rot = await client.get("/api/v1/research/experiments/b1_rotation")
    assert res_rot.status_code == 200
    rot = res_rot.json()
    assert rot["id"] == "b1_rotation"
    assert rot["track"] == "B1"
    assert len(rot["conditions"]) == 4  # S1 to S4
    assert any(c["severity"] == 1 for c in rot["conditions"])
    assert any(c["severity"] == 4 for c in rot["conditions"])
    assert len(rot["findings"]) >= 1

    # 4. Test B2 Standard Receipt Enhancement Preprocessing
    res_b2 = await client.get("/api/v1/research/experiments/b2_standard_receipt_enhancement")
    assert res_b2.status_code == 200
    b2 = res_b2.json()
    assert b2["id"] == "b2_standard_receipt_enhancement"
    assert b2["track"] == "B2"
    assert len(b2["conditions"]) > 0
    assert len(b2["findings"]) >= 1


@pytest.mark.asyncio
async def test_research_security_and_unknown_ids(client: AsyncClient):
    """Test 404 behavior for unknown experiment IDs and path traversal protection."""
    # 1. Nonexistent experiment ID
    res_nonexistent = await client.get("/api/v1/research/experiments/nonexistent_experiment_xyz")
    assert res_nonexistent.status_code == 404

    # 2. Path traversal attempts
    res_traversal1 = await client.get("/api/v1/research/experiments/..%2F..%2Fetc%2Fpasswd")
    assert res_traversal1.status_code == 404

    res_traversal2 = await client.get("/api/v1/research/experiments/..%2F..%2Fexperiments%2Fruns%2Focr_baseline_report.json")
    assert res_traversal2.status_code == 404


@pytest.mark.asyncio
async def test_research_artifacts_read_only_isolation(client: AsyncClient):
    """Verify that calling research endpoints leaves research artifacts bitwise identical."""
    repo_root = Path(settings.RESEARCH_ROOT_PATH).resolve()
    target_artifact = repo_root / "experiments" / "runs" / "b0_clean_validation_n126_gpu" / "summary.json"
    if not target_artifact.exists():
        target_artifact = repo_root / "experiments" / "runs" / "b0_clean_validation_n126" / "summary.json"

    assert target_artifact.exists(), f"Target research artifact {target_artifact} must exist."

    # Compute SHA256 before API requests
    with open(target_artifact, "rb") as f:
        hash_before = hashlib.sha256(f.read()).hexdigest()

    # Invoke all research endpoints multiple times
    await client.get("/api/v1/research/overview")
    await client.get("/api/v1/research/experiments")
    await client.get("/api/v1/research/experiments/b0_clean_validation")
    await client.get("/api/v1/research/experiments/b1_gaussian_blur")
    await client.get("/api/v1/research/experiments/b2_standard_receipt_enhancement")

    # Compute SHA256 after API requests
    with open(target_artifact, "rb") as f:
        hash_after = hashlib.sha256(f.read()).hexdigest()

    assert hash_before == hash_after, "Research artifact was mutated! Must be strictly read-only."
