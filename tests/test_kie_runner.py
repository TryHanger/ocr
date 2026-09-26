"""Unit and integration tests for KIE baseline runner and execution trace audit."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from scripts.run_kie_baseline import run_kie_baseline


def test_run_kie_baseline_on_fixture(tmp_path: Path):
    kie_config = Path("configs/kie.yaml")
    ocr_config = Path("configs/ocr.yaml")
    data_root = Path("tests/fixtures/sroie_valid")

    report = run_kie_baseline(
        kie_config_path=kie_config,
        ocr_config_path=ocr_config,
        output_dir=tmp_path,
        data_root=data_root,
        split="train",
        kie_engine_name="mock",
        ocr_engine_name="mock",
    )

    # 1. Status must remain PENDING_REAL_DATA on fixtures
    assert report["status"] == "PENDING_REAL_DATA"
    assert report["dataset_type"] == "synthetic_fixture"

    # 2. Execution order trace audit
    audit = report["execution_order_audit"]
    assert audit["enforced_phases"] == ["ocr_inference", "kie_inference", "gt_loading", "evaluation"]
    assert audit["gt_leakage_prevented"] is True
    assert "extract(ocr_result: OCRResult, document_id: str) -> KIEResult" in audit["kie_api_contract"]

    # 3. Both metric sections must be present
    assert "analytical_metrics" in report
    assert "sroie_official_compatible" in report

    # 4. Report file generated and valid JSON
    report_file = tmp_path / "kie_baseline_report.json"
    assert report_file.exists()
    with open(report_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["title"] == report["title"]
    assert len(loaded["documents"]) == 1
    assert loaded["documents"][0]["phases_completed"] == ["ocr_inference", "kie_inference", "gt_loading", "evaluation"]


def test_run_kie_baseline_with_rule_based_engine(tmp_path: Path):
    kie_config = Path("configs/kie.yaml")
    ocr_config = Path("configs/ocr.yaml")
    data_root = Path("tests/fixtures/sroie_valid")

    report = run_kie_baseline(
        kie_config_path=kie_config,
        ocr_config_path=ocr_config,
        output_dir=tmp_path,
        data_root=data_root,
        split="train",
        kie_engine_name="rule_based",
        ocr_engine_name="mock",
    )

    assert report["status"] == "PENDING_REAL_DATA"
    assert report["engine_configurations"]["kie_engine"]["name"] == "rule_based"
    assert report["engine_configurations"]["kie_engine"]["fallback_largest_amount"] is False
