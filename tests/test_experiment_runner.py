"""Comprehensive unit and regression tests for Unified Experiment Runner (TASK-010)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import pytest
import numpy as np

from src.datasets.sroie import SROIEAdapter
from src.experiments.bootstrap import aggregate_condition_records, compute_bootstrap_ci
from src.experiments.conditions import (
    ExperimentCondition,
    build_condition_id,
    build_conditions_matrix,
)
from src.experiments.config import canonicalize_for_hashing, compute_config_hash, load_experiment_config
from src.experiments.runner import UnifiedExperimentRunner
from src.experiments.seed import derive_seed
from src.experiments.split import resolve_dataset_root, resolve_split_documents


# --- 1. Determinism and Seed Derivation ---

def test_seed_derivation_deterministic():
    """Verify that same inputs produce identical derived seed."""
    s1 = derive_seed(42, "doc_001", "gaussian_blur", 1, "none")
    s2 = derive_seed(42, "doc_001", "gaussian_blur", 1, "none")
    assert s1 == s2
    assert isinstance(s1, int)
    assert 0 <= s1 < 2**32


def test_seed_derivation_separation():
    """Verify that differing documents, severities, or degradations yield distinct seeds."""
    base = derive_seed(42, "doc_001", "gaussian_blur", 1, "none")
    diff_doc = derive_seed(42, "doc_002", "gaussian_blur", 1, "none")
    diff_sev = derive_seed(42, "doc_001", "gaussian_blur", 4, "none")
    diff_deg = derive_seed(42, "doc_001", "motion_blur", 1, "none")
    diff_prep = derive_seed(42, "doc_001", "gaussian_blur", 1, "p_clahe")

    seeds = {base, diff_doc, diff_sev, diff_deg, diff_prep}
    # All 5 seeds must be distinct
    assert len(seeds) == 5


# --- 2. Condition Semantics & Matrix Expansion ---

def test_condition_severity_semantics():
    """Verify that severity 0 is strictly reserved for D0_S0_P0 and D1..D8 severity 0 is forbidden."""
    # Control condition is valid
    c_control = ExperimentCondition("D0_S0_P0", "none", 0, "none", "B0")
    assert c_control.severity == 0
    assert c_control.baseline_type == "B0"

    # Attempting to assign severity 0 to degradation D1 must raise ValueError
    with pytest.raises(ValueError, match="Severity 0 is strictly reserved for control condition"):
        ExperimentCondition("D1_S0_P0", "gaussian_blur", 0, "none", "B1")

    # Attempting to assign severity > 0 to degradation none must raise ValueError
    with pytest.raises(ValueError, match="Degradation type cannot be 'none' when severity > 0"):
        ExperimentCondition("D0_S1_P0", "none", 1, "none", "B1")


def test_build_conditions_matrix_smoke_mode():
    """Verify default smoke matrix produces exactly the 5 required conditions."""
    cfg = {"matrix": {"include_control": True}}
    smoke_conds = build_conditions_matrix(cfg, smoke_mode=True)
    c_ids = [c.condition_id for c in smoke_conds]
    assert c_ids == [
        "D0_S0_P0",
        "D1_S1_P0",
        "D1_S4_P0",
        "D1_S1_P_clahe",
        "D1_S4_P_clahe",
    ]
    # Check baseline types
    assert smoke_conds[0].baseline_type == "B0"
    assert smoke_conds[1].baseline_type == "B1"
    assert smoke_conds[2].baseline_type == "B1"
    assert smoke_conds[3].baseline_type == "B2"
    assert smoke_conds[4].baseline_type == "B2"


def test_build_conditions_matrix_full_never_generates_severity_0_for_degradations():
    """Verify that full matrix expansion never contains severity 0 for D1..D8."""
    cfg = {
        "matrix": {
            "include_control": True,
            "degradations": ["gaussian_blur", "motion_blur"],
            "severities": [0, 1, 4],  # Attempt to sneak severity 0 into severities list
            "preprocessing_pipelines": {"p_clahe": ["clahe"]},
        }
    }
    conds = build_conditions_matrix(cfg, smoke_mode=False)
    for c in conds:
        if c.degradation_type != "none":
            assert c.severity in (1, 4)
            assert c.severity != 0
        else:
            assert c.condition_id == "D0_S0_P0"


# --- 3. Split Safety and Test Split Guard ---

def test_split_guard_test_rejected_without_override():
    """Verify that requesting 'test' split without allow_test raises ValueError."""
    adapter = SROIEAdapter("tests/fixtures/sroie_valid")
    with pytest.raises(ValueError, match="strictly frozen for final evaluation"):
        resolve_split_documents(adapter, split="test", allow_test=False, allow_fixture=True)


def test_split_guard_test_accepted_with_override():
    """Verify that requesting 'test' split with allow_test succeeds."""
    adapter = SROIEAdapter("tests/fixtures/sroie_valid")
    doc_ids, split_name, _ = resolve_split_documents(adapter, split="test", allow_test=True, allow_fixture=True)
    assert split_name == "test"
    assert isinstance(doc_ids, list)


def test_split_guard_real_data_required():
    """Verify that running without real data and without allow_fixture fails with REAL_DATA_REQUIRED."""
    # When point to non-real fixture path without allow_fixture
    adapter = SROIEAdapter("tests/fixtures/sroie_valid")
    with pytest.raises(FileNotFoundError, match="REAL_DATA_REQUIRED"):
        resolve_split_documents(adapter, split="development", allow_fixture=False)


# --- 4. Configuration Hashing ---

def test_config_hash_stability():
    """Verify identical configs produce identical hashes, ignoring volatile keys."""
    cfg1 = {
        "experiment_seed": 42,
        "dataset": {"split": "development"},
        "timestamp": "2026-09-26T12:00:00",
        "output_dir": "experiments/runs/run1",
    }
    cfg2 = {
        "output_dir": "experiments/runs/run2",
        "timestamp": "2026-09-26T12:05:00",
        "dataset": {"split": "development"},
        "experiment_seed": 42,
    }
    assert compute_config_hash(cfg1) == compute_config_hash(cfg2)

    # Changing a semantic parameter changes hash
    cfg3 = copy.deepcopy(cfg1)
    cfg3["experiment_seed"] = 99
    assert compute_config_hash(cfg1) != compute_config_hash(cfg3)


# --- 5. Bootstrap CI Statistics ---

def test_compute_bootstrap_ci():
    """Verify deterministic 1000-iteration document-level bootstrap confidence intervals."""
    values = [0.10, 0.12, 0.11, 0.15, 0.09, 0.13, 0.10, 0.14]
    ci1 = compute_bootstrap_ci(values, iterations=1000, ci=0.95, seed=42)
    ci2 = compute_bootstrap_ci(values, iterations=1000, ci=0.95, seed=42)
    assert ci1 == ci2
    assert ci1[0] <= np.mean(values) <= ci1[1]

    # Edge cases
    assert compute_bootstrap_ci([]) == (0.0, 0.0)
    assert compute_bootstrap_ci([0.25]) == (0.25, 0.25)


# --- 6. GT Isolation & Execution Order ---

def test_runner_execution_order_gt_isolation(tmp_path: Path):
    """Verify GT is never accessed before OCR and KIE inference."""
    call_order: list[str] = []

    class SpyingAdapter(SROIEAdapter):
        def get_image(self, document_id: str):
            call_order.append("get_image")
            return super().get_image(document_id)

        def get_ocr_ground_truth(self, document_id: str):
            call_order.append("get_ocr_gt")
            return super().get_ocr_ground_truth(document_id)

        def get_kie_ground_truth(self, document_id: str):
            call_order.append("get_kie_gt")
            return super().get_kie_ground_truth(document_id)

    cfg = {
        "experiment_seed": 42,
        "dataset": {"candidate_roots": ["tests/fixtures/sroie_valid"], "split": "development"},
        "ocr": {"engine_name": "mock"},
        "kie": {"engine_name": "mock"},
        "evaluation": {"bootstrap": {"iterations": 50}},
    }

    runner = UnifiedExperimentRunner(
        config=cfg,
        output_dir=tmp_path,
        smoke_mode=True,
        allow_fixture=True,
        experiment_id="gt_isolation_test",
    )

    # Replace adapter with spying adapter
    orig_setup = runner._setup_engines
    spy_adapter = SpyingAdapter("tests/fixtures/sroie_valid")
    runner._setup_engines = lambda: (  # type: ignore[assignment]
        orig_setup()[0],
        orig_setup()[1],
        orig_setup()[2],
        spy_adapter,
    )

    summary = runner.run()
    assert summary["experiment_id"] == "gt_isolation_test"
    assert len(call_order) > 0
    # get_image must always occur before ground truth retrieval
    image_idx = call_order.index("get_image")
    ocr_gt_idx = call_order.index("get_ocr_gt")
    kie_gt_idx = call_order.index("get_kie_gt")
    assert image_idx < ocr_gt_idx
    assert image_idx < kie_gt_idx


# --- 7. Failure Accounting ---

def test_runner_failure_accounting_no_silent_drops(tmp_path: Path):
    """Verify that OCR and KIE failures are recorded with status and error info without dropping documents."""
    cfg = {
        "experiment_seed": 42,
        "dataset": {"candidate_roots": ["tests/fixtures/sroie_valid"], "split": "development"},
        "ocr": {"engine_name": "mock"},
        "kie": {"engine_name": "mock"},
        "evaluation": {"bootstrap": {"iterations": 50}},
    }

    runner = UnifiedExperimentRunner(
        config=cfg,
        output_dir=tmp_path,
        smoke_mode=True,
        allow_fixture=True,
        experiment_id="failure_accounting_test",
    )

    # Intercept OCR engine to raise on purpose
    orig_setup = runner._setup_engines
    def failing_setup():
        ocr_eng, kie_eng, evalr, adapter = orig_setup()
        def failing_recognize(image, doc_id):
            raise RuntimeError("Simulated OCR inference failure")
        ocr_eng.recognize = failing_recognize
        return ocr_eng, kie_eng, evalr, adapter

    runner._setup_engines = failing_setup  # type: ignore[assignment]

    summary = runner.run()
    doc_counts = summary["document_counts"]

    # All 5 smoke conditions × 1 fixture doc = 5 runs, all failed
    assert doc_counts["total_requested"] == 5
    assert doc_counts["total_successful"] == 0
    assert doc_counts["total_failed"] == 5
    assert doc_counts["failure_rate"] == 1.0

    # Verify per_document.jsonl records
    jsonl_path = tmp_path / "failure_accounting_test" / "per_document.jsonl"
    assert jsonl_path.exists()
    records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(records) == 5
    for r in records:
        assert r["status"] == "ocr_failed"
        assert r["error_type"] == "RuntimeError"
        assert "Simulated OCR inference failure" in r["error_message"]


# --- 8. Artifact Schema Verification ---

def test_runner_artifact_schema(tmp_path: Path):
    """Verify generated artifact bundle: config.yaml, manifest.json, per_document.jsonl, summary.json, logs/."""
    cfg = {
        "experiment_seed": 42,
        "dataset": {"candidate_roots": ["tests/fixtures/sroie_valid"], "split": "development"},
        "ocr": {"engine_name": "mock"},
        "kie": {"engine_name": "mock"},
        "evaluation": {"bootstrap": {"iterations": 50}},
    }

    runner = UnifiedExperimentRunner(
        config=cfg,
        output_dir=tmp_path,
        smoke_mode=True,
        allow_fixture=True,
        save_images=True,
        experiment_id="schema_test",
    )
    summary = runner.run()

    run_dir = tmp_path / "schema_test"
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "per_document.jsonl").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "logs" / "run.log").exists()
    assert (run_dir / "images").is_dir()

    # Manifest schema checks
    with open(run_dir / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["experiment_id"] == "schema_test"
    assert "config_hash" in manifest
    assert "git_commit" in manifest
    assert "runtime_metadata" in manifest
    assert "start_time" in manifest["runtime_metadata"]
    assert "end_time" in manifest["runtime_metadata"]
    assert "duration_seconds" in manifest["runtime_metadata"]
    assert manifest["num_conditions"] == 5

    # Summary schema checks (TASK-010-R2)
    with open(run_dir / "summary.json", "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    assert "conditions" in summary_data
    assert len(summary_data["conditions"]) == 5

    for cid, c_data in summary_data["conditions"].items():
        assert "document_counts" in c_data
        assert "ocr" in c_data
        assert "kie" in c_data

        # OCR metrics & Bootstrap CIs
        ocr = c_data["ocr"]
        for m in ("cer_raw", "wer_raw", "char_ned_raw", "cer_normalized", "wer_normalized", "char_ned_normalized"):
            assert m in ocr
            assert "mean" in ocr[m]
            assert "median" in ocr[m]
            assert "std" in ocr[m]
            assert "ci_95" in ocr[m]
            assert len(ocr[m]["ci_95"]) == 2

        # KIE analytical metrics & Bootstrap CIs
        kie = c_data["kie"]
        assert "per_field" in kie
        assert "raw_doc_em_rate" in kie
        assert "raw_doc_em_ci_95" in kie
        assert "normalized_doc_em_rate" in kie
        assert "normalized_doc_em_ci_95" in kie
        assert "macro_f1_normalized" in kie
        assert "macro_f1_ci_95" in kie

        # SROIE official-compatible entity metrics per condition
        assert "sroie_official_compatible" in kie
        sroie_off = kie["sroie_official_compatible"]
        assert "entity_precision" in sroie_off
        assert "entity_recall" in sroie_off
        assert "entity_hmean" in sroie_off
        assert "total_gt_entities" in sroie_off
        assert "total_pred_entities" in sroie_off
        assert "total_matched_entities" in sroie_off
        assert 0.0 <= sroie_off["entity_precision"] <= 1.0
        assert 0.0 <= sroie_off["entity_recall"] <= 1.0
        assert 0.0 <= sroie_off["entity_hmean"] <= 1.0


def test_aggregate_condition_records_sroie_official_compatible():
    """Verify aggregate_condition_records explicitly computes SROIE official-compatible entity metrics."""
    records = [
        {
            "status": "success",
            "ocr_metrics": {
                "cer_raw": 0.1, "wer_raw": 0.1, "char_ned_raw": 0.9,
                "cer_normalized": 0.05, "wer_normalized": 0.05, "char_ned_normalized": 0.95,
            },
            "kie_metrics": {
                "field_matches_raw": {"company": True, "date": True, "address": False, "total": False},
                "field_matches_normalized": {"company": True, "date": True, "address": True, "total": False},
                "raw_doc_em": False,
                "normalized_doc_em": False,
                "normalized_ground_truth": {"company": "A", "date": "B", "address": "C", "total": "D"},
                "normalized_predictions": {"company": "A", "date": "B", "address": "C", "total": "X"},
            },
        },
        {
            "status": "success",
            "ocr_metrics": {
                "cer_raw": 0.0, "wer_raw": 0.0, "char_ned_raw": 1.0,
                "cer_normalized": 0.0, "wer_normalized": 0.0, "char_ned_normalized": 1.0,
            },
            "kie_metrics": {
                "field_matches_raw": {"company": True, "date": True, "address": True, "total": True},
                "field_matches_normalized": {"company": True, "date": True, "address": True, "total": True},
                "raw_doc_em": True,
                "normalized_doc_em": True,
                "normalized_ground_truth": {"company": "A", "date": "B", "address": "C", "total": "D"},
                "normalized_predictions": {"company": "A", "date": "B", "address": "C", "total": "D"},
            },
        },
    ]

    res = aggregate_condition_records(records, bootstrap_seed=42)
    sroie_off = res["kie"]["sroie_official_compatible"]
    # Total GT entities = 4 + 4 = 8
    # Total Pred entities = 4 + 4 = 8
    # Total Matched entities = 3 (doc 1) + 4 (doc 2) = 7
    assert sroie_off["total_gt_entities"] == 8
    assert sroie_off["total_pred_entities"] == 8
    assert sroie_off["total_matched_entities"] == 7
    assert sroie_off["entity_precision"] == round(7 / 8, 4)
    assert sroie_off["entity_recall"] == round(7 / 8, 4)
    assert sroie_off["entity_hmean"] == round(7 / 8, 4)


def test_runner_debug_timing_stages_and_init(tmp_path: Path):
    """Verify runner with debug_timing=True logs INIT metrics and START/DONE for all pipeline stages."""
    cfg = {
        "experiment_seed": 42,
        "dataset": {"candidate_roots": ["tests/fixtures/sroie_valid"], "split": "development"},
        "ocr": {"engine_name": "mock"},
        "kie": {"engine_name": "mock"},
        "evaluation": {"bootstrap": {"iterations": 50}},
    }
    runner = UnifiedExperimentRunner(
        config=cfg,
        output_dir=tmp_path,
        smoke_mode=True,
        allow_fixture=True,
        debug_timing=True,
        experiment_id="timing_test",
    )
    summary = runner.run()
    assert summary is not None

    log_path = tmp_path / "timing_test" / "logs" / "run.log"
    assert log_path.exists()
    log_content = log_path.read_text(encoding="utf-8")

    # Verify [INIT] metrics logged
    assert "[INIT]" in log_content
    assert "dataset_load_seconds=" in log_content
    assert "engine_initialization_seconds=" in log_content
    assert "number_of_documents=" in log_content
    assert "number_of_conditions=" in log_content

    # Verify [START] and [DONE] logged for all 8 stages
    for stage in (
        "load_image",
        "degradation",
        "preprocessing",
        "ocr",
        "kie",
        "gt_load",
        "evaluation",
        "write_result",
    ):
        assert f"[START] doc=doc_valid_1 condition=D0_S0_P0 stage={stage}" in log_content
        assert f"[DONE]  doc=doc_valid_1 condition=D0_S0_P0 stage={stage} elapsed=" in log_content


def test_runner_single_engine_initialization(tmp_path: Path, monkeypatch):
    """Verify engines are initialized exactly once per runner invocation, not per document or condition."""
    cfg = {
        "experiment_seed": 42,
        "dataset": {"candidate_roots": ["tests/fixtures/sroie_valid"], "split": "development"},
        "ocr": {"engine_name": "mock"},
        "kie": {"engine_name": "mock"},
        "evaluation": {"bootstrap": {"iterations": 50}},
    }
    runner = UnifiedExperimentRunner(
        config=cfg,
        output_dir=tmp_path,
        smoke_mode=True,
        allow_fixture=True,
        experiment_id="single_init_test",
    )

    init_call_count = 0
    orig_setup = runner._setup_engines

    def counted_setup():
        nonlocal init_call_count
        init_call_count += 1
        return orig_setup()

    monkeypatch.setattr(runner, "_setup_engines", counted_setup)
    runner.run()

    # Must be called exactly once
    assert init_call_count == 1


def test_runner_incremental_flush(tmp_path: Path, monkeypatch):
    """Verify per_document.jsonl is incrementally flushed to disk after every evaluation."""
    cfg = {
        "experiment_seed": 42,
        "dataset": {"candidate_roots": ["tests/fixtures/sroie_valid"], "split": "development"},
        "ocr": {"engine_name": "mock"},
        "kie": {"engine_name": "mock"},
        "evaluation": {"bootstrap": {"iterations": 50}},
    }
    runner = UnifiedExperimentRunner(
        config=cfg,
        output_dir=tmp_path,
        smoke_mode=True,
        allow_fixture=True,
        experiment_id="flush_test",
    )

    per_doc_path = tmp_path / "flush_test" / "per_document.jsonl"
    flush_calls = 0
    orig_open = open

    class FlushingProxy:
        def __init__(self, f):
            self._f = f

        def write(self, s):
            return self._f.write(s)

        def flush(self):
            nonlocal flush_calls
            flush_calls += 1
            return self._f.flush()

        def __enter__(self):
            self._f.__enter__()
            return self

        def __exit__(self, *args):
            return self._f.__exit__(*args)

    def monitored_open(path, *args, **kwargs):
        f = orig_open(path, *args, **kwargs)
        if str(path).endswith("per_document.jsonl"):
            return FlushingProxy(f)
        return f

    monkeypatch.setattr("builtins.open", monitored_open)
    runner.run()

    # With 1 document in fixture and 5 conditions = 5 evaluations
    # Must flush after each of the 5 records
    assert flush_calls == 5
    with orig_open(per_doc_path, "r", encoding="utf-8") as f:
        final_lines = [l for l in f if l.strip()]
        assert len(final_lines) == 5

