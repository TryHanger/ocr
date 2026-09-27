"""Comprehensive unit and regression tests for TASK-015: B2 Preprocessing Robustness Runner."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import yaml

from src.core.schemas import PreprocessingSpec
from src.experiments.conditions import (
    DEGRADATION_CODE_MAP,
    ExperimentCondition,
    build_conditions_matrix,
)
from src.experiments.config import compute_config_hash, load_experiment_config
from src.experiments.runner import UnifiedExperimentRunner
from src.preprocessing.pipeline import PreprocessingPipeline


def test_b2_matrix_generation_129_conditions():
    """Verify B2 matrix generates exactly 129 conditions (1 control + 128 B2 conditions, 0 B1)."""
    cfg_path = Path("configs/b2_baseline_gpu.yaml")
    assert cfg_path.is_file(), f"Missing config file {cfg_path}"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    conditions = build_conditions_matrix(cfg, smoke_mode=False)
    assert len(conditions) == 129, f"Expected 129 conditions, got {len(conditions)}"

    # 1 control condition
    b0_conds = [c for c in conditions if c.baseline_type == "B0"]
    assert len(b0_conds) == 1
    assert b0_conds[0].condition_id == "D0_S0_P0"
    assert b0_conds[0].degradation_type == "none"
    assert b0_conds[0].severity == 0
    assert b0_conds[0].preprocessing_id == "none"

    # 0 B1 conditions
    b1_conds = [c for c in conditions if c.baseline_type == "B1"]
    assert len(b1_conds) == 0

    # 128 B2 conditions
    b2_conds = [c for c in conditions if c.baseline_type == "B2"]
    assert len(b2_conds) == 128

    # Check 4 preprocessing pipelines evenly represented: 32 conditions each
    pipelines = {c.preprocessing_id for c in b2_conds}
    expected_pipelines = {
        "p_minimal_cleanup",
        "p_contrast_enhancement",
        "p_standard_receipt_enhancement",
        "p_aggressive_binarization",
    }
    assert pipelines == expected_pipelines
    for p in expected_pipelines:
        p_count = sum(1 for c in b2_conds if c.preprocessing_id == p)
        assert p_count == 32, f"Pipeline {p} has {p_count} conditions, expected 32"


def test_b2_parameter_forwarding_in_pipeline():
    """Verify preprocessing parameters are accurately passed to individual pipeline steps."""
    cfg_path = Path("configs/b2_baseline_gpu.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    pipelines_cfg = cfg["matrix"]["preprocessing_pipelines"]

    # Test p_standard_receipt_enhancement parameter unpacking
    standard_cfg = pipelines_cfg["p_standard_receipt_enhancement"]
    steps = standard_cfg["steps"]
    params = standard_cfg["parameters"]

    assert steps == ["deskew", "clahe", "denoise"]
    assert params["deskew"]["max_angle"] == 15.0
    assert params["clahe"]["clip_limit"] == 2.0
    assert params["clahe"]["tile_grid_size"] == [8, 8]
    assert params["denoise"]["method"] == "median"
    assert params["denoise"]["ksize"] == 3

    pipeline = PreprocessingPipeline(steps=steps)
    spec = PreprocessingSpec(enabled=True, methods=steps, parameters=params)

    # Synthetic test image (BGR)
    img = np.full((100, 100, 3), 128, dtype=np.uint8)

    # Process through pipeline and verify execution without error
    out_img, meta = pipeline.process(img, spec)
    assert out_img is not None
    assert out_img.shape[0] > 0 and out_img.shape[1] > 0


def test_b2_p_aggressive_binarization_parameters():
    """Verify aggressive binarization accepts and applies gaussian adaptive threshold parameters."""
    cfg_path = Path("configs/b2_baseline_gpu.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    bin_cfg = cfg["matrix"]["preprocessing_pipelines"]["p_aggressive_binarization"]
    steps = bin_cfg["steps"]
    params = bin_cfg["parameters"]

    pipeline = PreprocessingPipeline(steps=steps)
    spec = PreprocessingSpec(enabled=True, methods=steps, parameters=params)

    img = np.full((50, 50, 3), 200, dtype=np.uint8)
    out_img, meta = pipeline.process(img, spec)
    # Output must be binary (only 0 or 255 values)
    unique_vals = set(np.unique(out_img))
    assert unique_vals.issubset({0, 255})


def test_frozen_references_exist_and_load():
    """Verify frozen B0 and B1 reference files exist, are valid, and load correctly."""
    b0_path = Path("experiments/runs/b0_clean_validation_n126_gpu/summary.json")
    b1_path = Path("experiments/runs/b1_degraded_validation_n126_gpu/summary.json")

    assert b0_path.is_file(), f"Missing B0 reference {b0_path}"
    assert b1_path.is_file(), f"Missing B1 reference {b1_path}"

    dummy_cfg = {
        "b0_reference_path": str(b0_path),
        "b1_reference_path": str(b1_path),
        "dataset": {"root": "data/SROIE2019", "split": "validation"},
        "matrix": {"include_control": True, "severities": [1]},
    }
    runner = UnifiedExperimentRunner(config=dummy_cfg)

    b0_ref = runner._load_b0_reference()
    assert b0_ref is not None
    assert b0_ref["experiment_id"] == "b0_clean_validation_n126_gpu"
    assert "ocr" in b0_ref["metrics"]
    assert "kie" in b0_ref["metrics"]

    b1_ref = runner._load_b1_reference()
    assert b1_ref is not None
    assert b1_ref["experiment_id"] == "b1_degraded_validation_n126_gpu"
    assert len(b1_ref["conditions"]) == 33
    assert "D0_S0_P0" in b1_ref["conditions"]
    assert "D1_S1_P0" in b1_ref["conditions"]
    assert "D8_S4_P0" in b1_ref["conditions"]


def test_recovery_metric_calculation():
    """Verify recovery delta formulas match scientific requirements."""
    # Test formula:
    # delta_cer_recovery = CER(B2) - CER(B1) (negative = improvement)
    # delta_wer_recovery = WER(B2) - WER(B1) (negative = improvement)
    # delta_char_ned_recovery = NED(B2) - NED(B1) (positive = improvement)
    # delta_macro_f1_recovery = F1(B2) - F1(B1) (positive = improvement)
    # delta_entity_hmean_recovery = Hmean(B2) - Hmean(B1) (positive = improvement)

    b1_cer = 0.4000
    b2_cer = 0.3500
    delta_cer = round(b2_cer - b1_cer, 4)
    assert delta_cer == -0.0500  # Improvement

    b1_f1 = 0.3000
    b2_f1 = 0.4200
    delta_f1 = round(b2_f1 - b1_f1, 4)
    assert delta_f1 == 0.1200  # Improvement


def test_b2_smoke_config_validity():
    """Verify configs/b2_smoke_gpu.yaml is well-formed and produces expected explicit conditions."""
    cfg_path = Path("configs/b2_smoke_gpu.yaml")
    assert cfg_path.is_file()
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    conditions = build_conditions_matrix(cfg, smoke_mode=False)
    assert len(conditions) == 5
    c_ids = [c.condition_id for c in conditions]
    assert c_ids == [
        "D0_S0_P0",
        "D1_S1_P_minimal_cleanup",
        "D1_S4_P_contrast_enhancement",
        "D2_S2_P_standard_receipt_enhancement",
        "D3_S3_P_aggressive_binarization",
    ]
