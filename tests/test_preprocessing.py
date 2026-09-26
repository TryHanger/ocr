"""Comprehensive unit and integration tests for preprocessing engine and B0/B1/B2 baselines."""

import inspect
import json
from pathlib import Path
from typing import Any, List
import cv2
import numpy as np
import pytest

from src.core.contracts import BasePreprocessor
from src.core.schemas import DegradationSpec, PreprocessingSpec
from src.ocr.factory import MockOCREngine, get_ocr_engine
from src.preprocessing import (
    BasePreprocessingPrimitive,
    BinarizationPreprocessor,
    CLAHEPreprocessor,
    DenoisePreprocessor,
    DeskewPreprocessor,
    GrayscalePreprocessor,
    PREPROCESSOR_REGISTRY,
    PreprocessingPipeline,
    get_preprocessor,
)
from src.preprocessing.baseline import (
    BaselineComparisonResult,
    run_b0_b1_b2_comparison,
)
from scripts.run_preprocessing_baseline import main as baseline_main, run_preprocessing_baseline


@pytest.fixture
def synthetic_rgb_image() -> np.ndarray:
    """Fixture returning a standard 100x100 synthetic RGB uint8 image with text."""
    img = np.full((120, 160, 3), 255, dtype=np.uint8)
    cv2.putText(img, "TEST RECEIPT", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "TOTAL 50.00", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img


@pytest.fixture
def synthetic_gray_image() -> np.ndarray:
    """Fixture returning a standard 120x160 synthetic 2D grayscale uint8 image."""
    img = np.full((120, 160), 255, dtype=np.uint8)
    cv2.putText(img, "TEST RECEIPT", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
    return img


# ==============================================================================
# 1. Base Contract & Input Validation Tests
# ==============================================================================


def test_preprocessors_satisfy_base_contract():
    """Verify that all registered preprocessors implement BasePreprocessor."""
    for name, cls in PREPROCESSOR_REGISTRY.items():
        instance = cls()
        assert isinstance(instance, BasePreprocessor)
        assert isinstance(instance, BasePreprocessingPrimitive)


def test_base_input_validation(synthetic_rgb_image):
    """Verify comprehensive validation of image input types and shapes."""
    prep = GrayscalePreprocessor()
    spec = PreprocessingSpec(enabled=True)

    # 1. Non-numpy
    with pytest.raises(TypeError, match="np.ndarray"):
        prep.process("not_an_array", spec)  # type: ignore

    # 2. Non-uint8
    float_img = synthetic_rgb_image.astype(np.float32) / 255.0
    with pytest.raises(TypeError, match="np.uint8"):
        prep.process(float_img, spec)

    # 3. Invalid dimensionality
    with pytest.raises(ValueError, match="Image ndim"):
        prep.process(np.zeros((10, 10, 3, 2), dtype=np.uint8), spec)
    with pytest.raises(ValueError, match="Image ndim"):
        prep.process(np.zeros((10,), dtype=np.uint8), spec)

    # 4. Zero dimensions
    with pytest.raises(ValueError, match="positive"):
        prep.process(np.zeros((0, 10, 3), dtype=np.uint8), spec)

    # 5. Invalid channels
    with pytest.raises(ValueError, match="Image channels"):
        prep.process(np.zeros((10, 10, 5), dtype=np.uint8), spec)

    # 6. Invalid spec type
    with pytest.raises(TypeError, match="PreprocessingSpec"):
        prep.process(synthetic_rgb_image, "not_a_spec")  # type: ignore


def test_input_immutability(synthetic_rgb_image):
    """Verify that input image array is never modified in-place."""
    orig_copy = synthetic_rgb_image.copy()

    for name, cls in PREPROCESSOR_REGISTRY.items():
        instance = cls()
        spec = PreprocessingSpec(enabled=True)
        res, _ = instance.process(synthetic_rgb_image, spec)

        # Original image must be bitwise identical to before processing
        assert np.array_equal(synthetic_rgb_image, orig_copy), f"{name} mutated input in-place"
        # Returned array must be independent object
        assert res is not synthetic_rgb_image


def test_disabled_spec_returns_copy(synthetic_rgb_image):
    """Verify that spec.enabled=False passes through an exact independent copy."""
    spec = PreprocessingSpec(enabled=False)
    for name, cls in PREPROCESSOR_REGISTRY.items():
        instance = cls()
        res, returned_spec = instance.process(synthetic_rgb_image, spec)
        assert np.array_equal(res, synthetic_rgb_image)
        assert res is not synthetic_rgb_image
        assert returned_spec is spec


def test_determinism(synthetic_rgb_image):
    """Verify that identical input and spec produce bitwise identical output."""
    for name, cls in PREPROCESSOR_REGISTRY.items():
        instance = cls()
        spec = PreprocessingSpec(enabled=True)
        res1, _ = instance.process(synthetic_rgb_image, spec)
        res2, _ = instance.process(synthetic_rgb_image, spec)
        assert np.array_equal(res1, res2), f"{name} output is not deterministic"


# ==============================================================================
# 2. GT Leakage Audit
# ==============================================================================


def test_gt_leakage_audit_preprocessing():
    """Verify that preprocessing process API accepts strictly (image, spec)."""
    base_sig = inspect.signature(BasePreprocessor.process)
    base_params = [p for p in base_sig.parameters.keys() if p != "self"]
    assert base_params == ["image", "spec"]

    prim_sig = inspect.signature(BasePreprocessingPrimitive.process)
    prim_params = [p for p in prim_sig.parameters.keys() if p != "self"]
    assert prim_params == ["image", "spec"]

    pipe_sig = inspect.signature(PreprocessingPipeline.process)
    pipe_params = [p for p in pipe_sig.parameters.keys() if p != "self"]
    assert pipe_params == ["image", "spec"]

    # Verify no registered preprocessor accepts GT parameters
    for name, cls in PREPROCESSOR_REGISTRY.items():
        sig = inspect.signature(cls.process)
        for param in sig.parameters.keys():
            assert "gt" not in param.lower(), f"{name} accepts gt"
            assert "box" not in param.lower(), f"{name} accepts box"
            assert "truth" not in param.lower(), f"{name} accepts truth"
            assert "label" not in param.lower(), f"{name} accepts label"


# ==============================================================================
# 3. Primitive P1 — Grayscale Tests
# ==============================================================================


def test_grayscale_from_rgb(synthetic_rgb_image):
    prep = GrayscalePreprocessor()
    res, _ = prep.process(synthetic_rgb_image, PreprocessingSpec(enabled=True))
    assert res.ndim == 2
    assert res.shape == (synthetic_rgb_image.shape[0], synthetic_rgb_image.shape[1])
    assert res.dtype == np.uint8


def test_grayscale_from_rgba():
    prep = GrayscalePreprocessor()
    rgba = np.full((50, 50, 4), 200, dtype=np.uint8)
    res, _ = prep.process(rgba, PreprocessingSpec(enabled=True))
    assert res.ndim == 2
    assert res.shape == (50, 50)
    assert res.dtype == np.uint8


def test_grayscale_from_grayscale_passthrough(synthetic_gray_image):
    prep = GrayscalePreprocessor()
    res, _ = prep.process(synthetic_gray_image, PreprocessingSpec(enabled=True))
    assert res.ndim == 2
    assert res.shape == synthetic_gray_image.shape
    assert np.array_equal(res, synthetic_gray_image)


def test_grayscale_from_single_channel_3d():
    prep = GrayscalePreprocessor()
    img_3d_1 = np.full((30, 40, 1), 128, dtype=np.uint8)
    res, _ = prep.process(img_3d_1, PreprocessingSpec(enabled=True))
    assert res.ndim == 2
    assert res.shape == (30, 40)
    assert np.array_equal(res, img_3d_1[:, :, 0])


# ==============================================================================
# 4. Primitive P2 — Denoising Tests
# ==============================================================================


def test_denoise_median(synthetic_rgb_image):
    prep = DenoisePreprocessor()
    spec = PreprocessingSpec(
        enabled=True,
        parameters={"method": "median", "ksize": 3},
    )
    res, _ = prep.process(synthetic_rgb_image, spec)
    assert res.shape == synthetic_rgb_image.shape
    assert res.dtype == np.uint8


def test_denoise_bilateral(synthetic_rgb_image):
    prep = DenoisePreprocessor()
    spec = PreprocessingSpec(
        enabled=True,
        parameters={"method": "bilateral", "d": 5, "sigma_color": 50.0, "sigma_space": 50.0},
    )
    res, _ = prep.process(synthetic_rgb_image, spec)
    assert res.shape == synthetic_rgb_image.shape
    assert res.dtype == np.uint8


def test_denoise_fast_nl_means_rgb(synthetic_rgb_image):
    prep = DenoisePreprocessor()
    spec = PreprocessingSpec(
        enabled=True,
        parameters={"method": "fast_nl_means", "h": 5.0, "template_window_size": 7, "search_window_size": 21},
    )
    res, _ = prep.process(synthetic_rgb_image, spec)
    assert res.shape == synthetic_rgb_image.shape
    assert res.dtype == np.uint8


def test_denoise_fast_nl_means_grayscale(synthetic_gray_image):
    prep = DenoisePreprocessor()
    spec = PreprocessingSpec(
        enabled=True,
        parameters={"method": "fast_nl_means", "h": 5.0},
    )
    res, _ = prep.process(synthetic_gray_image, spec)
    assert res.shape == synthetic_gray_image.shape
    assert res.dtype == np.uint8


def test_denoise_invalid_parameters(synthetic_rgb_image):
    prep = DenoisePreprocessor()

    # Unknown method
    with pytest.raises(ValueError, match="Unsupported denoise method"):
        prep.process(synthetic_rgb_image, PreprocessingSpec(parameters={"method": "unknown"}))

    # Even ksize for median
    with pytest.raises(ValueError, match="ksize"):
        prep.process(synthetic_rgb_image, PreprocessingSpec(parameters={"method": "median", "ksize": 4}))

    # Non-positive bilateral parameters
    with pytest.raises(ValueError, match="d for bilateral"):
        prep.process(synthetic_rgb_image, PreprocessingSpec(parameters={"method": "bilateral", "d": 0}))
    with pytest.raises(ValueError, match="sigma_color"):
        prep.process(synthetic_rgb_image, PreprocessingSpec(parameters={"method": "bilateral", "sigma_color": -1.0}))

    # Non-positive fast_nl_means parameters
    with pytest.raises(ValueError, match="h for fast_nl_means"):
        prep.process(synthetic_rgb_image, PreprocessingSpec(parameters={"method": "fast_nl_means", "h": -2.0}))


# ==============================================================================
# 5. Primitive P3 — CLAHE Tests
# ==============================================================================


def test_clahe_grayscale(synthetic_gray_image):
    prep = CLAHEPreprocessor()
    spec = PreprocessingSpec(enabled=True, parameters={"clip_limit": 2.0, "tile_grid_size": (8, 8)})
    res, _ = prep.process(synthetic_gray_image, spec)
    assert res.shape == synthetic_gray_image.shape
    assert res.dtype == np.uint8


def test_clahe_rgb(synthetic_rgb_image):
    prep = CLAHEPreprocessor()
    spec = PreprocessingSpec(enabled=True, parameters={"clip_limit": 3.0, "tile_grid_size": [4, 4]})
    res, _ = prep.process(synthetic_rgb_image, spec)
    assert res.shape == synthetic_rgb_image.shape
    assert res.dtype == np.uint8


def test_clahe_invalid_parameters(synthetic_rgb_image):
    prep = CLAHEPreprocessor()

    # clip_limit <= 0
    with pytest.raises(ValueError, match="clip_limit"):
        prep.process(synthetic_rgb_image, PreprocessingSpec(parameters={"clip_limit": 0.0}))

    # invalid tile_grid_size
    with pytest.raises(TypeError, match="tile_grid_size"):
        prep.process(synthetic_rgb_image, PreprocessingSpec(parameters={"tile_grid_size": "invalid"}))
    with pytest.raises(ValueError, match="positive"):
        prep.process(synthetic_rgb_image, PreprocessingSpec(parameters={"tile_grid_size": (0, 8)}))


# ==============================================================================
# 6. Primitive P4 — Adaptive Binarization Tests
# ==============================================================================


def test_binarization_gaussian(synthetic_rgb_image):
    prep = BinarizationPreprocessor()
    spec = PreprocessingSpec(
        enabled=True,
        parameters={"method": "gaussian", "block_size": 11, "c": 2.0},
    )
    res, _ = prep.process(synthetic_rgb_image, spec)
    assert res.ndim == 2
    assert res.shape == (synthetic_rgb_image.shape[0], synthetic_rgb_image.shape[1])
    assert res.dtype == np.uint8
    unique_vals = set(np.unique(res))
    assert unique_vals.issubset({0, 255})


def test_binarization_mean(synthetic_gray_image):
    prep = BinarizationPreprocessor()
    spec = PreprocessingSpec(
        enabled=True,
        parameters={"method": "mean", "block_size": 15, "c": 3.0},
    )
    res, _ = prep.process(synthetic_gray_image, spec)
    assert res.ndim == 2
    assert res.shape == synthetic_gray_image.shape
    assert set(np.unique(res)).issubset({0, 255})


def test_binarization_otsu(synthetic_rgb_image):
    prep = BinarizationPreprocessor()
    spec = PreprocessingSpec(enabled=True, parameters={"method": "otsu"})
    res, _ = prep.process(synthetic_rgb_image, spec)
    assert res.ndim == 2
    assert set(np.unique(res)).issubset({0, 255})


def test_binarization_invalid_parameters(synthetic_rgb_image):
    prep = BinarizationPreprocessor()

    with pytest.raises(ValueError, match="Unsupported binarization method"):
        prep.process(synthetic_rgb_image, PreprocessingSpec(parameters={"method": "invalid"}))

    # Even block size
    with pytest.raises(ValueError, match="block_size"):
        prep.process(synthetic_rgb_image, PreprocessingSpec(parameters={"method": "gaussian", "block_size": 10}))


# ==============================================================================
# 7. Primitive P5 — Deskew Tests
# ==============================================================================


def test_deskew_on_straight_image(synthetic_rgb_image):
    prep = DeskewPreprocessor()
    res, _ = prep.process(synthetic_rgb_image, PreprocessingSpec(enabled=True))
    assert res.shape == synthetic_rgb_image.shape
    assert res.dtype == np.uint8


def test_deskew_angle_estimation_and_correction():
    """Verify deskew detects and corrects skew on artificially tilted text."""
    img = np.full((300, 300, 3), 255, dtype=np.uint8)
    cv2.putText(img, "STORE NAME RECEIPT NUMBER", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(img, "TOTAL AMOUNT 123.45 DUE", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(img, "THANK YOU FOR SHOPPING TODAY", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Rotate by 6.0 degrees
    center = (150, 150)
    M = cv2.getRotationMatrix2D(center, 6.0, 1.0)
    rotated = cv2.warpAffine(img, M, (300, 300), borderValue=(255, 255, 255))

    prep = DeskewPreprocessor()
    estimated_angle = prep.estimate_skew_angle(rotated, max_angle=15.0)
    # The estimated angle to correct should be close to -6.0 deg
    assert abs(abs(estimated_angle) - 6.0) < 1.0

    res, _ = prep.process(rotated, PreprocessingSpec(enabled=True, parameters={"max_angle": 15.0}))
    assert res.shape == rotated.shape
    assert res.dtype == np.uint8


def test_deskew_exceeding_max_angle_is_ignored():
    """Verify deskew does not perform extreme rotation when max_angle is exceeded."""
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    prep = DeskewPreprocessor()
    # If estimated angle exceeds max_angle, it should return 0.0 correction
    angle = prep.estimate_skew_angle(img, max_angle=2.0)
    assert angle == 0.0


# ==============================================================================
# 8. PreprocessingPipeline & Factory Tests
# ==============================================================================


def test_pipeline_sequential_composition(synthetic_rgb_image):
    """Verify pipeline applies steps in explicit order and produces valid output."""
    pipeline = PreprocessingPipeline(steps=["grayscale", "clahe", "denoise"])
    spec = PreprocessingSpec(
        enabled=True,
        parameters={
            "clahe": {"clip_limit": 2.0},
            "denoise": {"method": "median", "ksize": 3},
        },
    )
    res, returned_spec = pipeline.process(synthetic_rgb_image, spec)
    assert res.ndim == 2  # Converted to grayscale
    assert res.dtype == np.uint8
    assert res.shape == (synthetic_rgb_image.shape[0], synthetic_rgb_image.shape[1])


def test_pipeline_rejection_of_unknown_method(synthetic_rgb_image):
    pipeline = PreprocessingPipeline()
    spec = PreprocessingSpec(enabled=True, methods=["grayscale", "unsupported_step"])
    with pytest.raises(KeyError, match="Unknown preprocessing step"):
        pipeline.process(synthetic_rgb_image, spec)


def test_factory_get_preprocessor():
    for name in ["grayscale", "denoise", "clahe", "binarization", "deskew"]:
        p = get_preprocessor(name)
        assert isinstance(p, BasePreprocessingPrimitive)

    with pytest.raises(KeyError, match="Unknown preprocessor"):
        get_preprocessor("non_existent")


# ==============================================================================
# 9. B0 / B1 / B2 Baseline Interface & Runner Tests
# ==============================================================================


def test_b0_b1_b2_comparison_runner(synthetic_rgb_image):
    """Verify run_b0_b1_b2_comparison executes all three branches cleanly."""
    mock_ocr = MockOCREngine()
    deg_spec = DegradationSpec(type="gaussian_blur", severity=1)
    prep_spec = PreprocessingSpec(enabled=True, methods=["grayscale", "denoise"])

    res = run_b0_b1_b2_comparison(
        image=synthetic_rgb_image,
        document_id="doc_mock_1",
        gt_text="MOCK GROUND TRUTH TEXT",
        degradation_spec=deg_spec,
        preprocessing_spec=prep_spec,
        ocr_engine=mock_ocr,
    )

    assert isinstance(res, BaselineComparisonResult)
    assert res.document_id == "doc_mock_1"
    assert res.b0.branch == "B0_original"
    assert res.b1.branch == "B1_degraded"
    assert res.b2.branch == "B2_degraded_preprocessed"

    # All branches must have evaluated metrics
    for branch_res in [res.b0, res.b1, res.b2]:
        assert "cer_raw" in branch_res.metrics
        assert "char_ned_normalized" in branch_res.metrics
        assert branch_res.processing_time_ms is not None

    d = res.to_dict()
    assert "b0" in d
    assert "b1" in d
    assert "b2" in d


def test_sroie_preprocessing_baseline_runner_fixture(tmp_path: Path):
    """Verify run_preprocessing_baseline runs against local fixture."""
    out_dir = tmp_path / "prep_baseline_out"
    report = run_preprocessing_baseline(
        config_ocr_path=Path("configs/ocr.yaml"),
        config_prep_path=Path("configs/preprocessing.yaml"),
        output_dir=out_dir,
        data_root=Path("tests/fixtures/sroie_valid"),
        split="train",
        degradation_type="gaussian_blur",
        severity=1,
        preprocessing_preset="minimal_cleanup",
    )

    assert report["real_data_preprocessing_baseline"] == "PENDING"
    assert report["dataset_type"] == "synthetic_fixture"
    assert "summary_metrics" in report
    assert "mean_b0_cer_raw" in report["summary_metrics"]
    assert "mean_b1_cer_raw" in report["summary_metrics"]
    assert "mean_b2_cer_raw" in report["summary_metrics"]

    report_file = out_dir / "preprocessing_baseline_report.json"
    assert report_file.exists()
    data = json.loads(report_file.read_text(encoding="utf-8"))
    assert data["real_data_preprocessing_baseline"] == "PENDING"


def test_cli_run_preprocessing_baseline(monkeypatch, tmp_path: Path):
    """Verify CLI entrypoint for preprocessing baseline runner."""
    out_dir = tmp_path / "cli_prep_baseline"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_preprocessing_baseline.py",
            "--config-ocr",
            "configs/ocr.yaml",
            "--config-prep",
            "configs/preprocessing.yaml",
            "--output-dir",
            str(out_dir),
            "--data-root",
            "tests/fixtures/sroie_valid",
            "--split",
            "train",
            "--degradation",
            "gaussian_blur",
            "--severity",
            "1",
            "--preset",
            "minimal_cleanup",
        ],
    )
    code = baseline_main()
    assert code == 0
    assert (out_dir / "preprocessing_baseline_report.json").exists()
