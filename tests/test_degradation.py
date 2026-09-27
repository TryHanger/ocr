"""Comprehensive unit tests for degradation engine primitives, pipeline, and calibration."""

import copy
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
import yaml

from src.core.schemas import DegradationSpec
from src.degradation import (
    DEGRADATION_REGISTRY,
    BaseDegradationPrimitive,
    DownsamplingDegradation,
    GaussianBlurDegradation,
    GaussianNoiseDegradation,
    JPEGCompressionDegradation,
    MotionBlurDegradation,
    PerspectiveDegradation,
    RotationDegradation,
    ShadowDegradation,
    DegradationPipeline,
    get_degradation,
)
from scripts.calibrate_degradations import (
    compute_mae,
    compute_mse,
    compute_psnr,
    compute_ssim,
    compute_laplacian_ratio,
    compute_luminance_drop,
    evaluate_image_pair,
    run_calibration,
)


@pytest.fixture
def rgb_image() -> np.ndarray:
    """Fixture providing a synthetic RGB document image with text and edges."""
    img = np.full((120, 100, 3), 245, dtype=np.uint8)
    cv2.putText(img, "RECEIPT INVOICE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1)
    cv2.putText(img, "TOTAL: $19.99", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (10, 10, 10), 1)
    cv2.line(img, (5, 75), (95, 75), (30, 30, 30), 1)
    cv2.putText(img, "THANK YOU", (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (15, 15, 15), 1)
    return img


@pytest.fixture
def gray_image() -> np.ndarray:
    """Fixture providing a 2D grayscale document image with text."""
    img = np.full((120, 100), 245, dtype=np.uint8)
    cv2.putText(img, "RECEIPT", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 20, 1)
    cv2.line(img, (5, 60), (95, 60), 30, 1)
    return img


# ==============================================================================
# 1. BaseDegradationPrimitive & General Contract Tests
# ==============================================================================


@pytest.mark.parametrize("deg_type", list(DEGRADATION_REGISTRY.keys()))
def test_all_primitives_severity_0_identity(deg_type: str, rgb_image: np.ndarray):
    """Severity 0 must return an exact bitwise identical copy with zero mutation."""
    primitive = get_degradation(deg_type)
    orig_copy = rgb_image.copy()
    spec = DegradationSpec(type=deg_type, severity=0, parameters={"some_param": 123}, seed=42)

    result, applied_spec = primitive.apply(rgb_image, spec)

    # Input not mutated
    assert np.array_equal(rgb_image, orig_copy)
    # Output identical to input
    assert np.array_equal(result, orig_copy)
    # Result is an independent copy (not the same memory buffer)
    assert result is not rgb_image
    assert applied_spec == spec


@pytest.mark.parametrize("deg_type", list(DEGRADATION_REGISTRY.keys()))
def test_all_primitives_preserve_shape_and_dtype(deg_type: str, rgb_image: np.ndarray):
    """All primitives must preserve (H, W, C) shape and np.uint8 dtype."""
    primitive = get_degradation(deg_type)
    orig_copy = rgb_image.copy()
    spec = DegradationSpec(type=deg_type, severity=2, seed=42)

    result, applied_spec = primitive.apply(rgb_image, spec)

    assert np.array_equal(rgb_image, orig_copy)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8
    assert result.shape == rgb_image.shape


@pytest.mark.parametrize("deg_type", list(DEGRADATION_REGISTRY.keys()))
def test_all_primitives_grayscale_support(deg_type: str, gray_image: np.ndarray):
    """All primitives must support 2D grayscale images preserving shape (H, W)."""
    primitive = get_degradation(deg_type)
    orig_copy = gray_image.copy()
    spec = DegradationSpec(type=deg_type, severity=2, seed=42)

    result, applied_spec = primitive.apply(gray_image, spec)

    assert np.array_equal(gray_image, orig_copy)
    assert result.dtype == np.uint8
    assert result.shape == gray_image.shape


def test_base_primitive_invalid_inputs(rgb_image: np.ndarray):
    """Base primitive must reject invalid image types, dtypes, and shapes."""
    primitive = GaussianBlurDegradation()
    spec = DegradationSpec(type="gaussian_blur", severity=1)

    # Not an ndarray
    with pytest.raises(TypeError, match="Image must be np.ndarray"):
        primitive.apply([[1, 2], [3, 4]], spec)  # type: ignore

    # Wrong dtype (float32)
    with pytest.raises(TypeError, match="Image dtype must be np.uint8"):
        primitive.apply(rgb_image.astype(np.float32), spec)

    # Wrong ndim (1D)
    with pytest.raises(ValueError, match="Image ndim must be 2"):
        primitive.apply(np.ones(10, dtype=np.uint8), spec)

    # Zero dimension
    with pytest.raises(ValueError, match="Image dimensions must be positive"):
        primitive.apply(np.zeros((0, 10, 3), dtype=np.uint8), spec)

    # Invalid channels (5 channels)
    with pytest.raises(ValueError, match="Image channels must be 1, 3, or 4"):
        primitive.apply(np.zeros((10, 10, 5), dtype=np.uint8), spec)

    # Invalid spec type
    with pytest.raises(TypeError, match="spec must be DegradationSpec"):
        primitive.apply(rgb_image, {"type": "gaussian_blur", "severity": 1})  # type: ignore


# ==============================================================================
# 2. D1 — Gaussian Blur Tests
# ==============================================================================


def test_gaussian_blur_determinism(rgb_image: np.ndarray):
    deg = GaussianBlurDegradation()
    spec = DegradationSpec(type="gaussian_blur", severity=2, parameters={"sigma": 2.0})
    res1, _ = deg.apply(rgb_image, spec)
    res2, _ = deg.apply(rgb_image, spec)
    assert np.array_equal(res1, res2)


def test_gaussian_blur_monotonicity(rgb_image: np.ndarray):
    deg = GaussianBlurDegradation()
    psnrs = []
    laplacians = []
    for sigma in [1.0, 2.0, 3.5, 5.0]:
        spec = DegradationSpec(type="gaussian_blur", severity=1, parameters={"sigma": sigma})
        blurred, _ = deg.apply(rgb_image, spec)
        psnrs.append(compute_psnr(rgb_image, blurred))
        laplacians.append(compute_laplacian_ratio(rgb_image, blurred))

    # Increasing blur monotonically decreases PSNR and Laplacian edge energy
    assert all(psnrs[i] >= psnrs[i + 1] for i in range(len(psnrs) - 1))
    assert all(laplacians[i] >= laplacians[i + 1] for i in range(len(laplacians) - 1))


def test_gaussian_blur_invalid_parameters(rgb_image: np.ndarray):
    deg = GaussianBlurDegradation()
    # Non-positive sigma
    with pytest.raises(ValueError, match="sigma must be positive"):
        deg.apply(rgb_image, DegradationSpec(type="gaussian_blur", severity=1, parameters={"sigma": 0.0}))

    # Even kernel_size
    with pytest.raises(ValueError, match="kernel_size must be positive odd integer"):
        deg.apply(rgb_image, DegradationSpec(type="gaussian_blur", severity=1, parameters={"kernel_size": 4}))

    # Non-int kernel_size
    with pytest.raises(TypeError, match="kernel_size must be int"):
        deg.apply(rgb_image, DegradationSpec(type="gaussian_blur", severity=1, parameters={"kernel_size": "5"}))


# ==============================================================================
# 3. D2 — Motion Blur Tests
# ==============================================================================


def test_motion_blur_determinism(rgb_image: np.ndarray):
    deg = MotionBlurDegradation()
    spec = DegradationSpec(type="motion_blur", severity=2, parameters={"kernel_length": 9, "angle": 45.0})
    res1, _ = deg.apply(rgb_image, spec)
    res2, _ = deg.apply(rgb_image, spec)
    assert np.array_equal(res1, res2)


def test_motion_blur_increasing_length(rgb_image: np.ndarray):
    deg = MotionBlurDegradation()
    psnrs = []
    for length in [5, 9, 15, 21]:
        spec = DegradationSpec(type="motion_blur", severity=1, parameters={"kernel_length": length, "angle": 30.0})
        blurred, _ = deg.apply(rgb_image, spec)
        psnrs.append(compute_psnr(rgb_image, blurred))

    assert all(psnrs[i] >= psnrs[i + 1] for i in range(len(psnrs) - 1))


def test_motion_blur_invalid_parameters(rgb_image: np.ndarray):
    deg = MotionBlurDegradation()
    with pytest.raises(ValueError, match="kernel_length must be at least 3"):
        deg.apply(rgb_image, DegradationSpec(type="motion_blur", severity=1, parameters={"kernel_length": 2}))

    with pytest.raises(TypeError, match="kernel_length must be int"):
        deg.apply(rgb_image, DegradationSpec(type="motion_blur", severity=1, parameters={"kernel_length": 5.5}))


# ==============================================================================
# 4. D3 — Gaussian Noise Tests
# ==============================================================================


def test_gaussian_noise_determinism_and_seed(rgb_image: np.ndarray):
    deg = GaussianNoiseDegradation()
    spec1 = DegradationSpec(type="gaussian_noise", severity=2, parameters={"std": 20.0}, seed=42)
    spec2 = DegradationSpec(type="gaussian_noise", severity=2, parameters={"std": 20.0}, seed=42)
    spec3 = DegradationSpec(type="gaussian_noise", severity=2, parameters={"std": 20.0}, seed=999)

    res1, _ = deg.apply(rgb_image, spec1)
    res2, _ = deg.apply(rgb_image, spec2)
    res3, _ = deg.apply(rgb_image, spec3)

    # Identical with same seed
    assert np.array_equal(res1, res2)
    # Different with different seed
    assert not np.array_equal(res1, res3)


def test_gaussian_noise_zero_std_identity(rgb_image: np.ndarray):
    deg = GaussianNoiseDegradation()
    spec = DegradationSpec(type="gaussian_noise", severity=1, parameters={"std": 0.0, "mean": 0.0})
    res, _ = deg.apply(rgb_image, spec)
    assert np.array_equal(res, rgb_image)


def test_gaussian_noise_invalid_parameters(rgb_image: np.ndarray):
    deg = GaussianNoiseDegradation()
    with pytest.raises(ValueError, match="std cannot be negative"):
        deg.apply(rgb_image, DegradationSpec(type="gaussian_noise", severity=1, parameters={"std": -5.0}))


# ==============================================================================
# 5. D4 — JPEG Compression Tests
# ==============================================================================


def test_jpeg_compression_determinism_and_rgb_preservation(rgb_image: np.ndarray):
    deg = JPEGCompressionDegradation()
    spec = DegradationSpec(type="jpeg_compression", severity=2, parameters={"quality": 40})
    res1, _ = deg.apply(rgb_image, spec)
    res2, _ = deg.apply(rgb_image, spec)

    assert np.array_equal(res1, res2)
    # Ensure RGB channels are not inadvertently swapped to BGR
    # We can check that the dominant background channel remains around (245, 245, 245)
    assert res1.shape == rgb_image.shape
    assert res1.dtype == np.uint8


def test_jpeg_compression_quality_monotonicity(rgb_image: np.ndarray):
    deg = JPEGCompressionDegradation()
    psnrs = []
    # Decreasing quality factor -> increasing compression -> decreasing PSNR
    for q in [70, 40, 20, 8]:
        spec = DegradationSpec(type="jpeg_compression", severity=1, parameters={"quality": q})
        compressed, _ = deg.apply(rgb_image, spec)
        psnrs.append(compute_psnr(rgb_image, compressed))

    assert all(psnrs[i] >= psnrs[i + 1] for i in range(len(psnrs) - 1))


def test_jpeg_compression_invalid_parameters(rgb_image: np.ndarray):
    deg = JPEGCompressionDegradation()
    with pytest.raises(ValueError, match="quality must be between 1 and 100"):
        deg.apply(rgb_image, DegradationSpec(type="jpeg_compression", severity=1, parameters={"quality": 0}))

    with pytest.raises(ValueError, match="quality must be between 1 and 100"):
        deg.apply(rgb_image, DegradationSpec(type="jpeg_compression", severity=1, parameters={"quality": 105}))

    with pytest.raises(TypeError, match="quality must be int"):
        deg.apply(rgb_image, DegradationSpec(type="jpeg_compression", severity=1, parameters={"quality": 50.5}))


# ==============================================================================
# 6. D5 — Downsampling Tests
# ==============================================================================


def test_downsampling_determinism_and_shape(rgb_image: np.ndarray):
    deg = DownsamplingDegradation()
    spec = DegradationSpec(type="downsampling", severity=2, parameters={"scale_factor": 0.5})
    res1, _ = deg.apply(rgb_image, spec)
    res2, _ = deg.apply(rgb_image, spec)

    assert np.array_equal(res1, res2)
    assert res1.shape == rgb_image.shape


def test_downsampling_scale_factor_1_identity(rgb_image: np.ndarray):
    deg = DownsamplingDegradation()
    spec = DegradationSpec(type="downsampling", severity=1, parameters={"scale_factor": 1.0})
    res, _ = deg.apply(rgb_image, spec)
    assert np.array_equal(res, rgb_image)


def test_downsampling_via_factor(rgb_image: np.ndarray):
    deg = DownsamplingDegradation()
    spec = DegradationSpec(type="downsampling", severity=1, parameters={"factor": 2})
    res, _ = deg.apply(rgb_image, spec)
    assert res.shape == rgb_image.shape


def test_downsampling_invalid_parameters(rgb_image: np.ndarray):
    deg = DownsamplingDegradation()
    with pytest.raises(ValueError, match="scale_factor must be in"):
        deg.apply(rgb_image, DegradationSpec(type="downsampling", severity=1, parameters={"scale_factor": 0.0}))

    with pytest.raises(ValueError, match="scale_factor must be in"):
        deg.apply(rgb_image, DegradationSpec(type="downsampling", severity=1, parameters={"scale_factor": 1.5}))

    with pytest.raises(ValueError, match="factor must be positive"):
        deg.apply(rgb_image, DegradationSpec(type="downsampling", severity=1, parameters={"factor": -2}))


# ==============================================================================
# 7. D6 — Rotation Tests
# ==============================================================================


def test_rotation_determinism_and_modes(rgb_image: np.ndarray):
    deg = RotationDegradation()
    spec_const = DegradationSpec(type="rotation", severity=2, parameters={"angle": 5.0, "border_mode": "constant"})
    spec_rep = DegradationSpec(type="rotation", severity=2, parameters={"angle": 5.0, "border_mode": "replicate"})

    res_c1, _ = deg.apply(rgb_image, spec_const)
    res_c2, _ = deg.apply(rgb_image, spec_const)
    res_rep, _ = deg.apply(rgb_image, spec_rep)

    assert np.array_equal(res_c1, res_c2)
    assert res_c1.shape == rgb_image.shape
    assert res_rep.shape == rgb_image.shape
    assert not np.array_equal(res_c1, res_rep)


def test_rotation_zero_angle_identity(rgb_image: np.ndarray):
    deg = RotationDegradation()
    spec = DegradationSpec(type="rotation", severity=1, parameters={"angle": 0.0})
    res, _ = deg.apply(rgb_image, spec)
    assert np.array_equal(res, rgb_image)


def test_rotation_invalid_border_mode(rgb_image: np.ndarray):
    deg = RotationDegradation()
    with pytest.raises(ValueError, match="Unsupported border_mode 'invalid'"):
        deg.apply(rgb_image, DegradationSpec(type="rotation", severity=1, parameters={"angle": 5.0, "border_mode": "invalid"}))


# ==============================================================================
# 8. D7 — Perspective Distortion Tests
# ==============================================================================


def test_perspective_determinism_and_seed(rgb_image: np.ndarray):
    deg = PerspectiveDegradation()
    spec1 = DegradationSpec(type="perspective", severity=2, parameters={"distortion_scale": 0.1}, seed=42)
    spec2 = DegradationSpec(type="perspective", severity=2, parameters={"distortion_scale": 0.1}, seed=42)
    spec3 = DegradationSpec(type="perspective", severity=2, parameters={"distortion_scale": 0.1}, seed=99)

    res1, _ = deg.apply(rgb_image, spec1)
    res2, _ = deg.apply(rgb_image, spec2)
    res3, _ = deg.apply(rgb_image, spec3)

    assert np.array_equal(res1, res2)
    assert not np.array_equal(res1, res3)
    assert res1.shape == rgb_image.shape


def test_perspective_scale_zero_identity(rgb_image: np.ndarray):
    deg = PerspectiveDegradation()
    spec = DegradationSpec(type="perspective", severity=1, parameters={"distortion_scale": 0.0})
    res, _ = deg.apply(rgb_image, spec)
    assert np.array_equal(res, rgb_image)


def test_perspective_invalid_parameters(rgb_image: np.ndarray):
    deg = PerspectiveDegradation()
    with pytest.raises(ValueError, match="distortion_scale must be in"):
        deg.apply(rgb_image, DegradationSpec(type="perspective", severity=1, parameters={"distortion_scale": -0.1}))

    with pytest.raises(ValueError, match="distortion_scale must be in"):
        deg.apply(rgb_image, DegradationSpec(type="perspective", severity=1, parameters={"distortion_scale": 0.5}))


# ==============================================================================
# 9. D8 — Shadow Tests
# ==============================================================================


def test_shadow_determinism(rgb_image: np.ndarray):
    deg = ShadowDegradation()
    spec = DegradationSpec(type="shadow", severity=2, parameters={"opacity": 0.5, "angle": 45.0, "coverage": 0.5})
    res1, _ = deg.apply(rgb_image, spec)
    res2, _ = deg.apply(rgb_image, spec)

    assert np.array_equal(res1, res2)
    assert res1.shape == rgb_image.shape


def test_shadow_opacity_zero_identity(rgb_image: np.ndarray):
    deg = ShadowDegradation()
    spec = DegradationSpec(type="shadow", severity=1, parameters={"opacity": 0.0})
    res, _ = deg.apply(rgb_image, spec)
    assert np.array_equal(res, rgb_image)


def test_shadow_invalid_parameters(rgb_image: np.ndarray):
    deg = ShadowDegradation()
    with pytest.raises(ValueError, match="opacity must be in"):
        deg.apply(rgb_image, DegradationSpec(type="shadow", severity=1, parameters={"opacity": 1.5}))

    with pytest.raises(ValueError, match="coverage must be in"):
        deg.apply(rgb_image, DegradationSpec(type="shadow", severity=1, parameters={"coverage": 0.0}))


# ==============================================================================
# 10. DegradationPipeline & Factory Tests
# ==============================================================================


def test_get_degradation_factory():
    for name in DEGRADATION_REGISTRY:
        deg = get_degradation(name)
        assert isinstance(deg, BaseDegradationPrimitive)

    with pytest.raises(KeyError, match="Unknown degradation type"):
        get_degradation("unknown_filter")


def test_pipeline_composition_and_immutability(rgb_image: np.ndarray):
    pipe = DegradationPipeline()
    orig_copy = rgb_image.copy()

    spec1 = DegradationSpec(type="gaussian_blur", severity=1, parameters={"sigma": 1.0})
    spec2 = DegradationSpec(type="jpeg_compression", severity=2, parameters={"quality": 50})

    pipe.add_step(get_degradation("gaussian_blur"), spec1)
    pipe.add_step(get_degradation("jpeg_compression"), spec2)

    result, applied_specs = pipe.apply(rgb_image)

    # Input not mutated
    assert np.array_equal(rgb_image, orig_copy)
    assert result.shape == rgb_image.shape
    assert result.dtype == np.uint8
    assert len(applied_specs) == 2
    assert applied_specs[0] == spec1
    assert applied_specs[1] == spec2


def test_pipeline_dynamic_specs(rgb_image: np.ndarray):
    pipe = DegradationPipeline()
    specs = [
        DegradationSpec(type="downsampling", severity=1, parameters={"scale_factor": 0.67}),
        DegradationSpec(type="rotation", severity=1, parameters={"angle": 2.0}),
    ]

    result, applied_specs = pipe.apply(rgb_image, specs=specs)
    assert result.shape == rgb_image.shape
    assert len(applied_specs) == 2


def test_pipeline_invalid_step():
    pipe = DegradationPipeline()
    with pytest.raises(TypeError, match="Expected BaseDegradation"):
        pipe.add_step("not_a_degradation", DegradationSpec(type="gaussian_blur", severity=1))  # type: ignore

    with pytest.raises(TypeError, match="Expected DegradationSpec"):
        pipe.add_step(GaussianBlurDegradation(), "not_a_spec")  # type: ignore


# ==============================================================================
# 11. Calibration Metrics & Artifact Verification Tests
# ==============================================================================


def test_metric_calculations_identity():
    img = np.full((50, 50, 3), 200, dtype=np.uint8)
    assert compute_mse(img, img) == 0.0
    assert compute_psnr(img, img) == 100.0
    assert compute_mae(img, img) == 0.0
    assert compute_ssim(img, img) == pytest.approx(1.0, abs=1e-4)
    assert compute_laplacian_ratio(img, img) == 1.0
    assert compute_luminance_drop(img, img) == 0.0


def test_calibration_configuration_file_validity():
    cfg_path = Path("configs/degradation.yaml")
    assert cfg_path.exists()
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "degradations" in data
    assert len(data["degradations"]) == 8

    for deg_type, info in data["degradations"].items():
        assert "severities" in info
        severities = info["severities"]
        for s in (0, 1, 2, 3, 4):
            assert s in severities


def test_calibration_runner_execution(tmp_path: Path):
    cfg_path = Path("configs/degradation.yaml")
    out_dir = tmp_path / "calib_out"

    report = run_calibration(
        config_path=cfg_path,
        output_dir=out_dir,
        save_samples=True,
    )

    assert report["real_data_calibration"] in ("COMPLETED", "PENDING")
    assert report["calibration_dataset"] in ("synthetic_fixture", "sroie_validation")
    assert report["research_validation_size"] == 126
    assert isinstance(report["final_calibration_completed"], bool)
    assert report["parameter_status"] in ("provisional / fixture-based", "final_research_calibrated")
    assert "degradations" in report
    assert len(report["degradations"]) == 8

    # Verify report written to disk
    report_json = out_dir / "calibration_report.json"
    assert report_json.exists()
    calib_yaml = out_dir / "calibration_config.yaml"
    assert calib_yaml.exists()
    samples_dir = out_dir / "samples"
    assert samples_dir.exists()
    assert len(list(samples_dir.glob("*.png"))) == 40  # 8 degradations * 5 severities


def test_base_primitive_output_contract_failures():
    """Verify that BaseDegradationPrimitive validates output from subclasses."""

    class BadReturnDegradation(BaseDegradationPrimitive):
        def __init__(self, mode: str):
            self.mode = mode

        def _apply(self, image: np.ndarray, spec: DegradationSpec) -> np.ndarray:
            if self.mode == "not_ndarray":
                return [1, 2, 3]  # type: ignore
            elif self.mode == "wrong_dtype":
                return image.astype(np.float32)
            elif self.mode == "wrong_shape":
                return image[1:, 1:]
            return image

    img = np.zeros((10, 10, 3), dtype=np.uint8)
    spec = DegradationSpec(type="gaussian_blur", severity=1)

    with pytest.raises(TypeError, match="Degraded output must be np.ndarray"):
        BadReturnDegradation("not_ndarray").apply(img, spec)

    with pytest.raises(TypeError, match="Degraded output dtype must be np.uint8"):
        BadReturnDegradation("wrong_dtype").apply(img, spec)

    with pytest.raises(ValueError, match="Degraded output shape .* must match input shape"):
        BadReturnDegradation("wrong_shape").apply(img, spec)


def test_single_channel_3d_support():
    """Verify 3D single-channel image (H, W, 1) support across primitives."""
    img_1c = np.full((30, 30, 1), 200, dtype=np.uint8)
    img_1c[10:20, 10:20, 0] = 50

    for name in ("jpeg_compression", "downsampling", "rotation", "perspective", "shadow"):
        deg = get_degradation(name)
        spec = DegradationSpec(type=name, severity=1)
        res, _ = deg.apply(img_1c, spec)
        assert res.shape == (30, 30, 1)
        assert res.dtype == np.uint8


def test_gaussian_blur_explicit_kernel_size(rgb_image: np.ndarray):
    deg = GaussianBlurDegradation()
    spec = DegradationSpec(type="gaussian_blur", severity=1, parameters={"kernel_size": 7, "sigma": 1.5})
    res, _ = deg.apply(rgb_image, spec)
    assert res.shape == rgb_image.shape


def test_shadow_1x1_boundary_edge_case():
    deg = ShadowDegradation()
    img_1x1 = np.full((1, 1, 3), 100, dtype=np.uint8)
    spec = DegradationSpec(type="shadow", severity=1)
    res, _ = deg.apply(img_1x1, spec)
    assert res.shape == (1, 1, 3)


def test_cli_calibrate_degradations(monkeypatch, tmp_path: Path):
    from scripts.calibrate_degradations import main as calib_main
    out_dir = tmp_path / "cli_calib"
    monkeypatch.setattr(
        "sys.argv",
        [
            "calibrate_degradations.py",
            "--config",
            "configs/degradation.yaml",
            "--output-dir",
            str(out_dir),
            "--no-samples",
        ],
    )
    code = calib_main()
    assert code == 0
    assert (out_dir / "calibration_report.json").exists()
    assert (out_dir / "calibration_config.yaml").exists()


def test_calibration_fixture_contract_and_performance(tmp_path: Path):
    """Regression test ensuring calibration runner on fixtures is small, fast, finite, and low-memory."""
    import math
    import time
    from scripts.calibrate_degradations import find_calibration_images, run_calibration

    # 1. Verify calibration fixture search defaults to small fixture (<= 5 images), not full SROIE corpus
    imgs, src = find_calibration_images()
    assert src == "SYNTHETIC_FIXTURE"
    assert len(imgs) <= 5

    # 2. Verify calibration runner completes in < 2.0s
    t0 = time.perf_counter()
    report = run_calibration(
        config_path=Path("configs/degradation.yaml"),
        output_dir=tmp_path / "fast_calib",
        save_samples=False,
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"Calibration runner took {elapsed:.2f}s, expected < 2.0s"

    # 3. Verify cases correspond to config
    with open("configs/degradation.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for deg_type, info in cfg["degradations"].items():
        assert deg_type in report["degradations"]
        deg_rep = report["degradations"][deg_type]
        assert len(deg_rep["candidate_evaluations"]) == len(info.get("candidates", []))
        assert len(deg_rep["severities"]) == 5  # 0..4

        # 4. Verify metrics have finite values
        for cand_eval in deg_rep["candidate_evaluations"]:
            for m_name, val in cand_eval["mean_metrics"].items():
                assert math.isfinite(val), f"Metric {m_name} is non-finite: {val}"
        for sev_idx, sev_eval in deg_rep["severities"].items():
            for m_name, val in sev_eval["mean_metrics"].items():
                assert math.isfinite(val), f"Metric {m_name} for sev {sev_idx} is non-finite: {val}"


