"""Unit tests for PreprocessingPolicy decisions and image operations."""

import io
import numpy as np
from PIL import Image
import pytest

from app.schemas.quality import QualityResultDTO
from app.services.preprocessing_policy import PreprocessingPolicy


def _make_dummy_quality(
    blur: float = 250.0,
    contrast: float = 65.0,
    noise: float = 2.0,
    angle: float = 0.0,
) -> QualityResultDTO:
    return QualityResultDTO(
        document_id="test_doc",
        blur_score=blur,
        contrast_score=contrast,
        noise_score=noise,
        rotation_angle=angle,
        resolution_dpi=300.0,
        quality_score=0.95,
        profile_summary="clean",
        metrics={},
        recommendations={},
    )


def test_plan_pipeline_clean_document_pass_through():
    policy = PreprocessingPolicy()
    clean_quality = _make_dummy_quality(angle=0.2, contrast=60.0, noise=1.5)
    plan = policy.plan_operations(clean_quality)
    assert plan == []  # No-op pass-through for clean docs


def test_plan_pipeline_triggers_deskew():
    policy = PreprocessingPolicy()
    rotated_quality = _make_dummy_quality(angle=4.5)
    plan = policy.plan_operations(rotated_quality)
    assert any(step["operation"] == "deskew" for step in plan)
    deskew_step = [s for s in plan if s["operation"] == "deskew"][0]
    assert deskew_step["parameters"]["angle"] == 4.5


def test_plan_pipeline_triggers_clahe_for_low_contrast():
    policy = PreprocessingPolicy()
    low_contrast_quality = _make_dummy_quality(contrast=24.0)
    plan = policy.plan_operations(low_contrast_quality)
    assert any(step["operation"] == "clahe" for step in plan)


def test_plan_pipeline_triggers_denoise_for_high_noise():
    policy = PreprocessingPolicy()
    noisy_quality = _make_dummy_quality(noise=15.0)
    plan = policy.plan_operations(noisy_quality)
    assert any(step["operation"] == "denoise_median" for step in plan)


def test_apply_pipeline_transformations():
    policy = PreprocessingPolicy()

    # Create dummy 100x100 RGB image
    arr = np.ones((100, 100, 3), dtype=np.uint8) * 128
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    quality = _make_dummy_quality(angle=3.0, contrast=20.0, noise=12.0)
    processed_bytes, steps_meta = policy.execute_plan(raw_bytes, quality)

    assert len(processed_bytes) > 0
    assert len(steps_meta) == 3
    assert [s["operation"] for s in steps_meta] == ["deskew", "clahe", "denoise_median"]
    for s in steps_meta:
        assert s["duration_ms"] >= 0.0
        assert s["status"] == "APPLIED"
