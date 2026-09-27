"""Tests for RapidOCR execution provider configuration and fail-fast validation."""

from __future__ import annotations

import pytest

from src.ocr.factory import get_ocr_engine
from src.ocr.rapid_ocr import RapidOCREngine


def test_rapid_ocr_default_cpu_provider():
    """Verify that default initialization selects CPU and passes stack verification."""
    eng = RapidOCREngine()
    assert eng.execution_provider == "cpu"
    manifest = eng.get_model_manifest()
    assert manifest["environment"]["execution_provider"] == "CPUExecutionProvider"
    assert manifest["resolved"]["runtime"]["actual_providers"][0] == "CPUExecutionProvider"

    valid, errors = eng.verify_model_stack()
    assert valid is True
    assert errors == []


def test_rapid_ocr_cuda_provider_initialization():
    """Verify that execution_provider='cuda' selects CUDAExecutionProvider as primary."""
    eng = RapidOCREngine(execution_provider="cuda")
    assert eng.execution_provider == "cuda"
    manifest = eng.get_model_manifest()
    assert manifest["environment"]["execution_provider"] == "CUDAExecutionProvider"
    assert manifest["resolved"]["detector"]["providers"][0] == "CUDAExecutionProvider"
    assert manifest["resolved"]["recognizer"]["providers"][0] == "CUDAExecutionProvider"

    valid, errors = eng.verify_model_stack()
    assert valid is True
    assert errors == []


def test_rapid_ocr_invalid_provider_raises():
    """Verify that invalid provider string raises ValueError immediately."""
    with pytest.raises(ValueError, match="Unsupported execution_provider"):
        RapidOCREngine(execution_provider="invalid_provider")


def test_factory_supports_execution_provider():
    """Verify that get_ocr_engine forwards execution_provider properly."""
    eng = get_ocr_engine("rapidocr", config={"execution_provider": "cpu"})
    assert getattr(eng, "execution_provider", None) == "cpu"
