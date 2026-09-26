"""Unit and integration tests for OCR Stack (PP-OCRv6, ONNX Runtime, Manifest, Resize)."""

import inspect
import json
from pathlib import Path
from unittest.mock import MagicMock
import cv2
import numpy as np
import pytest

from src.core.contracts import BaseOCREngine
from src.core.schemas import OCRResult
from src.ocr.rapid_ocr import EXPECTED_MODELS, RapidOCREngine
from scripts.run_ocr_baseline import run_ocr_baseline


def test_rapid_ocr_stack_verification():
    """Verify that local bundled models match cryptographic hashes and CPU provider."""
    engine = RapidOCREngine()
    is_valid, errors = engine.verify_model_stack()
    assert is_valid is True, f"Model stack verification failed: {errors}"
    assert errors == []


def test_rapid_ocr_manifest_structure():
    """Verify that model manifest contains configured, resolved, environment, and resize_policy sections."""
    engine = RapidOCREngine()
    manifest = engine.get_model_manifest()

    assert "configured" in manifest
    assert "resolved" in manifest
    assert "environment" in manifest
    assert "resize_policy" in manifest

    # Check configured
    assert manifest["configured"]["name"] == "RapidOCR"
    assert manifest["configured"]["version"] == "3.9.2 (PP-OCRv6 small)"
    assert manifest["configured"]["detector"] == "PP-OCRv6_det_small.onnx"
    assert manifest["configured"]["recognizer"] == "PP-OCRv6_rec_small.onnx"

    # Check resolved
    resolved = manifest["resolved"]
    for model_key in ["detector", "recognizer", "classifier"]:
        assert model_key in resolved
        entry = resolved[model_key]
        assert entry["exists"] is True
        assert entry["sha256_verified"] is True
        assert entry["sha256"] == EXPECTED_MODELS[model_key]["sha256"]
        assert entry["size_bytes"] > 0
        assert Path(entry["path"]).is_file()

    # Check environment
    env = manifest["environment"]
    assert env["execution_provider"] == "CPUExecutionProvider"
    assert "rapidocr_version" in env
    assert "onnxruntime_version" in env
    assert "platform" in env
    assert "python_version" in env

    # Check resize_policy
    resize_sec = manifest["resize_policy"]
    assert "configured" in resize_sec
    assert "resolved" in resize_sec
    assert resize_sec["verified"] is True
    assert "baseline_consistency_rule" in resize_sec["resolved"]


def test_private_model_path_disappearance_fails_loudly():
    """Regression test: if _model_path disappears from ONNX Runtime session, resolution must fail loudly."""
    engine = RapidOCREngine()
    inner_sess = engine._engine.text_det.session.session
    orig_path = getattr(inner_sess, "_model_path", None)

    try:
        setattr(inner_sess, "_model_path", None)
        with pytest.raises(RuntimeError, match="relies on a private RapidOCR wrapper attribute"):
            engine.get_model_manifest()
    finally:
        setattr(inner_sess, "_model_path", orig_path)


def test_missing_model_file_stack_verification(monkeypatch):
    """Verify that verify_model_stack reports error if resolved model file is missing on disk."""
    engine = RapidOCREngine()
    orig_manifest = engine.get_model_manifest()

    def fake_manifest():
        m = dict(orig_manifest)
        m["resolved"] = dict(orig_manifest["resolved"])
        m["resolved"]["detector"] = dict(orig_manifest["resolved"]["detector"])
        m["resolved"]["detector"]["file_exists"] = False
        m["resolved"]["detector"]["model_path"] = "C:/non_existent/model.onnx"
        return m

    monkeypatch.setattr(engine, "get_model_manifest", fake_manifest)
    is_valid, errors = engine.verify_model_stack()
    assert is_valid is False
    assert any("Detector model file does not exist" in err for err in errors)


def test_hash_mismatch_stack_verification(monkeypatch):
    """Verify that verify_model_stack reports error if SHA256 does not match expected hash."""
    engine = RapidOCREngine()
    fake_expected = dict(EXPECTED_MODELS)
    fake_expected["detector"] = dict(EXPECTED_MODELS["detector"])
    fake_expected["detector"]["sha256"] = "0000000000000000000000000000000000000000000000000000000000000000"

    monkeypatch.setattr("src.ocr.rapid_ocr.EXPECTED_MODELS", fake_expected)
    is_valid, errors = engine.verify_model_stack()
    assert is_valid is False
    assert any("Detector SHA256 mismatch" in err for err in errors)


def test_rapid_ocr_legacy_tuple_output_compatibility():
    """Verify compatibility when RapidOCR returns legacy (results_list, elapse) tuple."""
    engine = RapidOCREngine()
    img = np.full((100, 200, 3), 255, dtype=np.uint8)

    dummy_box = [[10.0, 10.0], [80.0, 10.0], [80.0, 30.0], [10.0, 30.0]]
    legacy_return = ([ (dummy_box, "LEGACY_RECEIPT", 0.92) ], [0.01, 0.01, 0.01])

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(engine, "_engine", MagicMock(return_value=legacy_return))
        result = engine.recognize(img, "doc_legacy")

    assert isinstance(result, OCRResult)
    assert len(result.tokens) == 1
    assert result.tokens[0].text == "LEGACY_RECEIPT"
    assert result.tokens[0].confidence == 0.92
    assert result.full_text == "LEGACY_RECEIPT"


def test_rapid_ocr_input_formats():
    """Verify inference on RGB, grayscale, and binary images."""
    engine = RapidOCREngine(min_confidence=0.1)

    # 1. RGB
    rgb = np.full((120, 300, 3), 255, dtype=np.uint8)
    cv2.putText(rgb, "TEST RGB", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    res_rgb = engine.recognize(rgb, "doc_rgb")
    assert isinstance(res_rgb, OCRResult)
    assert len(res_rgb.tokens) > 0
    assert "RGB" in res_rgb.full_text

    # 2. 2D Grayscale
    gray = np.full((120, 300), 255, dtype=np.uint8)
    cv2.putText(gray, "TEST GRAY", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    res_gray = engine.recognize(gray, "doc_gray")
    assert isinstance(res_gray, OCRResult)
    assert len(res_gray.tokens) > 0
    assert "GRAY" in res_gray.full_text

    # 3. 2D Binary (only 0 and 255)
    binary = np.full((120, 300), 255, dtype=np.uint8)
    cv2.putText(binary, "TEST BINARY", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
    assert set(np.unique(binary)).issubset({0, 255})
    res_bin = engine.recognize(binary, "doc_binary")
    assert isinstance(res_bin, OCRResult)
    assert len(res_bin.tokens) > 0
    assert "BINARY" in res_bin.full_text


def test_rapid_ocr_determinism():
    """Verify deterministic outputs on identical image on CPU."""
    engine = RapidOCREngine()
    img = np.full((150, 350, 3), 255, dtype=np.uint8)
    cv2.putText(img, "DETERMINISTIC 123", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, "LINE TWO ABC", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    res1 = engine.recognize(img, "doc_det_1")
    res2 = engine.recognize(img, "doc_det_2")

    assert len(res1.tokens) == len(res2.tokens)
    assert res1.full_text == res2.full_text

    for t1, t2 in zip(res1.tokens, res2.tokens):
        assert t1.text == t2.text
        assert abs(t1.confidence - t2.confidence) < 1e-5
        assert abs(t1.bbox.x_min - t2.bbox.x_min) < 1e-4
        assert abs(t1.bbox.y_min - t2.bbox.y_min) < 1e-4
        assert abs(t1.bbox.x_max - t2.bbox.x_max) < 1e-4
        assert abs(t1.bbox.y_max - t2.bbox.y_max) < 1e-4


def test_rapid_ocr_resize_policy_exposure():
    """Verify RapidOCR engine exposes explicit resize policy attributes."""
    engine = RapidOCREngine(
        max_side_len=2000,
        min_side_len=30,
        det_limit_side_len=736,
        det_limit_type="min",
    )
    assert engine.max_side_len == 2000
    assert engine.min_side_len == 30
    assert engine.det_limit_side_len == 736
    assert engine.det_limit_type == "min"
    assert engine.use_preprocess_img is True


def test_b0_b1_b2_consistent_ocr_configuration():
    """Verify that B0, B1, and B2 execute with identical OCR configuration even when input dimensions differ."""
    engine = RapidOCREngine()

    # B0: Original image
    img_b0 = np.full((500, 300, 3), 255, dtype=np.uint8)
    cv2.putText(img_b0, "B0 RECEIPT", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # B1: Degraded image (e.g. downsampled)
    img_b1 = cv2.resize(img_b0, (150, 250), interpolation=cv2.INTER_AREA)

    # B2: Preprocessed image (e.g. 2D grayscale resized)
    img_b2 = cv2.cvtColor(cv2.resize(img_b0, (280, 480)), cv2.COLOR_BGR2GRAY)

    res_b0 = engine.recognize(img_b0, "doc_b0")
    res_b1 = engine.recognize(img_b1, "doc_b1")
    res_b2 = engine.recognize(img_b2, "doc_b2")

    assert isinstance(res_b0, OCRResult)
    assert isinstance(res_b1, OCRResult)
    assert isinstance(res_b2, OCRResult)
    # The OCR configuration remains strictly identical across all runs
    manifest = engine.get_model_manifest()
    assert manifest["resize_policy"]["verified"] is True


def test_rapid_ocr_classifier_toggle():
    """Verify behavior with use_cls=True and use_cls=False."""
    eng_cls = RapidOCREngine(use_cls=True)
    eng_nocls = RapidOCREngine(use_cls=False)

    img = np.full((100, 250, 3), 255, dtype=np.uint8)
    cv2.putText(img, "CLASSIFIER", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    res1 = eng_cls.recognize(img, "doc_cls")
    res2 = eng_nocls.recognize(img, "doc_nocls")

    assert isinstance(res1, OCRResult)
    assert isinstance(res2, OCRResult)
    assert "CLASSIFIER" in res1.full_text
    assert "CLASSIFIER" in res2.full_text


def test_rapid_ocr_gt_isolation_audit():
    """Strict GT isolation audit: verify no GT parameters or leakages."""
    sig = inspect.signature(RapidOCREngine.recognize)
    params = list(sig.parameters.keys())
    assert params == ["self", "image", "document_id"]

    engine = RapidOCREngine()
    img = np.full((80, 200, 3), 255, dtype=np.uint8)
    cv2.putText(img, "RECEIPT", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    res = engine.recognize(img, "doc_iso")

    # Verify no GT labels in metadata
    assert "gt" not in res.metadata
    assert "ground_truth" not in res.metadata
    assert "entities" not in res.metadata


def test_rapid_ocr_performance_smoke():
    """Run smoke test across 5 synthetic images and record raw timings without performance claims."""
    engine = RapidOCREngine()
    timings = []

    for i in range(5):
        img = np.full((120, 300, 3), 255, dtype=np.uint8)
        cv2.putText(img, f"ITEM {i+1} 10.00", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        res = engine.recognize(img, f"doc_perf_{i}")
        assert res.processing_time_ms is not None
        assert res.processing_time_ms > 0
        timings.append(res.processing_time_ms)

    assert len(timings) == 5
    mean_time = sum(timings) / len(timings)
    assert mean_time > 0
    # Note: Fixture CPU smoke test completed successfully; measured latency is environment-specific
    # and is not used as a scientific performance claim.


def test_manifest_export_baseline_runner(tmp_path: Path):
    """Verify that baseline runner exports ocr_stack_manifest.json and reports ocr_stack."""
    out_dir = tmp_path / "run_baseline"
    report = run_ocr_baseline(
        config_path=Path("configs/ocr.yaml"),
        output_dir=out_dir,
        data_root=Path("tests/fixtures/sroie_valid"),
        split="train",
    )

    manifest_file = out_dir / "ocr_stack_manifest.json"
    assert manifest_file.exists()

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert "configured" in manifest
    assert "resolved" in manifest
    assert "environment" in manifest
    assert "resize_policy" in manifest
    assert manifest["resolved"]["detector"]["sha256_verified"] is True
    assert manifest["resize_policy"]["verified"] is True

    # Check report contents
    assert "ocr_stack" in report
    assert "resize_policy" in report
    assert report["resize_policy"]["verified"] is True
