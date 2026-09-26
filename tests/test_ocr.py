"""Unit tests for OCR engines, reading order, GT leakage audit, and SROIE integration."""

import inspect
import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from src.core.contracts import BaseOCREngine
from src.core.schemas import BoundingBox, OCRResult, OCRToken
from src.ocr import (
    BaseOCREnginePrimitive,
    MockOCREngine,
    RapidOCREngine,
    get_ocr_engine,
    sort_tokens_reading_order,
)
from scripts.run_ocr_baseline import run_ocr_baseline, main as baseline_main


# ==============================================================================
# 1. BaseOCREngine & Contract Tests
# ==============================================================================


def test_engines_satisfy_base_contract():
    mock_eng = MockOCREngine()
    rapid_eng = RapidOCREngine()

    assert isinstance(mock_eng, BaseOCREngine)
    assert isinstance(mock_eng, BaseOCREnginePrimitive)
    assert isinstance(rapid_eng, BaseOCREngine)
    assert isinstance(rapid_eng, BaseOCREnginePrimitive)


def test_base_ocr_input_validation():
    eng = MockOCREngine()

    # Non-ndarray
    with pytest.raises(TypeError, match="Image must be np.ndarray"):
        eng.recognize([[1, 2], [3, 4]], "doc_1")  # type: ignore

    # Non-uint8
    with pytest.raises(TypeError, match="Image dtype must be np.uint8"):
        eng.recognize(np.ones((10, 10, 3), dtype=np.float32), "doc_1")

    # Invalid ndim
    with pytest.raises(ValueError, match="Image ndim must be 2"):
        eng.recognize(np.ones(10, dtype=np.uint8), "doc_1")

    # Empty document_id
    with pytest.raises(ValueError, match="document_id must be a non-empty string"):
        eng.recognize(np.ones((10, 10, 3), dtype=np.uint8), "")


# ==============================================================================
# 2. GT Leakage Structural Audit
# ==============================================================================


def test_gt_leakage_audit_primary_api():
    """Verify that the primary OCR recognition API accepts strictly (image, document_id).

    GT annotations (boxes, transcriptions, entities) must never be passed to primary OCR.
    """
    base_sig = inspect.signature(BaseOCREngine.recognize)
    base_params = [p for p in base_sig.parameters.keys() if p != "self"]
    assert base_params == ["image", "document_id"]

    prim_sig = inspect.signature(BaseOCREnginePrimitive.recognize)
    prim_params = [p for p in prim_sig.parameters.keys() if p != "self"]
    assert prim_params == ["image", "document_id"]

    rapid_sig = inspect.signature(RapidOCREngine.recognize)
    rapid_params = [p for p in rapid_sig.parameters.keys() if p != "self"]
    assert rapid_params == ["image", "document_id"]

    # Verify no parameter contains 'gt' or 'ground_truth'
    for param in rapid_params:
        assert "gt" not in param.lower()
        assert "box" not in param.lower()
        assert "truth" not in param.lower()


def test_diagnostic_mode_isolated_from_primary_api():
    """Diagnostic GT-region mode is a separate method and requires explicit GT box input."""
    eng = MockOCREngine()
    assert hasattr(eng, "diagnostic_gt_region_recognition")
    sig = inspect.signature(eng.diagnostic_gt_region_recognition)
    assert "gt_boxes" in sig.parameters


# ==============================================================================
# 3. Reading Order Strategy Tests
# ==============================================================================


def test_reading_order_two_lines_left_to_right():
    """Tokens on distinct lines sorted top-to-bottom; within line sorted left-to-right."""
    # Line 1: y around 20 (height 10)
    tok1 = OCRToken(text="Receipt", bbox=BoundingBox(10.0, 20.0, 50.0, 30.0))
    tok2 = OCRToken(text="#1024", bbox=BoundingBox(60.0, 20.0, 90.0, 30.0))

    # Line 2: y around 60 (height 10)
    tok3 = OCRToken(text="Total:", bbox=BoundingBox(10.0, 60.0, 45.0, 70.0))
    tok4 = OCRToken(text="$42.50", bbox=BoundingBox(55.0, 60.0, 95.0, 70.0))

    # Pass in scrambled order
    scrambled = [tok4, tok1, tok3, tok2]
    ordered_tokens, full_text = sort_tokens_reading_order(scrambled, line_tolerance_factor=0.5)

    assert [t.text for t in ordered_tokens] == ["Receipt", "#1024", "Total:", "$42.50"]
    assert full_text == "Receipt #1024\nTotal: $42.50"


def test_reading_order_tie_breaking_determinism():
    """Tokens with identical coordinates preserve deterministic ordering."""
    tok1 = OCRToken(text="WordA", bbox=BoundingBox(10.0, 10.0, 30.0, 20.0))
    tok2 = OCRToken(text="WordB", bbox=BoundingBox(10.0, 10.0, 30.0, 20.0))

    ordered1, text1 = sort_tokens_reading_order([tok1, tok2])
    ordered2, text2 = sort_tokens_reading_order([tok1, tok2])

    assert [t.text for t in ordered1] == [t.text for t in ordered2]
    assert text1 == text2


def test_reading_order_configurable_tolerance():
    """Varying line tolerance factor affects grouping of vertically offset tokens."""
    tok1 = OCRToken(text="A", bbox=BoundingBox(10.0, 20.0, 30.0, 30.0))
    # tok2 slightly lower (y_min 26, overlap exists)
    tok2 = OCRToken(text="B", bbox=BoundingBox(40.0, 26.0, 60.0, 36.0))

    # Tight tolerance -> split into 2 lines
    _, text_tight = sort_tokens_reading_order([tok1, tok2], line_tolerance_factor=0.1)
    assert text_tight == "A\nB"

    # Relaxed tolerance -> merged into 1 line
    _, text_relaxed = sort_tokens_reading_order([tok1, tok2], line_tolerance_factor=0.8)
    assert text_relaxed == "A B"


def test_reading_order_empty():
    tokens, text = sort_tokens_reading_order([])
    assert tokens == []
    assert text == ""


# ==============================================================================
# 4. RapidOCR Real Local Inference Tests
# ==============================================================================


def test_rapid_ocr_inference_on_text_image():
    """Verify real RapidOCR execution on a synthetic receipt text snippet."""
    eng = RapidOCREngine(min_confidence=0.1)

    img = np.full((120, 250, 3), 255, dtype=np.uint8)
    cv2.putText(img, "INVOICE #99", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(img, "TOTAL 12.50", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    result = eng.recognize(img, "doc_test_synthetic")

    assert isinstance(result, OCRResult)
    assert result.document_id == "doc_test_synthetic"
    assert len(result.tokens) > 0
    assert "INVOICE" in result.full_text or "TOTAL" in result.full_text
    assert result.processing_time_ms is not None and result.processing_time_ms > 0
    assert result.model_name == "RapidOCR"
    assert "PP-OCRv4" in str(result.model_version)

    # Verify token bounding boxes and confidences
    for tok in result.tokens:
        assert isinstance(tok.bbox, BoundingBox)
        assert tok.bbox.x_max > tok.bbox.x_min
        assert tok.bbox.y_max > tok.bbox.y_min
        assert 0.0 <= tok.confidence <= 1.0

    # Verify raw quadrilaterals preserved in metadata
    raw_quads = result.metadata.get("raw_quadrilaterals", [])
    assert len(raw_quads) == len(result.tokens)


def test_rapid_ocr_blank_image():
    """RapidOCR on pure white blank image should return empty tokens without error."""
    eng = RapidOCREngine()
    blank = np.full((100, 100, 3), 255, dtype=np.uint8)

    result = eng.recognize(blank, "doc_blank")

    assert isinstance(result, OCRResult)
    assert len(result.tokens) == 0
    assert result.full_text == ""


def test_rapid_ocr_grayscale_image():
    """RapidOCR safely handles 2D grayscale image."""
    eng = RapidOCREngine()
    gray = np.full((100, 200), 255, dtype=np.uint8)
    cv2.putText(gray, "PAID", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)

    result = eng.recognize(gray, "doc_gray")
    assert isinstance(result, OCRResult)


def test_rapid_ocr_diagnostic_gt_region():
    """Verify diagnostic GT-region recognition mode executes on specified regions."""
    eng = RapidOCREngine()
    img = np.full((100, 200, 3), 255, dtype=np.uint8)
    cv2.putText(img, "CASH", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    gt_boxes = [BoundingBox(15.0, 25.0, 100.0, 65.0)]
    diag_res = eng.diagnostic_gt_region_recognition(img, "doc_diag", gt_boxes)

    assert isinstance(diag_res, OCRResult)
    assert len(diag_res.tokens) == 1
    assert diag_res.metadata.get("mode") == "diagnostic_gt_region"


# ==============================================================================
# 5. Factory and Configuration Tests
# ==============================================================================


def test_get_ocr_engine_factory():
    eng_rapid = get_ocr_engine("rapidocr")
    assert isinstance(eng_rapid, RapidOCREngine)

    eng_mock = get_ocr_engine("mock")
    assert isinstance(eng_mock, MockOCREngine)

    with pytest.raises(KeyError, match="Unknown OCR engine"):
        get_ocr_engine("non_existent_engine")


# ==============================================================================
# 6. SROIE Baseline Integration Tests
# ==============================================================================


def test_sroie_baseline_runner_fixture(tmp_path: Path):
    out_dir = tmp_path / "baseline_out"
    report = run_ocr_baseline(
        config_path=Path("configs/ocr.yaml"),
        output_dir=out_dir,
        data_root=Path("tests/fixtures/sroie_valid"),
        split="train",
    )

    assert report["real_data_ocr_baseline"] == "PENDING"
    assert report["dataset_type"] == "synthetic_fixture"
    assert "summary_metrics" in report
    assert "mean_cer_raw" in report["summary_metrics"]
    assert "mean_char_ned_normalized" in report["summary_metrics"]

    report_file = out_dir / "ocr_baseline_report.json"
    assert report_file.exists()
    data = json.loads(report_file.read_text(encoding="utf-8"))
    assert data["real_data_ocr_baseline"] == "PENDING"


def test_cli_run_ocr_baseline(monkeypatch, tmp_path: Path):
    out_dir = tmp_path / "cli_baseline"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_ocr_baseline.py",
            "--config",
            "configs/ocr.yaml",
            "--output-dir",
            str(out_dir),
            "--data-root",
            "tests/fixtures/sroie_valid",
            "--split",
            "train",
        ],
    )
    code = baseline_main()
    assert code == 0
    assert (out_dir / "ocr_baseline_report.json").exists()
