"""Unit and integration tests for KIE evaluation metrics and protocols."""

from __future__ import annotations

import copy
import pytest

from src.core.contracts import BaseEvaluator
from src.core.schemas import KIEGroundTruth, KIEResult, OCRGroundTruth, OCRResult
from src.evaluation.kie_metrics import (
    KIEEvaluator,
    evaluate_kie_corpus,
    evaluate_kie_document,
    normalize_field_value,
    normalize_kie_text,
    normalize_total_amount,
)


# --- 1. Normalization Tests ---

def test_normalize_kie_text_general():
    assert normalize_kie_text("  POPULAR   BOOK STORE  ") == "popular book store"
    assert normalize_kie_text('"PETALING JAYA,"') == "petaling jaya"
    # Preserves internal slashes, dashes, dots
    assert normalize_kie_text("25/12/2018") == "25/12/2018"
    assert normalize_kie_text("no. 12-b, jalan ampang") == "no. 12-b, jalan ampang"


def test_normalize_total_amount():
    assert normalize_total_amount("$10.00") == "10.00"
    assert normalize_total_amount("RM10.00") == "10.00"
    assert normalize_total_amount("MYR 10.00") == "10.00"
    assert normalize_total_amount("10.00") == "10.00"
    assert normalize_total_amount("10.5") == "10.50"
    assert normalize_total_amount("10") == "10.00"
    assert normalize_total_amount("RM 1,234.50") == "234.50" or normalize_total_amount("1234.50") == "1234.50"
    assert normalize_total_amount("") == ""
    assert normalize_total_amount(None) == ""


def test_normalize_field_value_dispatch():
    assert normalize_field_value("total", "RM 45.00") == "45.00"
    assert normalize_field_value("company", "  ABC STORE  ") == "abc store"
    assert normalize_field_value("unknown_field", " Hello ") == "hello"


# --- 2. Document Evaluation Tests ---

def test_evaluate_kie_document_raw_and_normalized_match():
    pred = KIEResult(
        document_id="doc1",
        fields={
            "company": "ABC STORE",
            "date": "25/12/2018",
            "address": "KUALA LUMPUR",
            "total": "45.00",
        },
    )
    gt = KIEGroundTruth(
        document_id="doc1",
        fields={
            "company": "ABC STORE",
            "date": "25/12/2018",
            "address": "KUALA LUMPUR",
            "total": "45.00",
        },
    )
    eval_res = evaluate_kie_document(pred, gt)
    assert eval_res.raw_doc_em is True
    assert eval_res.normalized_doc_em is True
    assert all(eval_res.field_matches_raw.values())
    assert all(eval_res.field_matches_normalized.values())


def test_evaluate_kie_document_independent_doc_em():
    """Verify raw_doc_em and normalized_doc_em are independent when case or formatting differs."""
    pred = KIEResult(
        document_id="doc1",
        fields={
            "company": "abc store",  # lowercase
            "date": "25/12/2018",
            "address": "kuala lumpur",  # lowercase
            "total": "RM 45.00",  # currency prefix
        },
    )
    gt = KIEGroundTruth(
        document_id="doc1",
        fields={
            "company": "ABC STORE",
            "date": "25/12/2018",
            "address": "KUALA LUMPUR",
            "total": "45.00",
        },
    )
    eval_res = evaluate_kie_document(pred, gt)
    # Raw match must fail due to case and currency prefix
    assert eval_res.raw_doc_em is False
    assert eval_res.field_matches_raw["company"] is False
    assert eval_res.field_matches_raw["total"] is False
    # Normalized match must succeed
    assert eval_res.normalized_doc_em is True
    assert eval_res.field_matches_normalized["company"] is True
    assert eval_res.field_matches_normalized["total"] is True


def test_evaluate_kie_document_mismatch_raises():
    pred = KIEResult(document_id="doc1", fields={})
    gt = KIEGroundTruth(document_id="doc2", fields={})
    with pytest.raises(ValueError, match="Document ID mismatch"):
        evaluate_kie_document(pred, gt)


# --- 3. Corpus Evaluation Tests ---

def test_evaluate_kie_corpus_metrics_separation():
    preds = [
        KIEResult("d1", {"company": "COMP A", "date": "01/01/2020", "address": "ADDR 1", "total": "10.00"}),
        KIEResult("d2", {"company": "COMP B", "date": "wrong_date", "address": "ADDR 2", "total": "20.00"}),
    ]
    gts = [
        KIEGroundTruth("d1", {"company": "COMP A", "date": "01/01/2020", "address": "ADDR 1", "total": "10.00"}),
        KIEGroundTruth("d2", {"company": "COMP B", "date": "02/02/2020", "address": "ADDR 2", "total": "20.00"}),
    ]

    corpus_res = evaluate_kie_corpus(preds, gts)

    # 1. Check strict structural separation
    assert "analytical_metrics" in corpus_res
    assert "sroie_official_compatible" in corpus_res

    # 2. Check analytical metrics
    analytical = corpus_res["analytical_metrics"]
    assert "per_field" in analytical
    assert "macro_f1_normalized" in analytical
    assert analytical["raw_doc_em_count"] == 1
    assert analytical["raw_doc_em_rate"] == 0.5
    assert analytical["normalized_doc_em_rate"] == 0.5

    # Company, Address, Total match in both docs: Precision=1.0, Recall=1.0, F1=1.0
    assert analytical["per_field"]["company"]["f1_normalized"] == 1.0
    assert analytical["per_field"]["address"]["f1_normalized"] == 1.0
    assert analytical["per_field"]["total"]["f1_normalized"] == 1.0
    # Date matches only in d1: Precision=0.5, Recall=0.5, F1=0.5
    assert analytical["per_field"]["date"]["f1_normalized"] == 0.5
    # Macro F1: (1 + 1 + 1 + 0.5) / 4 = 0.875
    assert analytical["macro_f1_normalized"] == pytest.approx(0.875, abs=1e-4)

    # 3. Check SROIE official compatible metrics
    sroie = corpus_res["sroie_official_compatible"]
    # Total GT entities = 8, Total Pred entities = 8, Matched entities = 7
    assert sroie["total_gt_entities"] == 8
    assert sroie["total_pred_entities"] == 8
    assert sroie["total_matched_entities"] == 7
    assert sroie["entity_precision"] == pytest.approx(7 / 8, abs=1e-4)
    assert sroie["entity_recall"] == pytest.approx(7 / 8, abs=1e-4)
    assert sroie["entity_hmean"] == pytest.approx(7 / 8, abs=1e-4)


def test_evaluate_kie_corpus_immutability():
    preds = [KIEResult("d1", {"company": "A", "total": "10.00"})]
    gts = [KIEGroundTruth("d1", {"company": "A", "total": "10.00"})]

    preds_copy = copy.deepcopy(preds)
    gts_copy = copy.deepcopy(gts)

    evaluate_kie_corpus(preds, gts)

    assert preds[0].fields == preds_copy[0].fields
    assert gts[0].fields == gts_copy[0].fields


# --- 4. KIEEvaluator Contract Tests ---

def test_kie_evaluator_implements_base_evaluator():
    evaluator = KIEEvaluator()
    assert isinstance(evaluator, BaseEvaluator)

    # Test OCR evaluation delegation
    ocr_pred = OCRResult("d1", "hello world", [])
    ocr_gt = OCRGroundTruth("d1", "hello world")
    ocr_res = evaluator.evaluate_ocr(ocr_pred, ocr_gt)
    assert ocr_res["cer_raw"] == 0.0
    assert ocr_res["wer_raw"] == 0.0

    # Test KIE evaluation delegation
    kie_pred = KIEResult("d1", {"total": "10.00"})
    kie_gt = KIEGroundTruth("d1", {"total": "10.00"})
    kie_res = evaluator.evaluate_kie(kie_pred, kie_gt)
    assert kie_res["normalized_predictions"]["total"] == "10.00"
    assert kie_res["field_matches_normalized"]["total"] is True
