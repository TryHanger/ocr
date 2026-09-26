"""Unit tests for OCR evaluation metrics: CER, WER, Character-NED similarity, and normalizers."""

import pytest
import unicodedata

from src.evaluation.ocr_metrics import (
    compute_cer,
    compute_character_ned_similarity,
    compute_levenshtein_distance,
    compute_wer,
    evaluate_ocr,
    normalize_ocr_text,
)


def test_levenshtein_distance_cases():
    assert compute_levenshtein_distance("", "") == 0
    assert compute_levenshtein_distance("abc", "") == 3
    assert compute_levenshtein_distance("", "abc") == 3
    assert compute_levenshtein_distance("kitten", "sitting") == 3
    assert compute_levenshtein_distance("flaw", "lawn") == 2
    assert compute_levenshtein_distance(["a", "b"], ["a", "b"]) == 0
    assert compute_levenshtein_distance(["a", "b"], ["a", "c"]) == 1


# ==============================================================================
# 1. Character-NED Similarity Tests
# ==============================================================================


def test_character_ned_similarity_identical():
    assert compute_character_ned_similarity("TOTAL $12.50", "TOTAL $12.50") == 1.0


def test_character_ned_similarity_both_empty():
    assert compute_character_ned_similarity("", "") == 1.0


def test_character_ned_similarity_empty_gt():
    assert compute_character_ned_similarity("some text", "") == 0.0


def test_character_ned_similarity_empty_pred():
    assert compute_character_ned_similarity("", "some text") == 0.0


def test_character_ned_similarity_completely_different_equal_length():
    # "abc" vs "xyz": dist=3, max_len=3 -> 1 - 3/3 = 0.0
    assert compute_character_ned_similarity("abc", "xyz") == 0.0


def test_character_ned_similarity_partial_match():
    # "hello" vs "helo": dist=1, max_len=5 -> 1 - 1/5 = 0.8
    assert compute_character_ned_similarity("hello", "helo") == pytest.approx(0.8, abs=1e-4)


def test_character_ned_similarity_range():
    # Must always be bounded in [0.0, 1.0]
    res1 = compute_character_ned_similarity("short", "much longer string with different chars")
    assert 0.0 <= res1 <= 1.0


# ==============================================================================
# 2. CER (Character Error Rate) Tests
# ==============================================================================


def test_cer_both_empty():
    assert compute_cer("", "") == 0.0


def test_cer_empty_gt_non_empty_pred():
    assert compute_cer("pred text", "") == 1.0


def test_cer_empty_pred_non_empty_gt():
    assert compute_cer("", "gt text") == 1.0


def test_cer_known_value():
    # "kitten" vs "sitting": dist=3, len(gt)=7 -> 3/7
    assert compute_cer("kitten", "sitting") == pytest.approx(3.0 / 7.0, abs=1e-4)


# ==============================================================================
# 3. WER (Word Error Rate) Tests
# ==============================================================================


def test_wer_both_empty():
    assert compute_wer("", "") == 0.0


def test_wer_empty_gt():
    assert compute_wer("one two", "") == 1.0


def test_wer_empty_pred():
    assert compute_wer("", "one two") == 1.0


def test_wer_known_value():
    pred = "the fast brown fox"
    gt = "the quick brown fox"
    # 1 word substitution out of 4 words -> 1/4 = 0.25
    assert compute_wer(pred, gt) == 0.25


# ==============================================================================
# 4. Text Normalization Tests
# ==============================================================================


def test_normalize_ocr_text_whitespace_collapse():
    raw = "  TOTAL   INVOICE  \n\n  DATE:  2020-01-01 \t\t $10.00  "
    expected = "TOTAL INVOICE DATE: 2020-01-01 $10.00"
    assert normalize_ocr_text(raw) == expected


def test_normalize_ocr_text_unicode_nfc():
    # 'e' + combining acute accent -> NFC composed 'é'
    decomposed = "e\u0301"
    composed = "\u00e9"
    assert normalize_ocr_text(decomposed) == composed


def test_normalize_ocr_text_control_character_removal():
    # Control chars (null, bell, escape)
    raw = "Hello\x00World\x07!\x1b"
    assert normalize_ocr_text(raw) == "HelloWorld!"


def test_normalize_ocr_text_empty():
    assert normalize_ocr_text("") == ""


# ==============================================================================
# 5. Full OCREvaluationResult Tests
# ==============================================================================


def test_evaluate_ocr_preserves_four_representations():
    raw_pred = " TOTAL:  $12.50 \n"
    raw_gt = "TOTAL: $12.50"

    res = evaluate_ocr(raw_pred, raw_gt)

    assert res.raw_prediction == raw_pred
    assert res.raw_ground_truth == raw_gt
    assert res.normalized_prediction == "TOTAL: $12.50"
    assert res.normalized_ground_truth == "TOTAL: $12.50"

    # Normalized comparison is identical
    assert res.cer_normalized == 0.0
    assert res.wer_normalized == 0.0
    assert res.char_ned_normalized == 1.0

    # Raw comparison had extra spaces/newlines
    assert res.char_edit_distance_raw > 0

    d = res.to_dict()
    assert "cer_raw" in d
    assert "char_ned_normalized" in d
