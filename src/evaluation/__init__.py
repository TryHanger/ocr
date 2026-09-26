"""Evaluation metrics module for OCR and KIE."""

from src.evaluation.kie_metrics import (
    TARGET_FIELDS,
    DocumentKIEEvaluation,
    KIEEvaluator,
    evaluate_kie_corpus,
    evaluate_kie_document,
    normalize_field_value,
    normalize_kie_text,
    normalize_total_amount,
)
from src.evaluation.ocr_metrics import (
    OCREvaluationResult,
    compute_cer,
    compute_character_ned_similarity,
    compute_levenshtein_distance,
    compute_wer,
    evaluate_ocr,
    normalize_ocr_text,
)

__all__ = [
    # OCR metrics
    "compute_levenshtein_distance",
    "normalize_ocr_text",
    "compute_cer",
    "compute_wer",
    "compute_character_ned_similarity",
    "OCREvaluationResult",
    "evaluate_ocr",
    # KIE metrics
    "TARGET_FIELDS",
    "normalize_kie_text",
    "normalize_total_amount",
    "normalize_field_value",
    "DocumentKIEEvaluation",
    "evaluate_kie_document",
    "evaluate_kie_corpus",
    "KIEEvaluator",
]
