"""Evaluation metrics module for OCR and KIE."""

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
    "compute_levenshtein_distance",
    "normalize_ocr_text",
    "compute_cer",
    "compute_wer",
    "compute_character_ned_similarity",
    "OCREvaluationResult",
    "evaluate_ocr",
]
