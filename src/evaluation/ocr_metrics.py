"""OCR evaluation metrics: CER, WER, Character-NED similarity, and text normalizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple
import unicodedata


def compute_levenshtein_distance(seq1: Sequence[Any], seq2: Sequence[Any]) -> int:
    """Compute standard Levenshtein edit distance between two sequences.

    Supports characters (strings) and words (lists of strings).
    Uses O(min(len1, len2)) memory.
    """
    if len(seq1) < len(seq2):
        seq1, seq2 = seq2, seq1

    if len(seq2) == 0:
        return len(seq1)

    previous_row = list(range(len(seq2) + 1))
    for i, elem1 in enumerate(seq1):
        current_row = [i + 1] + [0] * len(seq2)
        for j, elem2 in enumerate(seq2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (0 if elem1 == elem2 else 1)
            current_row[j + 1] = min(insertions, deletions, substitutions)
        previous_row = current_row

    return previous_row[-1]


def normalize_ocr_text(text: str) -> str:
    """Normalize OCR text according to frozen research protocol.

    Rules:
    1. Unicode NFC canonical decomposition & composition.
    2. Removal of control/service characters (Unicode category 'C*').
    3. Whitespace collapse: tabs/newlines/multiple spaces -> single space.
    4. Strip leading and trailing whitespace.
    """
    if not text:
        return ""

    # 1. Unicode NFC
    normalized = unicodedata.normalize("NFC", text)

    # 2. Filter control characters (keep standard space)
    filtered_chars = [
        ch for ch in normalized
        if not unicodedata.category(ch).startswith("C") or ch in ("\n", "\r", "\t")
    ]
    cleaned = "".join(filtered_chars)

    # 3. Collapse multiple whitespaces and strip
    collapsed = " ".join(cleaned.split())
    return collapsed


def compute_cer(pred: str, gt: str) -> float:
    """Compute Character Error Rate (CER).

    Formula:
        CER = edit_distance(pred, gt) / max(len(gt), 1)

    Edge cases:
        - both empty: 0.0
        - empty GT, non-empty pred: 1.0
        - empty pred, non-empty GT: 1.0
    """
    if len(gt) == 0 and len(pred) == 0:
        return 0.0
    if len(gt) == 0:
        return 1.0

    dist = compute_levenshtein_distance(pred, gt)
    return float(dist / len(gt))


def compute_wer(pred: str, gt: str) -> float:
    """Compute Word Error Rate (WER).

    Words are defined by whitespace splitting.

    Formula:
        WER = word_level_edit_distance(pred_words, gt_words) / max(len(gt_words), 1)

    Edge cases:
        - both empty: 0.0
        - empty GT, non-empty pred: 1.0
        - empty pred, non-empty GT: 1.0
    """
    pred_words = pred.split()
    gt_words = gt.split()

    if len(gt_words) == 0 and len(pred_words) == 0:
        return 0.0
    if len(gt_words) == 0:
        return 1.0

    dist = compute_levenshtein_distance(pred_words, gt_words)
    return float(dist / len(gt_words))


def compute_character_ned_similarity(pred: str, gt: str) -> float:
    """Compute Character-NED similarity.

    Formula:
        Character-NED similarity = 1.0 - (edit_distance(pred, gt) / max(len(pred), len(gt), 1))

    Range: [0.0, 1.0]

    Edge cases:
        - identical: 1.0
        - both empty: 1.0
        - empty GT + non-empty pred: 0.0
        - empty pred + non-empty GT: 0.0
        - completely different equal-length strings (dist == len): 0.0
    """
    if len(gt) == 0 and len(pred) == 0:
        return 1.0
    if len(gt) == 0 or len(pred) == 0:
        return 0.0

    dist = compute_levenshtein_distance(pred, gt)
    max_len = max(len(pred), len(gt))

    ratio = dist / max_len
    similarity = max(0.0, 1.0 - ratio)
    return float(similarity)


@dataclass(frozen=True)
class OCREvaluationResult:
    """Evaluation metrics for a single document OCR prediction against ground truth."""

    raw_prediction: str
    raw_ground_truth: str
    normalized_prediction: str
    normalized_ground_truth: str
    cer_raw: float
    wer_raw: float
    char_ned_raw: float
    cer_normalized: float
    wer_normalized: float
    char_ned_normalized: float
    char_edit_distance_raw: int
    char_edit_distance_normalized: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_prediction": self.raw_prediction,
            "raw_ground_truth": self.raw_ground_truth,
            "normalized_prediction": self.normalized_prediction,
            "normalized_ground_truth": self.normalized_ground_truth,
            "cer_raw": self.cer_raw,
            "wer_raw": self.wer_raw,
            "char_ned_raw": self.char_ned_raw,
            "cer_normalized": self.cer_normalized,
            "wer_normalized": self.wer_normalized,
            "char_ned_normalized": self.char_ned_normalized,
            "char_edit_distance_raw": self.char_edit_distance_raw,
            "char_edit_distance_normalized": self.char_edit_distance_normalized,
        }


def evaluate_ocr(pred: str, gt: str) -> OCREvaluationResult:
    """Evaluate OCR prediction against ground truth, preserving raw and normalized metrics."""
    norm_pred = normalize_ocr_text(pred)
    norm_gt = normalize_ocr_text(gt)

    raw_char_dist = compute_levenshtein_distance(pred, gt)
    norm_char_dist = compute_levenshtein_distance(norm_pred, norm_gt)

    return OCREvaluationResult(
        raw_prediction=pred,
        raw_ground_truth=gt,
        normalized_prediction=norm_pred,
        normalized_ground_truth=norm_gt,
        cer_raw=round(compute_cer(pred, gt), 4),
        wer_raw=round(compute_wer(pred, gt), 4),
        char_ned_raw=round(compute_character_ned_similarity(pred, gt), 4),
        cer_normalized=round(compute_cer(norm_pred, norm_gt), 4),
        wer_normalized=round(compute_wer(norm_pred, norm_gt), 4),
        char_ned_normalized=round(compute_character_ned_similarity(norm_pred, norm_gt), 4),
        char_edit_distance_raw=raw_char_dist,
        char_edit_distance_normalized=norm_char_dist,
    )
