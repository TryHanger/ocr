"""KIE evaluation metrics: per-field P/R/F1, Macro P/R/F1, Doc-EM, and SROIE Task-3 Hmean."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Sequence
import unicodedata

from src.core.contracts import BaseEvaluator
from src.core.schemas import KIEGroundTruth, KIEResult, OCRGroundTruth, OCRResult
from src.evaluation.ocr_metrics import evaluate_ocr as evaluate_ocr_func

TARGET_FIELDS = ("company", "date", "address", "total")


def normalize_kie_text(text: str) -> str:
    """Normalize general KIE text field according to research protocol.

    Rules:
    1. Unicode NFC canonical decomposition & composition.
    2. Filter control/service characters.
    3. Lowercase.
    4. Strip surrounding punctuation (e.g. quotes, periods, commas, colons at edges)
       while preserving internal slashes, dashes, dots, and hyphens.
    5. Collapse multiple whitespaces and strip.
    """
    if not text:
        return ""

    # 1. Unicode NFC
    normalized = unicodedata.normalize("NFC", str(text))

    # 2. Filter control characters
    filtered = [
        ch for ch in normalized
        if not unicodedata.category(ch).startswith("C") or ch in ("\n", "\r", "\t")
    ]
    cleaned = "".join(filtered).lower()

    # 3. Collapse multiple whitespaces
    collapsed = " ".join(cleaned.split())

    # 4. Strip surrounding punctuation without deleting internal punctuation
    collapsed = collapsed.strip(" '\"`:,.;#~!?*[](){}")

    return collapsed


def normalize_total_amount(val: Any) -> str:
    """Normalize total amount field for evaluator-side comparison.

    CRITICAL PROTOCOL NOTE:
    This numeric canonicalization is STRICTLY evaluator-side as part of the
    normalized evaluation protocol. It MUST NOT be applied during OCR or KIE inference.
    In raw evaluation, strict string equality (prediction == GT) applies without this formatting.

    Rules applied:
    1. Currency stripping: removes currency prefixes (RM, MYR, $) and suffixes.
    2. Whitespace normalization: strips leading/trailing whitespaces.
    3. Conservative numeric formatting: canonicalizes decimal representation (e.g. 10 -> 10.00, 10.5 -> 10.50).
    """
    if val is None:
        return ""

    s = str(val).strip()
    if not s:
        return ""

    # Remove currency prefixes and suffixes
    s = re.sub(r"^(?:RM|MYR|\$|\s)+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"(?:RM|MYR|\$|\s)+$", "", s, flags=re.IGNORECASE)
    s = s.strip()

    # Match numeric portion
    match = re.search(r"(\d+(?:\.\d+)?)", s)
    if match:
        num_str = match.group(1)
        if "." in num_str:
            parts = num_str.split(".")
            integer_part = parts[0]
            fractional_part = parts[1]
            if len(fractional_part) == 1:
                return f"{integer_part}.{fractional_part}0"
            elif len(fractional_part) == 2:
                return f"{integer_part}.{fractional_part}"
            else:
                return f"{integer_part}.{fractional_part[:2]}"
        else:
            return f"{num_str}.00"

    return s.lower()


def normalize_field_value(field_name: str, value: Any) -> str:
    """Dispatch normalization based on target field semantics."""
    if value is None:
        return ""
    if field_name == "total":
        return normalize_total_amount(value)
    return normalize_kie_text(str(value))


@dataclass(frozen=True)
class DocumentKIEEvaluation:
    """Document-level KIE evaluation comparison."""

    document_id: str
    target_fields: List[str]
    raw_predictions: Dict[str, str]
    raw_ground_truth: Dict[str, str]
    normalized_predictions: Dict[str, str]
    normalized_ground_truth: Dict[str, str]
    field_matches_raw: Dict[str, bool]
    field_matches_normalized: Dict[str, bool]
    raw_doc_em: bool
    normalized_doc_em: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "target_fields": self.target_fields,
            "raw_predictions": self.raw_predictions,
            "raw_ground_truth": self.raw_ground_truth,
            "normalized_predictions": self.normalized_predictions,
            "normalized_ground_truth": self.normalized_ground_truth,
            "field_matches_raw": self.field_matches_raw,
            "field_matches_normalized": self.field_matches_normalized,
            "raw_doc_em": self.raw_doc_em,
            "normalized_doc_em": self.normalized_doc_em,
        }


def evaluate_kie_document(
    prediction: KIEResult,
    ground_truth: KIEGroundTruth,
    target_fields: Sequence[str] = TARGET_FIELDS,
) -> DocumentKIEEvaluation:
    """Evaluate KIE prediction against ground truth for a single document.

    Calculates independent raw and normalized exact matches for each field,
    and document-level exact match (Doc-EM).
    """
    if prediction.document_id != ground_truth.document_id:
        raise ValueError(
            f"Document ID mismatch: prediction '{prediction.document_id}' vs GT '{ground_truth.document_id}'"
        )

    raw_preds: Dict[str, str] = {}
    raw_gts: Dict[str, str] = {}
    norm_preds: Dict[str, str] = {}
    norm_gts: Dict[str, str] = {}
    matches_raw: Dict[str, bool] = {}
    matches_norm: Dict[str, bool] = {}

    for f in target_fields:
        r_pred = str(prediction.fields.get(f, "")) if prediction.fields.get(f) is not None else ""
        r_gt = str(ground_truth.fields.get(f, "")) if ground_truth.fields.get(f) is not None else ""

        n_pred = normalize_field_value(f, r_pred)
        n_gt = normalize_field_value(f, r_gt)

        raw_preds[f] = r_pred
        raw_gts[f] = r_gt
        norm_preds[f] = n_pred
        norm_gts[f] = n_gt

        # Match condition: non-empty GT matches non-empty prediction
        if r_gt == "":
            # If ground truth does not have this entity: match only if prediction also empty
            matches_raw[f] = (r_pred == "")
            matches_norm[f] = (n_pred == "")
        else:
            matches_raw[f] = (r_pred == r_gt)
            matches_norm[f] = (n_pred == n_gt)

    raw_doc_em = all(matches_raw.get(f, False) for f in target_fields)
    norm_doc_em = all(matches_norm.get(f, False) for f in target_fields)

    return DocumentKIEEvaluation(
        document_id=prediction.document_id,
        target_fields=list(target_fields),
        raw_predictions=raw_preds,
        raw_ground_truth=raw_gts,
        normalized_predictions=norm_preds,
        normalized_ground_truth=norm_gts,
        field_matches_raw=matches_raw,
        field_matches_normalized=matches_norm,
        raw_doc_em=raw_doc_em,
        normalized_doc_em=norm_doc_em,
    )


def evaluate_kie_corpus(
    predictions: Sequence[KIEResult],
    ground_truths: Sequence[KIEGroundTruth],
    target_fields: Sequence[str] = TARGET_FIELDS,
) -> Dict[str, Any]:
    """Evaluate batch/corpus of KIE predictions against ground truths.

    Produces two clearly separated metric sections:
    1. analytical_metrics:
       - Per-field precision, recall, f1 (raw and normalized).
       - Macro precision, recall, f1 across fields.
       - raw_doc_em and normalized_doc_em corpus rates.
    2. sroie_official_compatible:
       - Entity-level micro precision, recall, Hmean per ICDAR SROIE Task 3.
    """
    gt_map = {gt.document_id: gt for gt in ground_truths}
    doc_evals: List[DocumentKIEEvaluation] = []

    for pred in predictions:
        if pred.document_id not in gt_map:
            raise KeyError(f"No ground truth found for predicted document '{pred.document_id}'")
        doc_evals.append(evaluate_kie_document(pred, gt_map[pred.document_id], target_fields))

    n_docs = len(doc_evals)
    if n_docs == 0:
        return {
            "analytical_metrics": {
                "per_field": {},
                "macro_precision_raw": 0.0,
                "macro_recall_raw": 0.0,
                "macro_f1_raw": 0.0,
                "macro_precision_normalized": 0.0,
                "macro_recall_normalized": 0.0,
                "macro_f1_normalized": 0.0,
                "raw_doc_em_rate": 0.0,
                "normalized_doc_em_rate": 0.0,
                "total_documents": 0,
            },
            "sroie_official_compatible": {
                "entity_precision": 0.0,
                "entity_recall": 0.0,
                "entity_hmean": 0.0,
                "total_gt_entities": 0,
                "total_pred_entities": 0,
                "total_matched_entities": 0,
            },
        }

    # 1. Compute per-field analytical metrics
    per_field_metrics: Dict[str, Dict[str, float]] = {}
    p_raw_list, r_raw_list, f1_raw_list = [], [], []
    p_norm_list, r_norm_list, f1_norm_list = [], [], []

    # SROIE official micro accumulators
    sroie_tp = 0
    sroie_pred_count = 0
    sroie_gt_count = 0

    for f in target_fields:
        tp_raw, fp_raw, fn_raw = 0, 0, 0
        tp_norm, fp_norm, fn_norm = 0, 0, 0

        for de in doc_evals:
            has_gt_raw = bool(de.raw_ground_truth.get(f, "").strip())
            has_pred_raw = bool(de.raw_predictions.get(f, "").strip())
            has_gt_norm = bool(de.normalized_ground_truth.get(f, "").strip())
            has_pred_norm = bool(de.normalized_predictions.get(f, "").strip())

            # Raw counts
            if has_gt_raw and has_pred_raw:
                if de.field_matches_raw[f]:
                    tp_raw += 1
                else:
                    fp_raw += 1
                    fn_raw += 1
            elif has_pred_raw and not has_gt_raw:
                fp_raw += 1
            elif has_gt_raw and not has_pred_raw:
                fn_raw += 1

            # Normalized counts
            if has_gt_norm and has_pred_norm:
                if de.field_matches_normalized[f]:
                    tp_norm += 1
                else:
                    fp_norm += 1
                    fn_norm += 1
            elif has_pred_norm and not has_gt_norm:
                fp_norm += 1
            elif has_gt_norm and not has_pred_norm:
                fn_norm += 1

            # SROIE official micro counts (entity-level normalized)
            if has_gt_norm:
                sroie_gt_count += 1
            if has_pred_norm:
                sroie_pred_count += 1
            if has_gt_norm and has_pred_norm and de.field_matches_normalized[f]:
                sroie_tp += 1

        def _calc_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            return round(prec, 4), round(rec, 4), round(f1, 4)

        p_r, r_r, f1_r = _calc_prf(tp_raw, fp_raw, fn_raw)
        p_n, r_n, f1_n = _calc_prf(tp_norm, fp_norm, fn_norm)

        per_field_metrics[f] = {
            "precision_raw": p_r,
            "recall_raw": r_r,
            "f1_raw": f1_r,
            "precision_normalized": p_n,
            "recall_normalized": r_n,
            "f1_normalized": f1_n,
            "tp_raw": tp_raw,
            "fp_raw": fp_raw,
            "fn_raw": fn_raw,
            "tp_normalized": tp_norm,
            "fp_normalized": fp_norm,
            "fn_normalized": fn_norm,
        }

        p_raw_list.append(p_r)
        r_raw_list.append(r_r)
        f1_raw_list.append(f1_r)
        p_norm_list.append(p_n)
        r_norm_list.append(r_n)
        f1_norm_list.append(f1_n)

    # Macro averages
    macro_p_raw = round(sum(p_raw_list) / len(p_raw_list), 4) if p_raw_list else 0.0
    macro_r_raw = round(sum(r_raw_list) / len(r_raw_list), 4) if r_raw_list else 0.0
    macro_f1_raw = round(sum(f1_raw_list) / len(f1_raw_list), 4) if f1_raw_list else 0.0

    macro_p_norm = round(sum(p_norm_list) / len(p_norm_list), 4) if p_norm_list else 0.0
    macro_r_norm = round(sum(r_norm_list) / len(r_norm_list), 4) if r_norm_list else 0.0
    macro_f1_norm = round(sum(f1_norm_list) / len(f1_norm_list), 4) if f1_norm_list else 0.0

    # Doc-EM rates
    raw_doc_em_count = sum(1 for de in doc_evals if de.raw_doc_em)
    norm_doc_em_count = sum(1 for de in doc_evals if de.normalized_doc_em)

    raw_doc_em_rate = round(raw_doc_em_count / n_docs, 4)
    norm_doc_em_rate = round(norm_doc_em_count / n_docs, 4)

    # SROIE official compatible micro metrics
    sroie_precision = (
        round(sroie_tp / sroie_pred_count, 4) if sroie_pred_count > 0 else 0.0
    )
    sroie_recall = (
        round(sroie_tp / sroie_gt_count, 4) if sroie_gt_count > 0 else 0.0
    )
    sroie_hmean = (
        round(
            (2.0 * sroie_precision * sroie_recall) / (sroie_precision + sroie_recall),
            4,
        )
        if (sroie_precision + sroie_recall) > 0
        else 0.0
    )

    return {
        "analytical_metrics": {
            "per_field": per_field_metrics,
            "macro_precision_raw": macro_p_raw,
            "macro_recall_raw": macro_r_raw,
            "macro_f1_raw": macro_f1_raw,
            "macro_precision_normalized": macro_p_norm,
            "macro_recall_normalized": macro_r_norm,
            "macro_f1_normalized": macro_f1_norm,
            "raw_doc_em_count": raw_doc_em_count,
            "raw_doc_em_rate": raw_doc_em_rate,
            "normalized_doc_em_count": norm_doc_em_count,
            "normalized_doc_em_rate": norm_doc_em_rate,
            "total_documents": n_docs,
        },
        "sroie_official_compatible": {
            "entity_precision": sroie_precision,
            "entity_recall": sroie_recall,
            "entity_hmean": sroie_hmean,
            "total_gt_entities": sroie_gt_count,
            "total_pred_entities": sroie_pred_count,
            "total_matched_entities": sroie_tp,
        },
    }


class KIEEvaluator(BaseEvaluator):
    """Evaluator implementing BaseEvaluator for KIE and OCR evaluation."""

    def __init__(self, target_fields: Sequence[str] = TARGET_FIELDS) -> None:
        self.target_fields = list(target_fields)

    def evaluate_ocr(
        self, prediction: OCRResult, ground_truth: OCRGroundTruth
    ) -> Dict[str, Any]:
        """Compute OCR quality metrics."""
        return evaluate_ocr_func(prediction.full_text, ground_truth.text).to_dict()

    def evaluate_kie(
        self, prediction: KIEResult, ground_truth: KIEGroundTruth
    ) -> Dict[str, Any]:
        """Compute KIE extraction metrics for a single document."""
        doc_eval = evaluate_kie_document(prediction, ground_truth, self.target_fields)
        return doc_eval.to_dict()

    def evaluate_batch(
        self,
        predictions: Sequence[KIEResult],
        ground_truths: Sequence[KIEGroundTruth],
    ) -> Dict[str, Any]:
        """Compute corpus-level KIE metrics."""
        return evaluate_kie_corpus(predictions, ground_truths, self.target_fields)
