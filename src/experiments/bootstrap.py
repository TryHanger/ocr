"""Statistical aggregation and document-level bootstrap confidence interval calculation."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np


def compute_bootstrap_ci(
    values: Sequence[float],
    iterations: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    stat_func: Optional[Any] = None,
) -> Tuple[float, float]:
    """Compute non-parametric percentile bootstrap confidence interval using document-level resampling.

    Args:
        values: Sequence of numerical document-level metric values.
        iterations: Number of bootstrap iterations (default: 1000).
        ci: Confidence interval nominal coverage (default: 0.95).
        seed: Deterministic seed for reproducible bootstrap sampling.
        stat_func: Estimator function to compute on each bootstrap sample (default: np.mean).

    Returns:
        Tuple of (ci_lower, ci_upper) rounded to 4 decimals.
    """
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        val = float(round(arr[0], 4))
        return (val, val)

    estimator = stat_func if stat_func is not None else np.mean
    rng = np.random.default_rng(seed)

    # Document-level resampling with replacement
    boot_indices = rng.integers(0, n, size=(iterations, n))
    boot_samples = arr[boot_indices]
    boot_stats = np.apply_along_axis(estimator, 1, boot_samples)

    alpha = 1.0 - ci
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    lower_val = float(np.percentile(boot_stats, lower_pct))
    upper_val = float(np.percentile(boot_stats, upper_pct))

    return round(lower_val, 4), round(upper_val, 4)


def aggregate_condition_records(
    records: List[Dict[str, Any]],
    bootstrap_iterations: int = 1000,
    ci: float = 0.95,
    bootstrap_seed: int = 42,
) -> Dict[str, Any]:
    """Aggregate per-document evaluation records for a single experiment condition.

    Computes:
    - requested, successful, failed document counts and failure rate;
    - OCR metrics: mean, median, std, and 95% bootstrap CI for CER, WER, Character-NED;
    - KIE metrics: per-field, Macro P/R/F1, Doc-EM, SROIE entity Hmean and bootstrap CIs.
    """
    n_requested = len(records)
    successful_records = [r for r in records if r.get("status") == "success"]
    failed_records = [r for r in records if r.get("status") != "success"]
    n_successful = len(successful_records)
    n_failed = len(failed_records)
    failure_rate = round(n_failed / n_requested, 4) if n_requested > 0 else 0.0

    if n_successful == 0:
        return {
            "document_counts": {
                "requested": n_requested,
                "successful": n_successful,
                "failed": n_failed,
                "failure_rate": failure_rate,
            },
            "ocr": {},
            "kie": {},
            "status": "ALL_FAILED" if n_requested > 0 else "EMPTY",
        }

    # Extract OCR metric series from successful records
    cer_raw_list = [r["ocr_metrics"]["cer_raw"] for r in successful_records if "ocr_metrics" in r]
    wer_raw_list = [r["ocr_metrics"]["wer_raw"] for r in successful_records if "ocr_metrics" in r]
    char_ned_raw_list = [r["ocr_metrics"]["char_ned_raw"] for r in successful_records if "ocr_metrics" in r]

    cer_norm_list = [r["ocr_metrics"]["cer_normalized"] for r in successful_records if "ocr_metrics" in r]
    wer_norm_list = [r["ocr_metrics"]["wer_normalized"] for r in successful_records if "ocr_metrics" in r]
    char_ned_norm_list = [r["ocr_metrics"]["char_ned_normalized"] for r in successful_records if "ocr_metrics" in r]

    def _stats(vals: List[float], b_seed: int) -> Dict[str, Any]:
        if not vals:
            return {"mean": 0.0, "median": 0.0, "std": 0.0, "ci_95": [0.0, 0.0]}
        arr = np.asarray(vals, dtype=np.float64)
        m = float(np.mean(arr))
        med = float(np.median(arr))
        s = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        ci_low, ci_high = compute_bootstrap_ci(vals, iterations=bootstrap_iterations, ci=ci, seed=b_seed)
        return {
            "mean": round(m, 4),
            "median": round(med, 4),
            "std": round(s, 4),
            "ci_95": [ci_low, ci_high],
        }

    ocr_summary = {
        "sample_size": len(cer_raw_list),
        "cer_raw": _stats(cer_raw_list, bootstrap_seed + 1),
        "wer_raw": _stats(wer_raw_list, bootstrap_seed + 2),
        "char_ned_raw": _stats(char_ned_raw_list, bootstrap_seed + 3),
        "cer_normalized": _stats(cer_norm_list, bootstrap_seed + 4),
        "wer_normalized": _stats(wer_norm_list, bootstrap_seed + 5),
        "char_ned_normalized": _stats(char_ned_norm_list, bootstrap_seed + 6),
    }

    # Extract KIE metrics
    # Per-field matches
    target_fields = ("company", "date", "address", "total")
    per_field_matches_raw = {f: [1.0 if r["kie_metrics"]["field_matches_raw"].get(f, False) else 0.0 for r in successful_records if "kie_metrics" in r] for f in target_fields}
    per_field_matches_norm = {f: [1.0 if r["kie_metrics"]["field_matches_normalized"].get(f, False) else 0.0 for r in successful_records if "kie_metrics" in r] for f in target_fields}

    raw_doc_em_list = [1.0 if r["kie_metrics"].get("raw_doc_em", False) else 0.0 for r in successful_records if "kie_metrics" in r]
    norm_doc_em_list = [1.0 if r["kie_metrics"].get("normalized_doc_em", False) else 0.0 for r in successful_records if "kie_metrics" in r]

    # Per-document macro f1 approximation (mean of matches across fields)
    doc_macro_f1_norm = [
        float(np.mean([1.0 if r["kie_metrics"]["field_matches_normalized"].get(f, False) else 0.0 for f in target_fields]))
        for r in successful_records if "kie_metrics" in r
    ]

    per_field_summary: Dict[str, Any] = {}
    for f in target_fields:
        r_vals = per_field_matches_raw[f]
        n_vals = per_field_matches_norm[f]
        per_field_summary[f] = {
            "match_rate_raw": round(float(np.mean(r_vals)), 4) if r_vals else 0.0,
            "match_rate_normalized": round(float(np.mean(n_vals)), 4) if n_vals else 0.0,
        }

    raw_doc_em_stats = _stats(raw_doc_em_list, bootstrap_seed + 10)
    norm_doc_em_stats = _stats(norm_doc_em_list, bootstrap_seed + 11)
    macro_f1_stats = _stats(doc_macro_f1_norm, bootstrap_seed + 12)

    # SROIE official compatible entity metrics across all successful docs
    sroie_tp = 0
    sroie_pred = 0
    sroie_gt = 0
    for r in successful_records:
        if "kie_metrics" in r:
            km = r["kie_metrics"]
            for f in target_fields:
                has_gt = bool(km.get("normalized_ground_truth", {}).get(f, "").strip())
                has_pr = bool(km.get("normalized_predictions", {}).get(f, "").strip())
                if has_gt:
                    sroie_gt += 1
                if has_pr:
                    sroie_pred += 1
                if has_gt and has_pr and km.get("field_matches_normalized", {}).get(f, False):
                    sroie_tp += 1

    s_prec = round(sroie_tp / sroie_pred, 4) if sroie_pred > 0 else 0.0
    s_rec = round(sroie_tp / sroie_gt, 4) if sroie_gt > 0 else 0.0
    s_hmean = round(2.0 * s_prec * s_rec / (s_prec + s_rec), 4) if (s_prec + s_rec) > 0 else 0.0

    kie_summary = {
        "sample_size": len(raw_doc_em_list),
        "per_field": per_field_summary,
        "raw_doc_em_rate": raw_doc_em_stats["mean"],
        "raw_doc_em_ci_95": raw_doc_em_stats["ci_95"],
        "normalized_doc_em_rate": norm_doc_em_stats["mean"],
        "normalized_doc_em_ci_95": norm_doc_em_stats["ci_95"],
        "macro_f1_normalized": macro_f1_stats["mean"],
        "macro_f1_ci_95": macro_f1_stats["ci_95"],
        "sroie_official_compatible": {
            "entity_precision": s_prec,
            "entity_recall": s_rec,
            "entity_hmean": s_hmean,
            "total_gt_entities": sroie_gt,
            "total_pred_entities": sroie_pred,
            "total_matched_entities": sroie_tp,
        },
    }

    return {
        "document_counts": {
            "requested": n_requested,
            "successful": n_successful,
            "failed": n_failed,
            "failure_rate": failure_rate,
        },
        "ocr": ocr_summary,
        "kie": kie_summary,
        "status": "PARTIAL_FAILURE" if n_failed > 0 else "SUCCESS",
    }
