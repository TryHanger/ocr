"""Compare B0 CPU (b0_clean_validation_n126) vs B0 GPU (b0_clean_validation_n126_gpu).

Evaluates the exact execution-provider effect across all N=126 validation documents:
- CER, WER, Character-NED
- Downstream KIE (Macro F1, Precision, Recall, Entity Hmean, Doc-EM)
- Per-document OCR text equivalence & exact matches
- Saves comparison artifact in experiments/benchmarks/b0_cpu_vs_gpu_comparison.json
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.ocr_metrics import (
    compute_cer,
    compute_character_ned_similarity,
    compute_wer,
    normalize_ocr_text,
)



def load_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            doc_id = item.get("document_id")
            if doc_id:
                records[doc_id] = item
    return records


def main() -> None:
    cpu_dir = Path("experiments/runs/b0_clean_validation_n126")
    gpu_dir = Path("experiments/runs/b0_clean_validation_n126_gpu")

    with open(cpu_dir / "summary.json", "r", encoding="utf-8") as f:
        cpu_summary = json.load(f)

    with open(gpu_dir / "summary.json", "r", encoding="utf-8") as f:
        gpu_summary = json.load(f)

    cpu_jsonl = load_jsonl(cpu_dir / "per_document.jsonl")
    gpu_jsonl = load_jsonl(gpu_dir / "per_document.jsonl")

    assert len(cpu_jsonl) == 126, f"Expected 126 CPU records, got {len(cpu_jsonl)}"
    assert len(gpu_jsonl) == 126, f"Expected 126 GPU records, got {len(gpu_jsonl)}"
    assert set(cpu_jsonl.keys()) == set(gpu_jsonl.keys()), "Document IDs mismatch between CPU and GPU B0 runs!"

    doc_ids = sorted(cpu_jsonl.keys())

    # Document-level OCR text comparison
    exact_text_matches = 0
    cer_vs_cpu_list = []
    wer_vs_cpu_list = []
    ned_vs_cpu_list = []
    token_diff_list = []
    per_doc_diffs = []

    kie_fields = ["company", "date", "address", "total"]
    kie_matches = {f: 0 for f in kie_fields}
    kie_diff_docs = {f: [] for f in kie_fields}

    for did in doc_ids:
        c_rec = cpu_jsonl[did]
        g_rec = gpu_jsonl[did]

        c_text = c_rec.get("ocr_result", {}).get("full_text", "")
        g_text = g_rec.get("ocr_result", {}).get("full_text", "")

        c_norm = normalize_ocr_text(c_text)
        g_norm = normalize_ocr_text(g_text)

        is_exact = (c_norm == g_norm)
        if is_exact:
            exact_text_matches += 1

        cer = compute_cer(g_norm, c_norm)
        wer = compute_wer(g_norm, c_norm)
        ned = compute_character_ned_similarity(g_norm, c_norm)

        c_tokens = len(c_rec.get("ocr_result", {}).get("tokens", []))
        g_tokens = len(g_rec.get("ocr_result", {}).get("tokens", []))
        t_diff = g_tokens - c_tokens

        cer_vs_cpu_list.append(cer)
        wer_vs_cpu_list.append(wer)
        ned_vs_cpu_list.append(ned)
        token_diff_list.append(t_diff)

        # KIE comparisons
        c_kie = c_rec.get("kie_result", {}).get("predictions", {})
        g_kie = g_rec.get("kie_result", {}).get("predictions", {})

        doc_kie_diff = {}
        for f in kie_fields:
            c_val = normalize_ocr_text(c_kie.get(f, {}).get("value", "") if isinstance(c_kie.get(f), dict) else str(c_kie.get(f) or ""))
            g_val = normalize_ocr_text(g_kie.get(f, {}).get("value", "") if isinstance(g_kie.get(f), dict) else str(g_kie.get(f) or ""))
            f_match = (c_val == g_val)
            if f_match:
                kie_matches[f] += 1
            else:
                kie_diff_docs[f].append({"doc_id": did, "cpu": c_val, "gpu": g_val})
            doc_kie_diff[f] = {"cpu": c_val, "gpu": g_val, "match": f_match}

        if not is_exact or not all(doc_kie_diff[f]["match"] for f in kie_fields):
            per_doc_diffs.append({
                "document_id": did,
                "exact_text_match": is_exact,
                "token_diff": t_diff,
                "cer_vs_cpu": round(cer, 4),
                "wer_vs_cpu": round(wer, 4),
                "char_ned_vs_cpu": round(ned, 4),
                "kie_differences": {f: doc_kie_diff[f] for f in kie_fields if not doc_kie_diff[f]["match"]},
            })

    # Aggregated metrics from summary.json
    c_cond = cpu_summary["conditions"]["D0_S0_P0"]
    g_cond = gpu_summary["conditions"]["D0_S0_P0"]

    c_ocr = c_cond["ocr"]
    g_ocr = g_cond["ocr"]
    c_kie = c_cond["kie"]
    g_kie = g_cond["kie"]

    comparison = {
        "title": "B0 CPU vs B0 GPU Execution Provider Effect Analysis",
        "num_documents": 126,
        "split": "validation",
        "seed": 42,
        "runs": {
            "b0_cpu": {
                "experiment_id": cpu_summary.get("experiment_id"),
                "config_hash": cpu_summary.get("config_hash"),
                "execution_provider": "CPUExecutionProvider",
                "execution_duration_sec": cpu_summary.get("execution_duration_sec"),
            },
            "b0_gpu": {
                "experiment_id": gpu_summary.get("experiment_id"),
                "config_hash": gpu_summary.get("config_hash"),
                "execution_provider": "CUDAExecutionProvider",
                "execution_duration_sec": gpu_summary.get("execution_duration_sec"),
            },
        },
        "ocr_metrics_comparison": {
            "cer_normalized": {
                "cpu": c_ocr["cer_normalized"]["mean"],
                "gpu": g_ocr["cer_normalized"]["mean"],
                "delta": round(g_ocr["cer_normalized"]["mean"] - c_ocr["cer_normalized"]["mean"], 4),
            },
            "cer_raw": {
                "cpu": c_ocr["cer_raw"]["mean"],
                "gpu": g_ocr["cer_raw"]["mean"],
                "delta": round(g_ocr["cer_raw"]["mean"] - c_ocr["cer_raw"]["mean"], 4),
            },
            "wer_normalized": {
                "cpu": c_ocr["wer_normalized"]["mean"],
                "gpu": g_ocr["wer_normalized"]["mean"],
                "delta": round(g_ocr["wer_normalized"]["mean"] - c_ocr["wer_normalized"]["mean"], 4),
            },
            "wer_raw": {
                "cpu": c_ocr["wer_raw"]["mean"],
                "gpu": g_ocr["wer_raw"]["mean"],
                "delta": round(g_ocr["wer_raw"]["mean"] - c_ocr["wer_raw"]["mean"], 4),
            },
            "char_ned_normalized": {
                "cpu": c_ocr["char_ned_normalized"]["mean"],
                "gpu": g_ocr["char_ned_normalized"]["mean"],
                "delta": round(g_ocr["char_ned_normalized"]["mean"] - c_ocr["char_ned_normalized"]["mean"], 4),
            },
            "char_ned_raw": {
                "cpu": c_ocr["char_ned_raw"]["mean"],
                "gpu": g_ocr["char_ned_raw"]["mean"],
                "delta": round(g_ocr["char_ned_raw"]["mean"] - c_ocr["char_ned_raw"]["mean"], 4),
            },
        },
        "kie_metrics_comparison": {
            "macro_f1_normalized": {
                "cpu": c_kie["macro_f1_normalized"],
                "gpu": g_kie["macro_f1_normalized"],
                "delta": round(g_kie["macro_f1_normalized"] - c_kie["macro_f1_normalized"], 4),
            },
            "macro_precision_normalized": {
                "cpu": c_kie["macro_precision_normalized"],
                "gpu": g_kie["macro_precision_normalized"],
                "delta": round(g_kie["macro_precision_normalized"] - c_kie["macro_precision_normalized"], 4),
            },
            "macro_recall_normalized": {
                "cpu": c_kie["macro_recall_normalized"],
                "gpu": g_kie["macro_recall_normalized"],
                "delta": round(g_kie["macro_recall_normalized"] - c_kie["macro_recall_normalized"], 4),
            },
            "entity_hmean": {
                "cpu": c_kie["sroie_official_compatible"]["entity_hmean"],
                "gpu": g_kie["sroie_official_compatible"]["entity_hmean"],
                "delta": round(g_kie["sroie_official_compatible"]["entity_hmean"] - c_kie["sroie_official_compatible"]["entity_hmean"], 4),
            },
            "normalized_doc_em": {
                "cpu": c_kie["normalized_doc_em_rate"],
                "gpu": g_kie["normalized_doc_em_rate"],
                "delta": round(g_kie["normalized_doc_em_rate"] - c_kie["normalized_doc_em_rate"], 4),
            },
        },
        "text_level_direct_equivalence": {
            "exact_text_match_count": exact_text_matches,
            "exact_text_match_rate": round(exact_text_matches / 126.0, 4),
            "mean_direct_cer": round(float(np.mean(cer_vs_cpu_list)), 4),
            "median_direct_cer": round(float(np.median(cer_vs_cpu_list)), 4),
            "p95_direct_cer": round(float(np.percentile(cer_vs_cpu_list, 95)), 4),
            "mean_direct_wer": round(float(np.mean(wer_vs_cpu_list)), 4),
            "mean_direct_ned": round(float(np.mean(ned_vs_cpu_list)), 4),
            "token_count_diff_sum": int(np.sum(token_diff_list)),
        },
        "kie_field_level_agreement": {
            "company_match_rate": round(kie_matches["company"] / 126.0, 4),
            "date_match_rate": round(kie_matches["date"] / 126.0, 4),
            "address_match_rate": round(kie_matches["address"] / 126.0, 4),
            "total_match_rate": round(kie_matches["total"] / 126.0, 4),
            "overall_agreement_rate": round(sum(kie_matches.values()) / (126.0 * 4), 4),
            "field_differences": kie_diff_docs,
        },
        "number_of_documents_with_any_difference": len(per_doc_diffs),
        "per_document_differences": per_doc_diffs,
    }

    out_file = Path("experiments/benchmarks/b0_cpu_vs_gpu_comparison.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("B0 CPU vs B0 GPU COMPARISON SUMMARY (N=126)")
    print("=" * 80)
    print(f"Exact Normalized Text Match: {exact_text_matches}/126 ({exact_text_matches/126*100:.2f}%)")
    print(f"Direct Inter-Provider Mean CER: {np.mean(cer_vs_cpu_list):.4f} ({np.mean(cer_vs_cpu_list)*100:.2f}%)")
    print(f"Direct Inter-Provider Mean WER: {np.mean(wer_vs_cpu_list):.4f} ({np.mean(wer_vs_cpu_list)*100:.2f}%)")
    print(f"Direct Inter-Provider Mean NED: {np.mean(ned_vs_cpu_list):.4f} ({np.mean(ned_vs_cpu_list)*100:.2f}%)")
    print("-" * 80)
    print(f"CER (normalized):   CPU={c_ocr['cer_normalized']['mean']:.4f} | GPU={g_ocr['cer_normalized']['mean']:.4f} | Delta={g_ocr['cer_normalized']['mean'] - c_ocr['cer_normalized']['mean']:+.4f}")
    print(f"WER (normalized):   CPU={c_ocr['wer_normalized']['mean']:.4f} | GPU={g_ocr['wer_normalized']['mean']:.4f} | Delta={g_ocr['wer_normalized']['mean'] - c_ocr['wer_normalized']['mean']:+.4f}")
    print(f"NED (normalized):   CPU={c_ocr['char_ned_normalized']['mean']:.4f} | GPU={g_ocr['char_ned_normalized']['mean']:.4f} | Delta={g_ocr['char_ned_normalized']['mean'] - c_ocr['char_ned_normalized']['mean']:+.4f}")
    print(f"KIE Macro F1 (norm):CPU={c_kie['macro_f1_normalized']:.4f} | GPU={g_kie['macro_f1_normalized']:.4f} | Delta={g_kie['macro_f1_normalized'] - c_kie['macro_f1_normalized']:+.4f}")
    print(f"KIE Entity Hmean:   CPU={c_kie['sroie_official_compatible']['entity_hmean']:.4f} | GPU={g_kie['sroie_official_compatible']['entity_hmean']:.4f} | Delta={g_kie['sroie_official_compatible']['entity_hmean'] - c_kie['sroie_official_compatible']['entity_hmean']:+.4f}")
    print("-" * 80)
    print("Downstream KIE Field Agreement:")
    for f in kie_fields:
        print(f"  - {f:<10}: {kie_matches[f]}/126 matches ({kie_matches[f]/126*100:.2f}%)")
    print(f"Overall KIE Agreement: {sum(kie_matches.values())/(126*4)*100:.2f}%")
    print("=" * 80)
    print(f"Saved artifact to: {out_file.resolve()}")


if __name__ == "__main__":
    main()
