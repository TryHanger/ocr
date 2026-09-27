"""Audit B2 artifacts, generate detailed conditions CSV, and print statistical summary."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np


def main() -> None:
    run_dir = Path("experiments/runs/b2_preprocessing_validation_n126_gpu")
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"
    jsonl_path = run_dir / "per_document.jsonl"

    print("=== AUDITING B2 RUN ARTIFACTS ===")
    assert summary_path.is_file(), f"Missing {summary_path}"
    assert manifest_path.is_file(), f"Missing {manifest_path}"
    assert jsonl_path.is_file(), f"Missing {jsonl_path}"

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 1. Total document counts & failure checks
    doc_counts = summary["document_counts"]
    print(f"Total requested:  {doc_counts['total_requested']}")
    print(f"Total successful: {doc_counts['total_successful']}")
    print(f"Total failed:     {doc_counts['total_failed']}")
    print(f"Failure rate:     {doc_counts['failure_rate']}")
    assert doc_counts["total_requested"] == 16254
    assert doc_counts["total_successful"] == 16254
    assert doc_counts["total_failed"] == 0
    assert doc_counts["failure_rate"] == 0.0

    # 2. JSONL line count
    with open(jsonl_path, "r", encoding="utf-8") as f:
        jsonl_count = sum(1 for _ in f)
    print(f"per_document.jsonl line count: {jsonl_count}")
    assert jsonl_count == 16254

    # 3. Number of conditions
    conditions = summary["conditions"]
    print(f"Number of conditions evaluated: {len(conditions)}")
    assert len(conditions) == 129

    # 4. Strict Control Verification (D0_S0_P0 vs B0_GPU)
    c0 = conditions["D0_S0_P0"]
    delta_b0 = c0.get("delta_from_b0", {})
    print("\n--- Control Condition Verification (D0_S0_P0 vs B0_GPU) ---")
    for k, v in delta_b0.items():
        if k != "b0_reference_provenance":
            print(f"  {k}: {v}")
            assert v == 0.0, f"Control mismatch for {k}: {v} != 0.0"
    print(">> Control D0_S0_P0 matches B0_GPU with EXACT ZERO DELTA (PASS)")

    # 5. Recovery coverage check
    print("\n--- Checking Recovery from B1 Reference ---")
    b2_conditions = [cid for cid, c in conditions.items() if c["condition"]["baseline_type"] == "B2"]
    assert len(b2_conditions) == 128
    missing_recovery = []
    for cid in b2_conditions:
        rec = conditions[cid].get("recovery_from_b1")
        if not rec or "delta_cer_normalized" not in rec:
            missing_recovery.append(cid)
    assert not missing_recovery, f"Missing recovery in: {missing_recovery}"
    print(f">> All 128 B2 conditions have verified recovery deltas from B1 reference (PASS)")

    # 6. Export Detailed CSV
    csv_path = run_dir / "b2_conditions_detailed.csv"
    csv_rows = []
    fieldnames = [
        "condition_id",
        "baseline_type",
        "degradation",
        "severity",
        "preprocessing",
        "sample_size",
        "cer_norm_mean",
        "cer_norm_ci_lower",
        "cer_norm_ci_upper",
        "wer_norm_mean",
        "ned_norm_mean",
        "macro_f1_norm_mean",
        "macro_f1_ci_lower",
        "macro_f1_ci_upper",
        "entity_hmean",
        "doc_em_norm",
        "delta_cer_from_b0",
        "delta_wer_from_b0",
        "delta_ned_from_b0",
        "delta_macro_f1_from_b0",
        "delta_entity_hmean_from_b0",
        "delta_doc_em_from_b0",
        "ref_b1_condition_id",
        "delta_cer_recovery_b1",
        "delta_wer_recovery_b1",
        "delta_ned_recovery_b1",
        "delta_macro_f1_recovery_b1",
        "delta_entity_hmean_recovery_b1",
        "delta_doc_em_recovery_b1",
    ]

    for cid in sorted(conditions.keys()):
        c = conditions[cid]
        c_info = c["condition"]
        ocr = c.get("ocr", {})
        kie = c.get("kie", {})
        d_b0 = c.get("delta_from_b0", {})
        rec = c.get("recovery_from_b1", {})

        cer_ci = ocr.get("cer_normalized", {}).get("ci_95", [None, None])
        f1_ci = kie.get("macro_f1_ci_95", [None, None])

        row = {
            "condition_id": cid,
            "baseline_type": c_info.get("baseline_type"),
            "degradation": c_info.get("degradation_type"),
            "severity": c_info.get("severity"),
            "preprocessing": c_info.get("preprocessing_id"),
            "sample_size": c.get("document_counts", {}).get("successful"),
            "cer_norm_mean": ocr.get("cer_normalized", {}).get("mean"),
            "cer_norm_ci_lower": cer_ci[0],
            "cer_norm_ci_upper": cer_ci[1],
            "wer_norm_mean": ocr.get("wer_normalized", {}).get("mean"),
            "ned_norm_mean": ocr.get("char_ned_normalized", {}).get("mean"),
            "macro_f1_norm_mean": kie.get("macro_f1_normalized"),
            "macro_f1_ci_lower": f1_ci[0],
            "macro_f1_ci_upper": f1_ci[1],
            "entity_hmean": kie.get("sroie_official_compatible", {}).get("entity_hmean"),
            "doc_em_norm": kie.get("normalized_doc_em_rate"),
            "delta_cer_from_b0": d_b0.get("delta_cer_normalized"),
            "delta_wer_from_b0": d_b0.get("delta_wer_normalized"),
            "delta_ned_from_b0": d_b0.get("delta_char_ned_normalized"),
            "delta_macro_f1_from_b0": d_b0.get("delta_macro_f1_normalized"),
            "delta_entity_hmean_from_b0": d_b0.get("delta_entity_hmean"),
            "delta_doc_em_from_b0": d_b0.get("delta_normalized_doc_em"),
            "ref_b1_condition_id": rec.get("reference_b1_condition_id"),
            "delta_cer_recovery_b1": rec.get("delta_cer_normalized"),
            "delta_wer_recovery_b1": rec.get("delta_wer_normalized"),
            "delta_ned_recovery_b1": rec.get("delta_char_ned_normalized"),
            "delta_macro_f1_recovery_b1": rec.get("delta_macro_f1_normalized"),
            "delta_entity_hmean_recovery_b1": rec.get("delta_entity_hmean"),
            "delta_doc_em_recovery_b1": rec.get("delta_normalized_doc_em"),
        }
        csv_rows.append(row)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n>> Detailed CSV exported to: {csv_path}")

    # 7. Statistical Aggregations across Pipelines
    print("\n=== PREPROCESSING PIPELINE PERFORMANCE SUMMARY ===")
    pipelines = [
        "p_minimal_cleanup",
        "p_contrast_enhancement",
        "p_standard_receipt_enhancement",
        "p_aggressive_binarization",
    ]

    for p in pipelines:
        p_rows = [r for r in csv_rows if r["preprocessing"] == p]
        d_cer_rec = [r["delta_cer_recovery_b1"] for r in p_rows]
        d_wer_rec = [r["delta_wer_recovery_b1"] for r in p_rows]
        d_f1_rec = [r["delta_macro_f1_recovery_b1"] for r in p_rows]
        d_hm_rec = [r["delta_entity_hmean_recovery_b1"] for r in p_rows]

        cer_improved = sum(1 for d in d_cer_rec if d < 0)
        cer_degraded = sum(1 for d in d_cer_rec if d > 0)
        cer_unchanged = sum(1 for d in d_cer_rec if d == 0)

        f1_improved = sum(1 for d in d_f1_rec if d > 0)
        f1_degraded = sum(1 for d in d_f1_rec if d < 0)
        f1_unchanged = sum(1 for d in d_f1_rec if d == 0)

        print(f"\nPipeline: {p} (N={len(p_rows)} conditions)")
        print(f"  Mean Delta CER Recovery:  {np.mean(d_cer_rec):+.4f} (median: {np.median(d_cer_rec):+.4f})")
        print(f"  CER: Improved {cer_improved}/32, Degraded {cer_degraded}/32, Unchanged {cer_unchanged}/32")
        print(f"  Mean Delta Macro F1 Recovery: {np.mean(d_f1_rec):+.4f} (median: {np.median(d_f1_rec):+.4f})")
        print(f"  F1:  Improved {f1_improved}/32, Degraded {f1_degraded}/32, Unchanged {f1_unchanged}/32")
        print(f"  Mean Delta Entity Hmean:      {np.mean(d_hm_rec):+.4f}")

    # 8. Check specific targeted hypotheses
    print("\n=== TARGETED DOMAIN HYPOTHESES CHECK ===")
    # D6 (Rotation) + p_standard_receipt_enhancement (deskew)
    rot_standard = [r for r in csv_rows if r["degradation"] == "rotation" and r["preprocessing"] == "p_standard_receipt_enhancement"]
    print("D6 (Rotation) with p_standard_receipt_enhancement (deskew):")
    for r in rot_standard:
        print(f"  S{r['severity']}: CER B1->B2: dCER_rec={r['delta_cer_recovery_b1']:+.4f}, dF1_rec={r['delta_macro_f1_recovery_b1']:+.4f} (CER_norm: {r['cer_norm_mean']:.4f})")

    # D8 (Shadow) + p_contrast_enhancement (clahe)
    shadow_clahe = [r for r in csv_rows if r["degradation"] == "shadow" and r["preprocessing"] == "p_contrast_enhancement"]
    print("\nD8 (Shadow) with p_contrast_enhancement (clahe):")
    for r in shadow_clahe:
        print(f"  S{r['severity']}: CER B1->B2: dCER_rec={r['delta_cer_recovery_b1']:+.4f}, dF1_rec={r['delta_macro_f1_recovery_b1']:+.4f} (CER_norm: {r['cer_norm_mean']:.4f})")

    # Aggressive binarization across degradations
    bin_rows = [r for r in csv_rows if r["preprocessing"] == "p_aggressive_binarization"]
    d_cer_bin = [r["delta_cer_recovery_b1"] for r in bin_rows]
    print(f"\np_aggressive_binarization overall effect: mean dCER_rec = {np.mean(d_cer_bin):+.4f} (positive = worsens CER)")

    # 9. Degradation x Pipeline Recovery Delta Table
    from collections import defaultdict
    deg_pipe_cer = defaultdict(dict)
    deg_pipe_f1 = defaultdict(dict)
    for r in csv_rows:
        if r["baseline_type"] == "B2":
            d = r["degradation"]
            p = r["preprocessing"]
            deg_pipe_cer[d].setdefault(p, []).append(float(r["delta_cer_recovery_b1"]))
            deg_pipe_f1[d].setdefault(p, []).append(float(r["delta_macro_f1_recovery_b1"]))

    print("\n=== MEAN DELTA CER RECOVERY BY DEGRADATION AND PIPELINE ===")
    hdr = f"{'Degradation':<20} | {'p_minimal':<11} | {'p_contrast':<11} | {'p_standard':<11} | {'p_binarization':<14}"
    print(hdr)
    print("-" * len(hdr))
    for d in sorted(deg_pipe_cer.keys()):
        min_v = np.mean(deg_pipe_cer[d]["p_minimal_cleanup"])
        con_v = np.mean(deg_pipe_cer[d]["p_contrast_enhancement"])
        std_v = np.mean(deg_pipe_cer[d]["p_standard_receipt_enhancement"])
        bin_v = np.mean(deg_pipe_cer[d]["p_aggressive_binarization"])
        print(f"{d:<20} | {min_v:+11.4f} | {con_v:+11.4f} | {std_v:+11.4f} | {bin_v:+14.4f}")

    print("\n=== MEAN DELTA MACRO F1 RECOVERY BY DEGRADATION AND PIPELINE ===")
    print(hdr)
    print("-" * len(hdr))
    for d in sorted(deg_pipe_f1.keys()):
        min_v = np.mean(deg_pipe_f1[d]["p_minimal_cleanup"])
        con_v = np.mean(deg_pipe_f1[d]["p_contrast_enhancement"])
        std_v = np.mean(deg_pipe_f1[d]["p_standard_receipt_enhancement"])
        bin_v = np.mean(deg_pipe_f1[d]["p_aggressive_binarization"])
        print(f"{d:<20} | {min_v:+11.4f} | {con_v:+11.4f} | {std_v:+11.4f} | {bin_v:+14.4f}")



if __name__ == "__main__":
    main()
