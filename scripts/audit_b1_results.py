"""
Audit and Analysis Script for TASK-014 (B1 Degradation Baseline on GPU)
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    run_dir = Path("experiments/runs/b1_degraded_validation_n126_gpu")
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"
    per_doc_path = run_dir / "per_document.jsonl"

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print("=" * 60)
    print("1. RUN INTEGRITY AUDIT")
    print("=" * 60)
    print(f"Experiment ID:      {manifest.get('experiment_id')}")
    print(f"Config Hash:        {manifest.get('config_hash')}")
    print(f"Git Commit:         {manifest.get('git_commit')}")
    print(f"Dataset Root:       {manifest.get('dataset_root')}")
    print(f"Split:              {manifest.get('split')}")
    print(f"Conditions Count:   {manifest.get('num_conditions')}")
    print(f"Total Evaluations:  {manifest.get('total_evaluations')}")
    print(f"Successful:         {manifest.get('total_successful')}")
    print(f"Failed:             {manifest.get('total_failed')}")
    print(f"Failure Rate:       {manifest.get('failure_rate'):.2%}")
    print(f"Execution Provider: {manifest.get('ocr_engine', {}).get('execution_provider')}")
    print(f"ONNX Providers:     {manifest.get('ocr_engine', {}).get('onnxruntime_providers')}")

    # Check per_document lines
    with open(per_doc_path, "r", encoding="utf-8") as f:
        doc_count = sum(1 for _ in f)
    print(f"per_document.jsonl: {doc_count} lines (Expected: 4158)")
    assert doc_count == 4158, f"Mismatch in per_document.jsonl: {doc_count} != 4158"

    conditions = summary["conditions"]

    # Control condition
    print("\n" + "=" * 60)
    print("2. CONTROL CONDITION VERIFICATION (D0_S0_P0 vs B0_GPU)")
    print("=" * 60)
    control = conditions.get("D0_S0_P0", {})
    ctrl_ocr = control["ocr"]
    ctrl_kie = control["kie"]
    ctrl_cer = ctrl_ocr["cer_normalized"]["mean"]
    ctrl_wer = ctrl_ocr["wer_normalized"]["mean"]
    ctrl_ned = ctrl_ocr["char_ned_normalized"]["mean"]
    ctrl_f1 = ctrl_kie["macro_f1_normalized"]
    ctrl_hmean = ctrl_kie["sroie_official_compatible"]["entity_hmean"]

    delta = control.get("delta_from_b0", {})
    print(f"CER (norm):      {ctrl_cer:.4f} (Delta: {delta.get('delta_cer_normalized', 0.0):+.6f})")
    print(f"WER (norm):      {ctrl_wer:.4f} (Delta: {delta.get('delta_wer_normalized', 0.0):+.6f})")
    print(f"Char-NED (norm): {ctrl_ned:.4f} (Delta: {delta.get('delta_char_ned_normalized', 0.0):+.6f})")
    print(f"Macro F1:        {ctrl_f1:.4f} (Delta: {delta.get('delta_macro_f1_normalized', 0.0):+.6f})")
    print(f"Entity Hmean:    {ctrl_hmean:.4f} (Delta: {delta.get('delta_entity_hmean', 0.0):+.6f})")

    deg_names = {
        "D1": "Gaussian Blur",
        "D2": "Motion Blur",
        "D3": "Gaussian Noise",
        "D4": "JPEG Compression",
        "D5": "Downsampling",
        "D6": "Rotation",
        "D7": "Perspective",
        "D8": "Shadow"
    }

    rows = []
    for cond_id, cond_data in conditions.items():
        if cond_id == "D0_S0_P0":
            continue
        parts = cond_id.split("_")
        deg = parts[0]
        sev = parts[1]
        
        ocr = cond_data["ocr"]
        kie = cond_data["kie"]
        d = cond_data.get("delta_from_b0", {})

        cer = ocr["cer_normalized"]["mean"]
        cer_ci = ocr["cer_normalized"]["ci_95"]
        d_cer = d.get("delta_cer_normalized", 0.0)
        
        wer = ocr["wer_normalized"]["mean"]
        d_wer = d.get("delta_wer_normalized", 0.0)

        ned = ocr["char_ned_normalized"]["mean"]
        d_ned = d.get("delta_char_ned_normalized", 0.0)
        
        f1 = kie["macro_f1_normalized"]
        f1_ci = kie.get("macro_f1_ci_95", [0.0, 0.0])
        d_f1 = d.get("delta_macro_f1_normalized", 0.0)

        hmean = kie["sroie_official_compatible"]["entity_hmean"]
        d_hmean = d.get("delta_entity_hmean", 0.0)

        pf = kie.get("per_field", {})
        f_comp = pf.get("company", {}).get("f1_normalized", 0.0)
        f_date = pf.get("date", {}).get("f1_normalized", 0.0)
        f_addr = pf.get("address", {}).get("f1_normalized", 0.0)
        f_tot = pf.get("total", {}).get("f1_normalized", 0.0)

        rows.append({
            "cond": cond_id,
            "deg": deg,
            "name": deg_names.get(deg, deg),
            "sev": sev,
            "cer": cer,
            "cer_ci_lower": cer_ci[0],
            "cer_ci_upper": cer_ci[1],
            "d_cer": d_cer,
            "wer": wer,
            "d_wer": d_wer,
            "ned": ned,
            "d_ned": d_ned,
            "f1": f1,
            "f1_ci_lower": f1_ci[0],
            "f1_ci_upper": f1_ci[1],
            "d_f1": d_f1,
            "hmean": hmean,
            "d_hmean": d_hmean,
            "f_comp": f_comp,
            "f_date": f_date,
            "f_addr": f_addr,
            "f_tot": f_tot
        })

    df = pd.DataFrame(rows)

    print("\n" + "=" * 60)
    print("3. FULL 32 DEGRADATION CONDITIONS TABLE")
    print("=" * 60)
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 1000)
    print(df[["cond", "name", "sev", "cer", "d_cer", "wer", "ned", "f1", "d_f1", "hmean", "d_hmean"]].to_string(index=False))

    print("\n" + "=" * 60)
    print("4. MONOTONICITY ANALYSIS BY DEGRADATION")
    print("=" * 60)
    for deg in sorted(df["deg"].unique()):
        sub = df[df["deg"] == deg].sort_values("sev")
        cer_vals = [ctrl_cer] + sub["cer"].tolist()
        f1_vals = [ctrl_f1] + sub["f1"].tolist()
        cer_mono = all(x <= y for x, y in zip(cer_vals, cer_vals[1:]))
        f1_mono = all(x >= y for x, y in zip(f1_vals, f1_vals[1:]))
        print(f"[{deg}] {deg_names[deg]:<18} | CER: {' -> '.join(f'{v:.4f}' for v in cer_vals)} (Monotonic incr: {cer_mono})")
        print(f"     {' ':<18} | F1 : {' -> '.join(f'{v:.4f}' for v in f1_vals)} (Monotonic decr: {f1_mono})")

    print("\n" + "=" * 60)
    print("5. S4 DESTRUCTIVENESS RANKING (WORST TO BEST)")
    print("=" * 60)
    s4 = df[df["sev"] == "S4"].sort_values(by="d_cer", ascending=False)
    print("\n--- By OCR Degradation (Delta CER descending) ---")
    print(s4[["deg", "name", "cer", "d_cer", "ned", "f1", "d_f1", "hmean"]].to_string(index=False))

    s4_f1 = df[df["sev"] == "S4"].sort_values(by="d_f1", ascending=True)
    print("\n--- By KIE Degradation (Delta Macro F1 ascending) ---")
    print(s4_f1[["deg", "name", "cer", "d_cer", "f1", "d_f1", "hmean", "d_hmean"]].to_string(index=False))

    print("\n" + "=" * 60)
    print("6. KIE ENTITY SENSITIVITY BREAKDOWN AT S4")
    print("=" * 60)
    print(s4[["deg", "name", "f_comp", "f_date", "f_addr", "f_tot"]].to_string(index=False))

    # Save summary table as CSV and JSON for reporting
    df.to_csv(run_dir / "b1_conditions_detailed.csv", index=False)
    print(f"\nDetailed CSV exported to: {run_dir / 'b1_conditions_detailed.csv'}")

if __name__ == "__main__":
    main()
