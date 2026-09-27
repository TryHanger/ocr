"""GPU Feasibility Benchmark: RapidOCR / ONNX Runtime CPU vs CUDA.

Evaluates latency, throughput, downstream KIE equivalence, and VRAM utilization
on a fixed 10-document subset of the SROIE validation split (N=126, seed=42).
Strictly adheres to safety guards (test split prohibited).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.sroie import SROIEAdapter
from src.evaluation.ocr_metrics import (
    compute_cer,
    compute_character_ned_similarity,
    compute_wer,
    normalize_ocr_text,
)
from src.experiments.split import resolve_split_documents
from src.kie.rule_based import RuleBasedKIEEngine
from src.ocr.rapid_ocr import RapidOCREngine


def query_nvidia_smi() -> Dict[str, Any]:
    """Query current GPU VRAM and utilization via nvidia-smi."""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,nounits,noheader",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        parts = [p.strip() for p in res.stdout.strip().split(",")]
        return {
            "available": True,
            "gpu_name": parts[0],
            "total_mb": float(parts[1]),
            "used_mb": float(parts[2]),
            "free_mb": float(parts[3]),
            "utilization_pct": float(parts[4]),
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def compute_stats(values: List[float]) -> Dict[str, float]:
    """Compute summary statistics for a list of floats."""
    arr = np.array(values, dtype=np.float64)
    return {
        "mean_ms": round(float(np.mean(arr)), 2),
        "median_ms": round(float(np.median(arr)), 2),
        "std_ms": round(float(np.std(arr)), 2),
        "min_ms": round(float(np.min(arr)), 2),
        "max_ms": round(float(np.max(arr)), 2),
        "p90_ms": round(float(np.percentile(arr, 90)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
    }


def run_benchmark(
    data_root: str = "data/SROIE2019",
    output_dir: str = "experiments/benchmarks/ocr_cpu_vs_gpu",
    num_docs: int = 10,
    num_warmup: int = 2,
) -> Dict[str, Any]:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GPU FEASIBILITY BENCHMARK: RapidOCR / ONNX Runtime (CPU vs CUDA)")
    print("=" * 80)

    # 1. Dataset Resolution (Validation Split Only)
    adapter = SROIEAdapter(data_root)
    all_val_ids, resolved_split, data_status = resolve_split_documents(
        adapter=adapter,
        split="validation",
        subset_size=None,
        subset_seed=42,
        allow_test=False,
        allow_fixture=False,
    )
    assert resolved_split == "validation", f"Safety violation: resolved split is {resolved_split}"
    print(f"Dataset: SROIE ({data_status}), Validation split size: {len(all_val_ids)}")

    selected_doc_ids = all_val_ids[:num_docs]
    warmup_doc_ids = all_val_ids[num_docs : num_docs + num_warmup]
    print(f"Benchmark subset (N={num_docs}): {selected_doc_ids}")
    print(f"Warmup subset (N={num_warmup}): {warmup_doc_ids}")

    # Preload images to isolate OCR inference time from disk I/O
    print("\nPreloading images into memory...")
    benchmark_images: Dict[str, np.ndarray] = {}
    for did in selected_doc_ids:
        benchmark_images[did] = adapter.get_image(did)

    warmup_images: Dict[str, np.ndarray] = {}
    for did in warmup_doc_ids:
        warmup_images[did] = adapter.get_image(did)

    # KIE Engine
    kie_engine = RuleBasedKIEEngine()

    # Query initial GPU status
    gpu_initial = query_nvidia_smi()
    print(f"Initial GPU Status: {gpu_initial}")

    # -------------------------------------------------------------
    # CPU BENCHMARK
    # -------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Starting CPU Benchmark (provider='cpu')...")
    t0_cpu_init = time.perf_counter()
    eng_cpu = RapidOCREngine(execution_provider="cpu")
    cpu_init_sec = time.perf_counter() - t0_cpu_init
    print(f"CPU RapidOCR Session Init Time: {cpu_init_sec:.3f} s")

    # Verify CPU providers
    cpu_manifest = eng_cpu.get_model_manifest()
    cpu_providers = cpu_manifest["resolved"]["detector"]["providers"]
    print(f"CPU Active Providers: {cpu_providers}")
    assert cpu_providers[0] == "CPUExecutionProvider", f"Unexpected CPU provider: {cpu_providers}"

    # CPU Warm-up
    print(f"Running {num_warmup} CPU warm-up inferences...")
    for wid in warmup_doc_ids:
        _ = eng_cpu.recognize(warmup_images[wid], wid)

    # CPU Timed Runs
    print(f"Running {num_docs} CPU timed evaluations...")
    cpu_ocr_latencies: List[float] = []
    cpu_kie_latencies: List[float] = []
    cpu_total_latencies: List[float] = []
    cpu_ocr_results: Dict[str, Any] = {}
    cpu_kie_results: Dict[str, Any] = {}

    for i, did in enumerate(selected_doc_ids, start=1):
        img = benchmark_images[did]
        t_start_ocr = time.perf_counter()
        ocr_res = eng_cpu.recognize(img, did)
        t_ocr = (time.perf_counter() - t_start_ocr) * 1000.0

        t_start_kie = time.perf_counter()
        kie_res = kie_engine.extract(ocr_res, did)
        t_kie = (time.perf_counter() - t_start_kie) * 1000.0

        t_total = t_ocr + t_kie
        cpu_ocr_latencies.append(t_ocr)
        cpu_kie_latencies.append(t_kie)
        cpu_total_latencies.append(t_total)

        cpu_ocr_results[did] = ocr_res
        cpu_kie_results[did] = kie_res

        print(f"  [CPU {i:02d}/{num_docs}] Doc: {did:<15} OCR: {t_ocr:7.2f} ms | KIE: {t_kie:5.2f} ms | Total: {t_total:7.2f} ms | Tokens: {len(ocr_res.tokens)}")

    # -------------------------------------------------------------
    # GPU BENCHMARK
    # -------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Starting GPU Benchmark (provider='cuda')...")
    gpu_before_init = query_nvidia_smi()

    t0_gpu_init = time.perf_counter()
    eng_gpu = RapidOCREngine(execution_provider="cuda")
    gpu_init_sec = time.perf_counter() - t0_gpu_init
    print(f"GPU RapidOCR Session Init Time: {gpu_init_sec:.3f} s")

    gpu_after_init = query_nvidia_smi()

    # Verify GPU providers
    gpu_manifest = eng_gpu.get_model_manifest()
    det_gpu_providers = gpu_manifest["resolved"]["detector"]["providers"]
    rec_gpu_providers = gpu_manifest["resolved"]["recognizer"]["providers"]
    cls_gpu_providers = gpu_manifest["resolved"]["classifier"]["providers"]
    print(f"GPU Active Providers - Det: {det_gpu_providers}, Rec: {rec_gpu_providers}, Cls: {cls_gpu_providers}")
    assert det_gpu_providers[0] == "CUDAExecutionProvider", f"Detector not on CUDA: {det_gpu_providers}"
    assert rec_gpu_providers[0] == "CUDAExecutionProvider", f"Recognizer not on CUDA: {rec_gpu_providers}"

    # GPU Warm-up
    print(f"Running {num_warmup} GPU warm-up inferences...")
    for wid in warmup_doc_ids:
        _ = eng_gpu.recognize(warmup_images[wid], wid)

    gpu_after_warmup = query_nvidia_smi()

    # GPU Timed Runs
    print(f"Running {num_docs} GPU timed evaluations...")
    gpu_ocr_latencies: List[float] = []
    gpu_kie_latencies: List[float] = []
    gpu_total_latencies: List[float] = []
    gpu_ocr_results: Dict[str, Any] = {}
    gpu_kie_results: Dict[str, Any] = {}
    gpu_vram_snapshots: List[Dict[str, Any]] = []

    for i, did in enumerate(selected_doc_ids, start=1):
        img = benchmark_images[did]
        t_start_ocr = time.perf_counter()
        ocr_res = eng_gpu.recognize(img, did)
        t_ocr = (time.perf_counter() - t_start_ocr) * 1000.0

        t_start_kie = time.perf_counter()
        kie_res = kie_engine.extract(ocr_res, did)
        t_kie = (time.perf_counter() - t_start_kie) * 1000.0

        t_total = t_ocr + t_kie
        gpu_ocr_latencies.append(t_ocr)
        gpu_kie_latencies.append(t_kie)
        gpu_total_latencies.append(t_total)

        gpu_ocr_results[did] = ocr_res
        gpu_kie_results[did] = kie_res

        vram = query_nvidia_smi()
        gpu_vram_snapshots.append(vram)

        print(f"  [GPU {i:02d}/{num_docs}] Doc: {did:<15} OCR: {t_ocr:7.2f} ms | KIE: {t_kie:5.2f} ms | Total: {t_total:7.2f} ms | Tokens: {len(ocr_res.tokens)} | VRAM: {vram.get('used_mb', 0):.0f} MB")

    # -------------------------------------------------------------
    # EQUIVALENCE & COMPARISON ANALYSIS
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ANALYSIS: CPU vs GPU EQUIVALENCE & SPEEDUP")
    print("=" * 80)

    per_doc_records: List[Dict[str, Any]] = []
    cer_list: List[float] = []
    wer_list: List[float] = []
    char_ned_list: List[float] = []
    exact_text_matches = 0
    token_diffs: List[int] = []

    kie_fields = ["company", "date", "address", "total"]
    kie_field_matches = {f: 0 for f in kie_fields}
    kie_field_total = {f: 0 for f in kie_fields}

    for idx, did in enumerate(selected_doc_ids):
        c_ocr = cpu_ocr_results[did]
        g_ocr = gpu_ocr_results[did]

        c_text_raw = c_ocr.full_text
        g_text_raw = g_ocr.full_text

        c_text_norm = normalize_ocr_text(c_text_raw)
        g_text_norm = normalize_ocr_text(g_text_raw)

        exact_match = (c_text_norm == g_text_norm)
        if exact_match:
            exact_text_matches += 1

        cer = compute_cer(g_text_norm, c_text_norm)
        wer = compute_wer(g_text_norm, c_text_norm)
        ned = compute_character_ned_similarity(g_text_norm, c_text_norm)
        tok_diff = len(g_ocr.tokens) - len(c_ocr.tokens)

        cer_list.append(cer)
        wer_list.append(wer)
        char_ned_list.append(ned)
        token_diffs.append(tok_diff)

        # KIE comparison
        c_kie = cpu_kie_results[did]
        g_kie = gpu_kie_results[did]

        c_preds = c_kie.predictions if hasattr(c_kie, "predictions") else {}
        g_preds = g_kie.predictions if hasattr(g_kie, "predictions") else {}

        kie_diff: Dict[str, Any] = {}
        for f in kie_fields:
            c_val = normalize_ocr_text(c_preds.get(f, {}).get("value", "") if isinstance(c_preds.get(f), dict) else str(c_preds.get(f) or ""))
            g_val = normalize_ocr_text(g_preds.get(f, {}).get("value", "") if isinstance(g_preds.get(f), dict) else str(g_preds.get(f) or ""))
            is_match = (c_val == g_val)
            if is_match:
                kie_field_matches[f] += 1
            kie_field_total[f] += 1
            kie_diff[f] = {
                "cpu_value": c_val,
                "gpu_value": g_val,
                "match": is_match,
            }

        rec = {
            "document_id": did,
            "cpu_ocr_latency_ms": round(cpu_ocr_latencies[idx], 2),
            "gpu_ocr_latency_ms": round(gpu_ocr_latencies[idx], 2),
            "speedup_ocr": round(cpu_ocr_latencies[idx] / max(gpu_ocr_latencies[idx], 0.001), 2),
            "cpu_total_latency_ms": round(cpu_total_latencies[idx], 2),
            "gpu_total_latency_ms": round(gpu_total_latencies[idx], 2),
            "speedup_total": round(cpu_total_latencies[idx] / max(gpu_total_latencies[idx], 0.001), 2),
            "cpu_tokens_count": len(c_ocr.tokens),
            "gpu_tokens_count": len(g_ocr.tokens),
            "token_count_diff": tok_diff,
            "exact_text_match": exact_match,
            "cer_vs_cpu": round(cer, 4),
            "wer_vs_cpu": round(wer, 4),
            "char_ned_vs_cpu": round(ned, 4),
            "kie_comparison": kie_diff,
            "gpu_vram_used_mb": gpu_vram_snapshots[idx].get("used_mb", 0),
        }
        per_doc_records.append(rec)

    # Statistical Aggregation
    cpu_ocr_stats = compute_stats(cpu_ocr_latencies)
    gpu_ocr_stats = compute_stats(gpu_ocr_latencies)
    cpu_total_stats = compute_stats(cpu_total_latencies)
    gpu_total_stats = compute_stats(gpu_total_latencies)

    mean_ocr_speedup = round(cpu_ocr_stats["mean_ms"] / max(gpu_ocr_stats["mean_ms"], 0.001), 2)
    median_ocr_speedup = round(cpu_ocr_stats["median_ms"] / max(gpu_ocr_stats["median_ms"], 0.001), 2)
    mean_total_speedup = round(cpu_total_stats["mean_ms"] / max(gpu_total_stats["mean_ms"], 0.001), 2)

    avg_cer = round(float(np.mean(cer_list)), 4)
    avg_wer = round(float(np.mean(wer_list)), 4)
    avg_ned = round(float(np.mean(char_ned_list)), 4)
    exact_text_match_rate = round(exact_text_matches / num_docs, 4)

    kie_field_accuracy = {f: round(kie_field_matches[f] / max(kie_field_total[f], 1), 4) for f in kie_fields}
    kie_overall_match_rate = round(sum(kie_field_matches.values()) / max(sum(kie_field_total.values()), 1), 4)

    # VRAM stats
    vram_used_values = [v.get("used_mb", 0) for v in gpu_vram_snapshots if v.get("used_mb") is not None]
    peak_vram_mb = max(vram_used_values) if vram_used_values else 0.0
    baseline_vram_mb = gpu_initial.get("used_mb", 0.0)
    net_vram_allocated_mb = peak_vram_mb - baseline_vram_mb

    # Full B1 projections
    # B1: 33 conditions * 126 docs = 4158 evaluations
    total_evals_b1 = 33 * 126
    b1_cpu_hours = round((total_evals_b1 * (cpu_total_stats["mean_ms"] / 1000.0)) / 3600.0, 2)
    b1_gpu_hours = round((total_evals_b1 * (gpu_total_stats["mean_ms"] / 1000.0)) / 3600.0, 2)
    b1_hours_saved = round(b1_cpu_hours - b1_gpu_hours, 2)

    results = {
        "benchmark_metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_split": "validation",
            "num_benchmark_documents": num_docs,
            "num_warmup_documents": num_warmup,
            "document_ids": selected_doc_ids,
            "seed": 42,
        },
        "session_initialization": {
            "cpu_init_seconds": round(cpu_init_sec, 3),
            "gpu_init_seconds": round(gpu_init_sec, 3),
        },
        "latency_statistics": {
            "cpu_ocr": cpu_ocr_stats,
            "gpu_ocr": gpu_ocr_stats,
            "cpu_total": cpu_total_stats,
            "gpu_total": gpu_total_stats,
        },
        "speedup": {
            "ocr_mean_speedup_x": mean_ocr_speedup,
            "ocr_median_speedup_x": median_ocr_speedup,
            "total_mean_speedup_x": mean_total_speedup,
        },
        "equivalence": {
            "exact_text_match_rate": exact_text_match_rate,
            "mean_cer_vs_cpu": avg_cer,
            "mean_wer_vs_cpu": avg_wer,
            "mean_char_ned_similarity": avg_ned,
            "token_count_differences": token_diffs,
            "kie_field_match_rates": kie_field_accuracy,
            "kie_overall_match_rate": kie_overall_match_rate,
        },
        "hardware_monitoring": {
            "gpu_name": gpu_initial.get("gpu_name", "Unknown"),
            "gpu_total_vram_mb": gpu_initial.get("total_mb", 4096.0),
            "gpu_initial_used_mb": baseline_vram_mb,
            "gpu_peak_used_mb": peak_vram_mb,
            "gpu_net_allocated_mb": net_vram_allocated_mb,
            "vram_headroom_mb": round(gpu_initial.get("total_mb", 4096.0) - peak_vram_mb, 1),
            "vram_sufficient_4gb": (gpu_initial.get("total_mb", 4096.0) - peak_vram_mb) > 500,
        },
        "projections_b1": {
            "total_evaluations": total_evals_b1,
            "cpu_estimated_hours": b1_cpu_hours,
            "gpu_estimated_hours": b1_gpu_hours,
            "estimated_hours_saved": b1_hours_saved,
        },
    }

    # Environment
    env_info = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "cuda_device": gpu_initial.get("gpu_name", "NVIDIA GPU"),
        "total_vram_mb": gpu_initial.get("total_mb", 4096.0),
        "onnxruntime_providers": cpu_providers,
        "gpu_providers": det_gpu_providers,
        "models": gpu_manifest.get("configured", {}).get("models", {}),
    }

    # Config used
    benchmark_config = {
        "dataset_root": str(data_root),
        "split": "validation",
        "num_documents": num_docs,
        "num_warmup": num_warmup,
        "ocr_engine": "rapidocr",
        "reading_order": {"line_tolerance_factor": 0.5},
        "kie_engine": "rule_based",
        "tested_providers": ["cpu", "cuda"],
    }

    # Write artifacts
    print("\nWriting benchmark artifacts...")
    with open(out_path / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(benchmark_config, f, default_flow_style=False)

    with open(out_path / "environment.json", "w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=2)

    with open(out_path / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(out_path / "per_document.jsonl", "w", encoding="utf-8") as f:
        for rec in per_doc_records:
            f.write(json.dumps(rec) + "\n")

    print(f"Artifacts successfully saved to: {out_path.resolve()}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RapidOCR CPU vs CUDA benchmark")
    parser.add_argument("--data-root", default="data/SROIE2019", help="Path to SROIE dataset")
    parser.add_argument("--output-dir", default="experiments/benchmarks/ocr_cpu_vs_gpu", help="Output directory")
    parser.add_argument("--num-docs", type=int, default=10, help="Number of benchmark documents")
    parser.add_argument("--num-warmup", type=int, default=2, help="Number of warmup documents")
    args = parser.parse_args()

    run_benchmark(
        data_root=args.data_root,
        output_dir=args.output_dir,
        num_docs=args.num_docs,
        num_warmup=args.num_warmup,
    )
