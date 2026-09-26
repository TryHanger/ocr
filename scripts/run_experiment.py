"""Unified Experiment Runner CLI for SROIE Robustness Matrix across B0, B1, and B2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

# Ensure project root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.experiments.config import load_experiment_config
from src.experiments.runner import UnifiedExperimentRunner


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unified Experiment Runner for SROIE Robustness Matrix",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment.yaml"),
        help="Path to YAML experiment configuration file",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split to evaluate ('development', 'validation', 'test')",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Number of documents to sample from split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Experiment seed for deterministic sampling and operations",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Base output directory for experiment runs",
    )
    parser.add_argument(
        "--allow-test",
        action="store_true",
        default=False,
        help="Explicit safety override required to execute on the frozen 'test' split",
    )
    parser.add_argument(
        "--allow-fixture",
        action="store_true",
        default=False,
        help="Explicit permission to run on local synthetic fixtures if real SROIE data is absent",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        default=False,
        help="Save processed/degraded images to disk (debug mode)",
    )
    parser.add_argument(
        "--smoke-run",
        action="store_true",
        default=False,
        help="Execute minimal CPU smoke matrix (D0_S0_P0, D1_S1/S4_P0, D1_S1/S4_P_clahe)",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Explicit experiment identifier slug",
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_experiment_config(args.config)
    except Exception as e:
        print(f"ERROR: Failed to load config '{args.config}': {e}", file=sys.stderr)
        return 1

    # Apply CLI overrides
    if args.split:
        config.setdefault("dataset", {})["split"] = args.split
    if args.subset_size is not None:
        config.setdefault("dataset", {})["subset_size"] = args.subset_size
    if args.seed is not None:
        config["experiment_seed"] = args.seed
        config.setdefault("dataset", {})["subset_seed"] = args.seed
    if args.save_images:
        config.setdefault("output", {})["save_images"] = True

    try:
        runner = UnifiedExperimentRunner(
            config=config,
            output_dir=args.output_dir,
            smoke_mode=args.smoke_run,
            allow_test=args.allow_test,
            allow_fixture=args.allow_fixture,
            save_images=args.save_images or config.get("output", {}).get("save_images", False),
            experiment_id=args.experiment_id,
        )
        summary = runner.run()
    except ValueError as ve:
        print(f"VALIDATION ERROR: {ve}", file=sys.stderr)
        return 2
    except FileNotFoundError as fnfe:
        print(f"DATA RESOLUTION ERROR: {fnfe}", file=sys.stderr)
        return 3
    except Exception as e:
        print(f"RUNNER EXECUTION ERROR: {e}", file=sys.stderr)
        return 4

    # Print Console Summary
    doc_counts = summary.get("document_counts", {})
    conditions = summary.get("conditions", {})

    print("\n" + "=" * 60)
    print("EXPERIMENT RUN COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Experiment ID:     {summary.get('experiment_id')}")
    print(f"Config Hash:       {summary.get('config_hash')[:12]}...")
    print(f"Data Status:       {summary.get('data_status')}")
    print(f"Split:             {summary.get('split')}")
    print(f"Smoke Mode:        {summary.get('smoke_mode')}")
    print(f"Conditions Count:  {len(conditions)}")
    print(f"Total Requested:   {doc_counts.get('total_requested')}")
    print(f"Total Successful:  {doc_counts.get('total_successful')}")
    print(f"Total Failed:      {doc_counts.get('total_failed')}")
    print(f"Failure Rate:      {doc_counts.get('failure_rate', 0.0) * 100:.2f}%")
    print(f"Execution Time:    {summary.get('execution_duration_sec', 0.0):.2f}s")
    print(f"Artifacts Dir:     {runner.run_dir}")
    print("=" * 60)

    # Output condition breakdown
    print("\nEvaluated Conditions Summary:")
    for cid, c_data in sorted(conditions.items()):
        ocr_s = c_data.get("ocr", {})
        kie_s = c_data.get("kie", {})
        cer_m = ocr_s.get("cer_normalized", {}).get("mean", 0.0)
        ned_m = ocr_s.get("char_ned_normalized", {}).get("mean", 0.0)
        macro_f1 = kie_s.get("macro_f1_normalized", 0.0)
        norm_em = kie_s.get("normalized_doc_em_rate", 0.0)
        req = c_data.get("document_counts", {}).get("requested", 0)
        succ = c_data.get("document_counts", {}).get("successful", 0)
        print(
            f"  [{cid:16s}] Docs: {succ}/{req} | "
            f"CER_norm: {cer_m:.4f} | NED_norm: {ned_m:.4f} | "
            f"Macro_F1: {macro_f1:.4f} | Norm_EM: {norm_em:.4f}"
        )
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
