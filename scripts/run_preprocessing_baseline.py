"""CLI and runner for B0/B1/B2 Preprocessing Baselines evaluation.

Protocol:
- B0: Original document -> OCR -> Metrics
- B1: Degraded document -> OCR -> Metrics
- B2: Degraded document -> Preprocessing -> OCR -> Metrics
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import yaml

# Ensure project root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.core.schemas import DegradationSpec, PreprocessingSpec
from src.core.seed import set_seed
from src.datasets.sroie import SROIEAdapter
from src.ocr.factory import get_ocr_engine
from src.preprocessing.baseline import BaselineComparisonResult, run_b0_b1_b2_comparison

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_preprocessing_baseline(
    config_ocr_path: Path,
    config_prep_path: Path,
    output_dir: Path,
    data_root: Optional[Path] = None,
    split: str = "test",
    degradation_type: str = "gaussian_blur",
    severity: int = 2,
    preprocessing_preset: str = "standard_receipt_enhancement",
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute B0/B1/B2 comparison on available documents."""
    set_seed(seed)

    # 1. Load Configurations
    with open(config_ocr_path, "r", encoding="utf-8") as f:
        ocr_cfg = yaml.safe_load(f)
    with open(config_prep_path, "r", encoding="utf-8") as f:
        prep_cfg = yaml.safe_load(f)

    # 2. Determine data source (Real vs Fixture)
    candidate_real = [
        data_root if data_root else None,
        Path("data/SROIE2019"),
        Path("data"),
    ]
    actual_root: Optional[Path] = None
    data_status = "PENDING"
    dataset_type = "synthetic_fixture"

    for p in candidate_real:
        if p and (p / "train" / "img").is_dir() and len(list((p / "train" / "img").glob("*.jpg"))) >= 100:
            actual_root = p
            data_status = "COMPLETED"
            dataset_type = "real_sroie"
            break

    if actual_root is None:
        fixture_path = Path("tests/fixtures/sroie_valid")
        if fixture_path.is_dir():
            actual_root = fixture_path
        else:
            actual_root = Path("tests/fixtures/sroie")

    # 3. Instantiate Adapter
    adapter = SROIEAdapter(actual_root)
    available_splits = [s for s in adapter.SUPPORTED_SPLITS if (adapter.root_dir / s).exists()]
    target_split = split if split in available_splits else (available_splits[0] if available_splits else "train")
    doc_ids = adapter.list_document_ids(target_split)

    # 4. Instantiate OCR Engine
    engine_cfg = ocr_cfg.get("engine", {})
    reading_order_cfg = ocr_cfg.get("reading_order", {})
    ocr_engine = get_ocr_engine(
        engine_cfg.get("name", "rapidocr"),
        {
            "line_tolerance_factor": reading_order_cfg.get("line_tolerance_factor", 0.5),
            "text_score": engine_cfg.get("parameters", {}).get("text_score", 0.5),
            "min_confidence": engine_cfg.get("parameters", {}).get("min_confidence", 0.0),
        },
    )

    # 5. Build Specs
    deg_spec = DegradationSpec(
        type=degradation_type,
        severity=severity,
        seed=seed,
    )

    preset_dict = prep_cfg.get("pipelines", {}).get(preprocessing_preset, {})
    prep_methods = preset_dict.get("methods", ["grayscale", "clahe", "denoise"])
    prep_params = preset_dict.get("parameters", {})
    prep_spec = PreprocessingSpec(
        enabled=True,
        methods=prep_methods,
        parameters=prep_params,
    )

    # 6. Execute Run
    comparison_results: List[BaselineComparisonResult] = []
    for doc_id in doc_ids:
        image = adapter.get_image(doc_id)
        gt = adapter.get_ocr_ground_truth(doc_id)

        res = run_b0_b1_b2_comparison(
            image=image,
            document_id=doc_id,
            gt_text=gt.text,
            degradation_spec=deg_spec,
            preprocessing_spec=prep_spec,
            ocr_engine=ocr_engine,
        )
        comparison_results.append(res)

    # 7. Aggregate Metrics
    n = len(comparison_results)
    summary: Dict[str, Any] = {
        "count": n,
        "mean_b0_cer_raw": float(sum(r.b0.metrics["cer_raw"] for r in comparison_results) / n) if n else 0.0,
        "mean_b0_char_ned_normalized": float(sum(r.b0.metrics["char_ned_normalized"] for r in comparison_results) / n) if n else 0.0,
        "mean_b1_cer_raw": float(sum(r.b1.metrics["cer_raw"] for r in comparison_results) / n) if n else 0.0,
        "mean_b1_char_ned_normalized": float(sum(r.b1.metrics["char_ned_normalized"] for r in comparison_results) / n) if n else 0.0,
        "mean_b2_cer_raw": float(sum(r.b2.metrics["cer_raw"] for r in comparison_results) / n) if n else 0.0,
        "mean_b2_char_ned_normalized": float(sum(r.b2.metrics["char_ned_normalized"] for r in comparison_results) / n) if n else 0.0,
    }

    report: Dict[str, Any] = {
        "title": "B0/B1/B2 Preprocessing Baseline Evaluation Report",
        "timestamp": datetime.now().isoformat(),
        "real_data_preprocessing_baseline": data_status,
        "dataset_type": dataset_type,
        "data_root": str(actual_root).replace("\\", "/"),
        "split": target_split,
        "num_documents": n,
        "degradation": deg_spec.to_dict(),
        "preprocessing_pipeline": {
            "preset": preprocessing_preset,
            "spec": prep_spec.to_dict(),
        },
        "notes": (
            "REAL-DATA PREPROCESSING BASELINE: PENDING. Evaluation was executed on local synthetic fixtures "
            "for pipeline verification. Real SROIE benchmark numbers are not claimed."
            if data_status == "PENDING"
            else "Evaluated on canonical SROIE data."
        ),
        "summary_metrics": summary,
        "documents": [r.to_dict() for r in comparison_results],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "preprocessing_baseline_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("Report written to: %s", report_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run B0/B1/B2 Preprocessing Baselines.")
    parser.add_argument("--config-ocr", type=str, default="configs/ocr.yaml", help="Path to OCR config")
    parser.add_argument("--config-prep", type=str, default="configs/preprocessing.yaml", help="Path to Preprocessing config")
    parser.add_argument("--output-dir", type=str, default="experiments/runs", help="Output directory")
    parser.add_argument("--data-root", type=str, default=None, help="Root path of SROIE dataset")
    parser.add_argument("--split", type=str, default="test", help="Split to evaluate")
    parser.add_argument("--degradation", type=str, default="gaussian_blur", help="Degradation type")
    parser.add_argument("--severity", type=int, default=2, help="Degradation severity 0-4")
    parser.add_argument("--preset", type=str, default="standard_receipt_enhancement", help="Preprocessing preset name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    report = run_preprocessing_baseline(
        config_ocr_path=Path(args.config_ocr),
        config_prep_path=Path(args.config_prep),
        output_dir=Path(args.output_dir),
        data_root=Path(args.data_root) if args.data_root else None,
        split=args.split,
        degradation_type=args.degradation,
        severity=args.severity,
        preprocessing_preset=args.preset,
        seed=args.seed,
    )

    print(f"B0/B1/B2 Preprocessing Baseline finished. Status: {report['real_data_preprocessing_baseline']}")
    print(f"Summary: B0 CER={report['summary_metrics']['mean_b0_cer_raw']:.3f} | B1 CER={report['summary_metrics']['mean_b1_cer_raw']:.3f} | B2 CER={report['summary_metrics']['mean_b2_cer_raw']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
