"""SROIE Downstream KIE Baseline and Evaluation Runner.

Baseline Pipeline:
    SROIE Image -> RapidOCR -> OCRResult -> RuleBasedKIE -> KIEResult -> Evaluation

Execution Order Audit (Strict GT Isolation):
    1. ocr_inference: Generate OCRResult from raw image
    2. kie_inference: Extract KIEResult using ONLY OCRResult and document_id (no image, no GT)
    3. gt_loading: Load KIEGroundTruth from dataset adapter
    4. evaluation: Compute independent raw/normalized analytical metrics and SROIE Task-3 Hmean
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional
import yaml

# Ensure project root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.datasets.sroie import SROIEAdapter
from src.evaluation.kie_metrics import KIEEvaluator, evaluate_kie_corpus
from src.kie.factory import get_kie_engine
from src.ocr.factory import get_ocr_engine


def run_kie_baseline(
    kie_config_path: Path,
    ocr_config_path: Path,
    output_dir: Path,
    data_root: Optional[Path] = None,
    split: str = "test",
    limit: Optional[int] = None,
    kie_engine_name: str = "rule_based",
    ocr_engine_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute end-to-end KIE baseline pipeline with strict execution audit."""
    execution_order_trace: List[str] = []

    # 1. Load Configurations
    with open(kie_config_path, "r", encoding="utf-8") as f:
        kie_cfg: Dict[str, Any] = yaml.safe_load(f) or {}

    ocr_cfg: Dict[str, Any] = {}
    if ocr_config_path.exists():
        with open(ocr_config_path, "r", encoding="utf-8") as f:
            ocr_cfg = yaml.safe_load(f) or {}

    # 2. Determine Dataset Root
    if data_root is not None:
        actual_root = Path(data_root)
        train_img_dir = actual_root / "train" / "img"
        if train_img_dir.is_dir() and len(list(train_img_dir.glob("*.jpg"))) >= 100:
            data_status = "COMPLETED"
            dataset_type = "real_sroie"
        else:
            data_status = "PENDING_REAL_DATA"
            dataset_type = "synthetic_fixture"
    else:
        candidate_roots = [
            Path("data/SROIE2019"),
            Path("data"),
        ]
        actual_root = None
        data_status = "PENDING_REAL_DATA"
        dataset_type = "synthetic_fixture"

        for cand in candidate_roots:
            if cand and (cand / "train" / "img").is_dir() and len(list((cand / "train" / "img").glob("*.jpg"))) >= 100:
                actual_root = cand
                data_status = "COMPLETED"
                dataset_type = "real_sroie"
                break

        if actual_root is None:
            fixture_path = Path("tests/fixtures/sroie_valid")
            if fixture_path.is_dir() and (fixture_path / "train" / "img").is_dir():
                actual_root = fixture_path
            else:
                actual_root = Path("tests/fixtures/sroie")

    adapter = SROIEAdapter(actual_root)
    available_splits = [s for s in adapter.SUPPORTED_SPLITS if (adapter.root_dir / s).exists()]
    target_split = split if split in available_splits else (available_splits[0] if available_splits else "train")
    doc_ids = adapter.list_document_ids(target_split)

    if limit is not None and limit > 0:
        doc_ids = doc_ids[:limit]

    # 3. Instantiate OCR and KIE Engines
    # OCR Engine
    if ocr_engine_name is None:
        ocr_engine_name = ocr_cfg.get("engine", {}).get("name", "rapidocr")

    reading_order_cfg = ocr_cfg.get("reading_order", {})
    if ocr_engine_name in ("mock", "mock_ocr"):
        ocr_params = {
            "line_tolerance_factor": reading_order_cfg.get("line_tolerance_factor", 0.5),
        }
    else:
        ocr_params = {
            "line_tolerance_factor": reading_order_cfg.get("line_tolerance_factor", 0.5),
            "text_score": ocr_cfg.get("engine", {}).get("parameters", {}).get("text_score", 0.5),
            "min_confidence": ocr_cfg.get("engine", {}).get("parameters", {}).get("min_confidence", 0.0),
            "use_cls": ocr_cfg.get("engine", {}).get("parameters", {}).get("use_cls", True),
        }
        if "max_side_len" in resize_cfg:
            ocr_params["max_side_len"] = resize_cfg["max_side_len"]
        if "min_side_len" in resize_cfg:
            ocr_params["min_side_len"] = resize_cfg["min_side_len"]
        if "det_limit_side_len" in resize_cfg:
            ocr_params["det_limit_side_len"] = resize_cfg["det_limit_side_len"]
        if "det_limit_type" in resize_cfg:
            ocr_params["det_limit_type"] = resize_cfg["det_limit_type"]

    ocr_engine = get_ocr_engine(ocr_engine_name, ocr_params)

    # KIE Engine
    kie_engine = get_kie_engine(kie_engine_name, config=kie_cfg)

    # Evaluator
    target_fields = kie_cfg.get("evaluation", {}).get(
        "target_fields", ["company", "date", "address", "total"]
    )
    evaluator = KIEEvaluator(target_fields=target_fields)

    # 4. Pipeline Execution with Strict Audit Sequence
    # Sequence: ocr_inference -> kie_inference -> gt_loading -> evaluation
    execution_order_trace = ["ocr_inference", "kie_inference", "gt_loading", "evaluation"]

    predictions: List[Any] = []
    ground_truths: List[Any] = []
    doc_traces: List[Dict[str, Any]] = []

    for doc_id in doc_ids:
        # Step 4.1: OCR Inference
        image = adapter.get_image(doc_id)
        t_ocr_start = time.perf_counter()
        ocr_result = ocr_engine.recognize(image, doc_id)
        t_ocr_end = time.perf_counter()

        # Step 4.2: KIE Inference (Strict GT Isolation: only ocr_result and doc_id)
        t_kie_start = time.perf_counter()
        kie_result = kie_engine.extract(ocr_result=ocr_result, document_id=doc_id)
        t_kie_end = time.perf_counter()

        # Step 4.3: GT Loading (Strictly AFTER inference)
        t_gt_start = time.perf_counter()
        kie_gt = adapter.get_kie_ground_truth(doc_id)
        t_gt_end = time.perf_counter()

        # Step 4.4: Document-Level Evaluation
        doc_eval = evaluator.evaluate_kie(kie_result, kie_gt)

        predictions.append(kie_result)
        ground_truths.append(kie_gt)

        doc_traces.append({
            "document_id": doc_id,
            "phases_completed": ["ocr_inference", "kie_inference", "gt_loading", "evaluation"],
            "ocr_time_ms": round((t_ocr_end - t_ocr_start) * 1000.0, 2),
            "kie_time_ms": round((t_kie_end - t_kie_start) * 1000.0, 2),
            "gt_load_time_ms": round((t_gt_end - t_gt_start) * 1000.0, 2),
            "fields_extracted": list(kie_result.fields.keys()),
            "evaluation": doc_eval,
        })

    # 5. Corpus Evaluation
    corpus_eval = evaluate_kie_corpus(
        predictions=predictions,
        ground_truths=ground_truths,
        target_fields=target_fields,
    )

    # 6. Build Final Report
    report: Dict[str, Any] = {
        "title": "SROIE Downstream KIE Baseline Evaluation Report",
        "timestamp": datetime.now().isoformat(),
        "baseline_pipeline": "B0: Image -> RapidOCR -> OCRResult -> RuleBasedKIE -> Evaluation",
        "status": data_status,
        "dataset_type": dataset_type,
        "data_root": actual_root.as_posix(),
        "split": target_split,
        "num_documents": len(doc_ids),
        "execution_order_audit": {
            "enforced_phases": execution_order_trace,
            "gt_leakage_prevented": True,
            "kie_api_contract": "extract(ocr_result: OCRResult, document_id: str) -> KIEResult",
        },
        "engine_configurations": {
            "ocr_engine": {
                "name": ocr_engine_name,
                "reading_order_tolerance": reading_order_cfg.get("line_tolerance_factor", 0.5),
            },
            "kie_engine": {
                "name": kie_engine_name,
                "model_name": getattr(kie_engine, "model_name", "rule_based"),
                "model_version": getattr(kie_engine, "model_version", "1.0.0"),
                "fallback_largest_amount": kie_cfg.get("engine", {}).get("fallback_largest_amount", False),
                "fallback_penalty": kie_cfg.get("engine", {}).get("fallback_penalty", 0.5),
                "min_confidence_threshold": getattr(kie_engine, "min_confidence", 0.2),
            },
        },
        "notes": (
            "KIE BASELINE: PENDING_REAL_DATA. Baseline ran on local synthetic fixtures for "
            "pipeline and GT isolation verification. Official ICDAR SROIE numbers are not claimed."
            if data_status == "PENDING_REAL_DATA"
            else "Real SROIE benchmark KIE evaluation completed."
        ),
        "analytical_metrics": corpus_eval["analytical_metrics"],
        "sroie_official_compatible": corpus_eval["sroie_official_compatible"],
        "documents": doc_traces,
    }

    # 7. Write Output Report
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "kie_baseline_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SROIE Downstream KIE Baseline Evaluation")
    parser.add_argument("--kie-config", type=Path, default=Path("configs/kie.yaml"), help="KIE config path")
    parser.add_argument("--ocr-config", type=Path, default=Path("configs/ocr.yaml"), help="OCR config path")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/runs"), help="Output directory")
    parser.add_argument("--data-root", type=Path, default=None, help="Path to SROIE dataset")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents")
    parser.add_argument("--kie-engine", type=str, default="rule_based", help="KIE engine name")
    parser.add_argument("--ocr-engine", type=str, default=None, help="OCR engine name")
    args = parser.parse_args()

    report = run_kie_baseline(
        kie_config_path=args.kie_config,
        ocr_config_path=args.ocr_config,
        output_dir=args.output_dir,
        data_root=args.data_root,
        split=args.split,
        limit=args.limit,
        kie_engine_name=args.kie_engine,
        ocr_engine_name=args.ocr_engine,
    )

    print(f"KIE Baseline evaluation completed. Status: {report['status']}")
    print(f"Report written to: {args.output_dir / 'kie_baseline_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
