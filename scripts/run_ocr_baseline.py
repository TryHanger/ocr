"""SROIE OCR Baseline and End-to-End Evaluation Runner.

Pipeline:
    SROIEAdapter -> Image -> RapidOCREngine -> OCRResult -> OCREvaluation -> CER/WER/Character-NED
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import yaml

# Ensure project root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.datasets.sroie import SROIEAdapter
from src.evaluation.ocr_metrics import evaluate_ocr
from src.ocr.factory import get_ocr_engine


def run_ocr_baseline(
    config_path: Path,
    output_dir: Path,
    data_root: Optional[Path] = None,
    split: str = "test",
) -> Dict[str, Any]:
    """Execute OCR baseline evaluation across available documents."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. Determine data source
    if data_root is not None:
        actual_root = Path(data_root)
        train_img_dir = actual_root / "train" / "img"
        if train_img_dir.is_dir() and len(list(train_img_dir.glob("*.jpg"))) >= 100:
            data_status = "COMPLETED"
            dataset_type = "real_sroie"
        else:
            data_status = "PENDING"
            dataset_type = "synthetic_fixture"
    else:
        candidate_real = [
            Path("data/SROIE2019"),
            Path("data"),
        ]
        actual_root = None
        data_status = "PENDING"
        dataset_type = "synthetic_fixture"

        for p in candidate_real:
            if p and (p / "train" / "img").is_dir() and len(list((p / "train" / "img").glob("*.jpg"))) >= 100:
                actual_root = p
                data_status = "COMPLETED"
                dataset_type = "real_sroie"
                break

        if actual_root is None:
            # Fallback to local valid fixture for pipeline verification
            fixture_path = Path("tests/fixtures/sroie_valid")
            if fixture_path.is_dir():
                actual_root = fixture_path
            else:
                actual_root = Path("tests/fixtures/sroie")

    # 2. Instantiate SROIE Adapter
    adapter = SROIEAdapter(actual_root)
    available_splits = [s for s in adapter.SUPPORTED_SPLITS if (adapter.root_dir / s).exists()]
    target_split = split if split in available_splits else (available_splits[0] if available_splits else "train")
    doc_ids = adapter.list_document_ids(target_split)

    # 3. Instantiate OCR Engine
    engine_cfg = cfg.get("engine", {})
    reading_order_cfg = cfg.get("reading_order", {})
    resize_cfg = cfg.get("resize_policy", {})
    engine_name = engine_cfg.get("name", "rapidocr")

    engine_params: Dict[str, Any] = {
        "line_tolerance_factor": reading_order_cfg.get("line_tolerance_factor", 0.5),
        "text_score": engine_cfg.get("parameters", {}).get("text_score", 0.5),
        "min_confidence": engine_cfg.get("parameters", {}).get("min_confidence", 0.0),
        "use_cls": engine_cfg.get("parameters", {}).get("use_cls", True),
    }
    if "max_side_len" in resize_cfg:
        engine_params["max_side_len"] = resize_cfg["max_side_len"]
    if "min_side_len" in resize_cfg:
        engine_params["min_side_len"] = resize_cfg["min_side_len"]
    if "det_limit_side_len" in resize_cfg:
        engine_params["det_limit_side_len"] = resize_cfg["det_limit_side_len"]
    if "det_limit_type" in resize_cfg:
        engine_params["det_limit_type"] = resize_cfg["det_limit_type"]

    ocr_engine = get_ocr_engine(engine_name, engine_params)

    # Verify model stack and export manifest if supported
    manifest = None
    if hasattr(ocr_engine, "verify_model_stack"):
        is_valid, errors = ocr_engine.verify_model_stack()
        if not is_valid:
            error_msg = f"OCR Model Stack Verification Failed: {errors}"
            print(f"ERROR: {error_msg}", file=sys.stderr)
            raise RuntimeError(error_msg)
    if hasattr(ocr_engine, "get_model_manifest"):
        manifest = ocr_engine.get_model_manifest()
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "ocr_stack_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    # 4. Run Evaluation
    results_list: List[Dict[str, Any]] = []
    cer_raw_list: List[float] = []
    wer_raw_list: List[float] = []
    char_ned_raw_list: List[float] = []
    cer_norm_list: List[float] = []
    wer_norm_list: List[float] = []
    char_ned_norm_list: List[float] = []

    for doc_id in doc_ids:
        # Load image and ground truth via SROIE adapter
        image = adapter.get_image(doc_id)
        ocr_gt = adapter.get_ocr_ground_truth(doc_id)

        # Primary OCR invocation: receives ONLY image and document_id (Strict GT Isolation)
        ocr_result = ocr_engine.recognize(image, doc_id)

        # Evaluate against GT full text
        eval_res = evaluate_ocr(pred=ocr_result.full_text, gt=ocr_gt.text)

        cer_raw_list.append(eval_res.cer_raw)
        wer_raw_list.append(eval_res.wer_raw)
        char_ned_raw_list.append(eval_res.char_ned_raw)
        cer_norm_list.append(eval_res.cer_normalized)
        wer_norm_list.append(eval_res.wer_normalized)
        char_ned_norm_list.append(eval_res.char_ned_normalized)

        doc_summary = {
            "document_id": doc_id,
            "processing_time_ms": ocr_result.processing_time_ms,
            "tokens_count": len(ocr_result.tokens),
            "metrics": eval_res.to_dict(),
        }
        results_list.append(doc_summary)

    num_docs = len(results_list)
    mean_metrics = {
        "mean_cer_raw": round(sum(cer_raw_list) / max(1, num_docs), 4),
        "mean_wer_raw": round(sum(wer_raw_list) / max(1, num_docs), 4),
        "mean_char_ned_raw": round(sum(char_ned_raw_list) / max(1, num_docs), 4),
        "mean_cer_normalized": round(sum(cer_norm_list) / max(1, num_docs), 4),
        "mean_wer_normalized": round(sum(wer_norm_list) / max(1, num_docs), 4),
        "mean_char_ned_normalized": round(sum(char_ned_norm_list) / max(1, num_docs), 4),
    }

    report = {
        "title": "OCR Baseline Evaluation Report",
        "timestamp": datetime.now().isoformat(),
        "real_data_ocr_baseline": data_status,
        "dataset_type": dataset_type,
        "data_root": actual_root.as_posix(),
        "split": target_split,
        "num_documents": num_docs,
        "notes": (
            "REAL-DATA OCR BASELINE: PENDING. Evaluation was executed on local synthetic fixtures "
            "for pipeline verification. Real SROIE benchmark numbers are not claimed."
            if data_status == "PENDING"
            else "Real SROIE benchmark evaluation completed."
        ),
        "engine": {
            "name": engine_name,
            "version": engine_cfg.get("package_version", "3.9.2"),
            "models": engine_cfg.get("models", {}),
            "reading_order_tolerance": reading_order_cfg.get("line_tolerance_factor", 0.5),
            "manifest_file": "ocr_stack_manifest.json" if manifest else None,
        },
        "ocr_stack": manifest if manifest else {},
        "resize_policy": manifest.get("resize_policy", resize_cfg) if manifest else resize_cfg,
        "summary_metrics": mean_metrics,
        "documents": results_list,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "ocr_baseline_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SROIE OCR Baseline and Evaluation")
    parser.add_argument("--config", type=Path, default=Path("configs/ocr.yaml"), help="Config file path")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/runs"), help="Output directory")
    parser.add_argument("--data-root", type=Path, default=None, help="Root path to SROIE dataset")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate")
    args = parser.parse_args()

    report = run_ocr_baseline(
        config_path=args.config,
        output_dir=args.output_dir,
        data_root=args.data_root,
        split=args.split,
    )

    print(f"OCR Baseline finished. Status: {report['real_data_ocr_baseline']}")
    print(f"Report written to: {(args.output_dir / 'ocr_baseline_report.json').as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
