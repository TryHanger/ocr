"""Unified Experiment Runner for SROIE Robustness Matrix across B0, B1, and B2."""

from __future__ import annotations

from datetime import datetime
import json
import logging
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import yaml

from src.core.schemas import DegradationSpec, PreprocessingSpec
from src.datasets.sroie import SROIEAdapter
from src.degradation import get_degradation
from src.evaluation.kie_metrics import KIEEvaluator
from src.experiments.bootstrap import aggregate_condition_records
from src.experiments.conditions import ExperimentCondition, build_conditions_matrix
from src.experiments.config import compute_config_hash, load_experiment_config
from src.experiments.seed import derive_seed
from src.experiments.split import resolve_dataset_root, resolve_split_documents
from src.kie.factory import get_kie_engine
from src.ocr.factory import get_ocr_engine
from src.preprocessing import get_preprocessor


def get_git_commit_sha() -> str:
    """Retrieve current git commit SHA, falling back gracefully if git unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=5.0,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "UNKNOWN_COMMIT"


class UnifiedExperimentRunner:
    """Sole entry point for executing the primary experimental matrix.

    Guarantees:
    - Same OCR engine and KIE engine instance configurations across all conditions.
    - Strict execution sequence: image -> degradation -> preprocessing -> OCR -> KIE -> GT -> evaluation.
    - Zero GT leakage: GT is never accessible during inference.
    - Deterministic per-document seed derivation.
    - Explicit failure accounting (no silent dropping).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: Optional[Path | str] = None,
        smoke_mode: bool = False,
        allow_test: bool = False,
        allow_fixture: bool = False,
        save_images: bool = False,
        experiment_id: Optional[str] = None,
    ) -> None:
        self.config = config
        self.smoke_mode = smoke_mode
        self.allow_test = allow_test
        self.allow_fixture = allow_fixture
        self.save_images = save_images

        # Experiment Identity
        self.experiment_seed = int(self.config.get("experiment_seed", 42))
        self.config_hash = compute_config_hash(self.config)
        self.git_commit = get_git_commit_sha()

        timestamp_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_id:
            self.experiment_id = experiment_id
        else:
            mode_prefix = "smoke" if self.smoke_mode else "exp"
            self.experiment_id = f"{mode_prefix}_{timestamp_slug}_{self.config_hash[:8]}"

        base_out = Path(output_dir or self.config.get("output", {}).get("output_dir", "experiments/runs"))
        self.run_dir = (base_out / self.experiment_id).resolve()
        self.logs_dir = self.run_dir / "logs"
        self.images_dir = self.run_dir / "images"

        # Initialize Logger
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        if self.save_images:
            self.images_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(f"runner_{self.experiment_id}")
        self.logger.setLevel(logging.INFO)
        log_file = self.logs_dir / "run.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
        self.logger.addHandler(file_handler)

    def _setup_engines(self) -> Tuple[Any, Any, Any, SROIEAdapter]:
        """Instantiate single OCR engine, KIE engine, and dataset adapter for the entire run."""
        # 1. Dataset Adapter
        data_cfg = self.config.get("dataset", {})
        candidate_roots = data_cfg.get("candidate_roots", [
            data_cfg.get("root"),
            Path("data/SROIE2019"),
            Path("data"),
        ])
        resolved_root, detected_status = resolve_dataset_root(candidate_roots)
        adapter = SROIEAdapter(resolved_root)

        # 2. OCR Engine
        ocr_cfg = self.config.get("ocr", {})
        ocr_engine_name = ocr_cfg.get("engine_name", "rapidocr")
        reading_order_cfg = ocr_cfg.get("reading_order", {})
        resize_cfg = ocr_cfg.get("resize_policy", {})

        if ocr_engine_name in ("mock", "mock_ocr"):
            ocr_params = {"line_tolerance_factor": reading_order_cfg.get("line_tolerance_factor", 0.5)}
        else:
            ocr_params = {
                "line_tolerance_factor": reading_order_cfg.get("line_tolerance_factor", 0.5),
                "text_score": ocr_cfg.get("parameters", {}).get("text_score", 0.5),
                "min_confidence": ocr_cfg.get("parameters", {}).get("min_confidence", 0.0),
                "use_cls": ocr_cfg.get("parameters", {}).get("use_cls", True),
            }
            if "max_side_len" in resize_cfg:
                ocr_params["max_side_len"] = resize_cfg["max_side_len"]
            if "min_side_len" in resize_cfg:
                ocr_params["min_side_len"] = resize_cfg["min_side_len"]
            if "det_limit_type" in resize_cfg:
                ocr_params["det_limit_type"] = resize_cfg["det_limit_type"]

        self.ocr_params = ocr_params
        ocr_engine = get_ocr_engine(ocr_engine_name, ocr_params)

        # 3. KIE Engine
        kie_cfg = self.config.get("kie", {})
        kie_engine_name = kie_cfg.get("engine_name", "rule_based")
        kie_engine = get_kie_engine(kie_engine_name, config=kie_cfg)

        # 4. Evaluator
        eval_cfg = self.config.get("evaluation", {})
        target_fields = eval_cfg.get("target_fields", ["company", "date", "address", "total"])
        evaluator = KIEEvaluator(target_fields=target_fields)

        return ocr_engine, kie_engine, evaluator, adapter

    def run(self) -> Dict[str, Any]:
        """Execute the configured experimental matrix."""
        start_time_iso = datetime.now().isoformat()
        t_global_start = time.perf_counter()

        self.logger.info("Initializing Unified Experiment Runner: %s", self.experiment_id)
        self.logger.info("Config Hash: %s, Git Commit: %s", self.config_hash, self.git_commit)

        # 1. Setup Engines and Adapter
        ocr_engine, kie_engine, evaluator, adapter = self._setup_engines()

        # 2. Resolve Split & Documents
        split_name = self.config.get("dataset", {}).get("split", "development")
        subset_size = self.config.get("dataset", {}).get("subset_size")
        subset_seed = self.config.get("dataset", {}).get("subset_seed", 42)

        doc_ids, resolved_split, data_status = resolve_split_documents(
            adapter=adapter,
            split=split_name,
            subset_size=subset_size,
            subset_seed=subset_seed,
            allow_test=self.allow_test,
            allow_fixture=self.allow_fixture,
        )

        self.logger.info(
            "Resolved split '%s' with %d documents (Data Status: %s)",
            resolved_split,
            len(doc_ids),
            data_status,
        )

        # 3. Build Condition Matrix
        conditions = build_conditions_matrix(self.config, smoke_mode=self.smoke_mode)
        self.logger.info("Generated %d experimental conditions to evaluate.", len(conditions))

        # 4. Prepare Preprocessing Specs Cache
        prep_pipelines_cfg = self.config.get("matrix", {}).get("preprocessing_pipelines", {})

        # 5. Open Artifact Files
        jsonl_path = self.run_dir / "per_document.jsonl"
        per_condition_records: Dict[str, List[Dict[str, Any]]] = {c.condition_id: [] for c in conditions}

        total_runs = len(conditions) * len(doc_ids)
        run_count = 0

        with open(jsonl_path, "w", encoding="utf-8") as jsonl_file:
            for cond in conditions:
                self.logger.info("Starting condition: %s (%s)", cond.condition_id, cond.baseline_type)

                for doc_id in doc_ids:
                    run_count += 1
                    derived_seed = derive_seed(
                        experiment_seed=self.experiment_seed,
                        document_id=doc_id,
                        degradation_type=cond.degradation_type,
                        severity=cond.severity,
                        preprocessing_id=cond.preprocessing_id,
                    )

                    record: Dict[str, Any] = {
                        "experiment_id": self.experiment_id,
                        "condition_id": cond.condition_id,
                        "document_id": doc_id,
                        "baseline_type": cond.baseline_type,
                        "degradation": cond.degradation_type,
                        "severity": cond.severity,
                        "preprocessing": cond.preprocessing_id,
                        "derived_seed": derived_seed,
                        "status": "success",
                        "error_type": None,
                        "error_message": None,
                    }

                    try:
                        # Step A: Load Raw Image
                        raw_image = adapter.get_image(doc_id).copy()

                        # Step B: Apply Degradation
                        if cond.severity > 0 and cond.degradation_type != "none":
                            deg_spec = DegradationSpec(
                                type=cond.degradation_type,
                                severity=cond.severity,
                                seed=derived_seed,
                            )
                            deg_engine = get_degradation(cond.degradation_type)
                            current_image, _ = deg_engine.apply(raw_image, deg_spec)
                        else:
                            current_image = raw_image

                        # Step C: Apply Preprocessing
                        if cond.preprocessing_id not in ("none", "p0", "raw", ""):
                            prep_methods = prep_pipelines_cfg.get(cond.preprocessing_id)
                            if not prep_methods:
                                clean_name = (
                                    cond.preprocessing_id[2:]
                                    if cond.preprocessing_id.lower().startswith("p_")
                                    else cond.preprocessing_id
                                )
                                prep_methods = [clean_name]
                            elif isinstance(prep_methods, str):
                                prep_methods = [prep_methods]

                            prep_spec = PreprocessingSpec(
                                enabled=True,
                                methods=prep_methods,
                            )
                            if len(prep_methods) == 1:
                                prep_engine = get_preprocessor(prep_methods[0])
                                current_image, _ = prep_engine.process(current_image, prep_spec)
                            else:
                                from src.preprocessing.pipeline import PreprocessingPipeline
                                prep_pipeline = PreprocessingPipeline(steps=prep_methods)
                                current_image, _ = prep_pipeline.process(current_image, prep_spec)

                        # Save debug image if requested
                        if self.save_images:
                            img_out_path = self.images_dir / f"{cond.condition_id}_{doc_id}.jpg"
                            cv2.imwrite(str(img_out_path), current_image)

                        # Step D: OCR Inference (Strict GT Isolation)
                        try:
                            ocr_result = ocr_engine.recognize(current_image, doc_id)
                        except Exception as e:
                            record["status"] = "ocr_failed"
                            record["error_type"] = type(e).__name__
                            record["error_message"] = str(e)
                            self.logger.error("OCR failure for %s in %s: %s", doc_id, cond.condition_id, e)
                            per_condition_records[cond.condition_id].append(record)
                            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                            continue

                        # Step E: KIE Inference (Strict GT Isolation: only ocr_result and doc_id)
                        try:
                            kie_result = kie_engine.extract(ocr_result=ocr_result, document_id=doc_id)
                        except Exception as e:
                            record["status"] = "kie_failed"
                            record["error_type"] = type(e).__name__
                            record["error_message"] = str(e)
                            self.logger.error("KIE failure for %s in %s: %s", doc_id, cond.condition_id, e)
                            per_condition_records[cond.condition_id].append(record)
                            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                            continue

                        # Step F: Load Ground Truth (Strictly AFTER inference)
                        ocr_gt = adapter.get_ocr_ground_truth(doc_id)
                        kie_gt = adapter.get_kie_ground_truth(doc_id)

                        # Step G: Evaluation
                        try:
                            ocr_metrics = evaluator.evaluate_ocr(ocr_result, ocr_gt)
                            kie_metrics = evaluator.evaluate_kie(kie_result, kie_gt)
                        except Exception as e:
                            record["status"] = "evaluation_failed"
                            record["error_type"] = type(e).__name__
                            record["error_message"] = str(e)
                            self.logger.error("Evaluation failure for %s in %s: %s", doc_id, cond.condition_id, e)
                            per_condition_records[cond.condition_id].append(record)
                            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                            continue

                        # Success Record
                        record["ocr_metrics"] = ocr_metrics
                        record["kie_metrics"] = kie_metrics
                        record["predicted_fields"] = kie_result.fields

                    except Exception as fatal_e:
                        record["status"] = "evaluation_failed"
                        record["error_type"] = type(fatal_e).__name__
                        record["error_message"] = str(fatal_e)
                        self.logger.error("Fatal pipeline error for %s in %s: %s", doc_id, cond.condition_id, fatal_e)

                    per_condition_records[cond.condition_id].append(record)
                    jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")

        t_global_end = time.perf_counter()
        end_time_iso = datetime.now().isoformat()
        total_duration_sec = round(t_global_end - t_global_start, 2)

        # 6. Compute Aggregations per Condition
        boot_cfg = self.config.get("evaluation", {}).get("bootstrap", {})
        boot_iterations = boot_cfg.get("iterations", 1000)
        boot_ci = boot_cfg.get("ci", 0.95)
        boot_seed = boot_cfg.get("seed", 42)

        condition_summaries: Dict[str, Any] = {}
        total_requested = 0
        total_successful = 0
        total_failed = 0

        for c_idx, cond in enumerate(conditions):
            records = per_condition_records[cond.condition_id]
            c_summary = aggregate_condition_records(
                records=records,
                bootstrap_iterations=boot_iterations,
                ci=boot_ci,
                bootstrap_seed=boot_seed + (c_idx * 20),
            )
            c_summary["condition"] = cond.to_dict()
            condition_summaries[cond.condition_id] = c_summary

            total_requested += c_summary["document_counts"]["requested"]
            total_successful += c_summary["document_counts"]["successful"]
            total_failed += c_summary["document_counts"]["failed"]

        overall_failure_rate = round(total_failed / total_requested, 4) if total_requested > 0 else 0.0

        # 7. Write summary.json
        summary_payload = {
            "title": "SROIE OCR/KIE Robustness Experiment Summary",
            "experiment_id": self.experiment_id,
            "config_hash": self.config_hash,
            "data_status": data_status,
            "split": resolved_split,
            "smoke_mode": self.smoke_mode,
            "document_counts": {
                "total_requested": total_requested,
                "total_successful": total_successful,
                "total_failed": total_failed,
                "failure_rate": overall_failure_rate,
            },
            "conditions": condition_summaries,
            "execution_duration_sec": total_duration_sec,
        }
        summary_path = self.run_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2, ensure_ascii=False)

        # 8. Write manifest.json
        # NOTE: Runtime timestamps (start_time, end_time) are recorded as metadata only
        # and do not participate in deterministic configuration identity.
        manifest_payload = {
            "experiment_id": self.experiment_id,
            "config_hash": self.config_hash,
            "git_commit": self.git_commit,
            "data_status": data_status,
            "dataset_root": adapter.root_dir.as_posix(),
            "split": resolved_split,
            "smoke_mode": self.smoke_mode,
            "num_conditions": len(conditions),
            "num_documents_per_condition": len(doc_ids),
            "total_evaluations": total_requested,
            "total_successful": total_successful,
            "total_failed": total_failed,
            "failure_rate": overall_failure_rate,
            "ocr_engine": {
                "name": getattr(ocr_engine, "name", "rapidocr"),
                "reading_order_tolerance": getattr(self, "ocr_params", {}).get("line_tolerance_factor", 0.5),
            },
            "kie_engine": {
                "name": getattr(kie_engine, "model_name", "rule_based"),
                "fallback_largest_amount": self.config.get("kie", {}).get("engine", {}).get("fallback_largest_amount", False),
            },
            "conditions_evaluated": [c.condition_id for c in conditions],
            "runtime_metadata": {
                "start_time": start_time_iso,
                "end_time": end_time_iso,
                "duration_seconds": total_duration_sec,
            },
        }
        manifest_path = self.run_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_payload, f, indent=2, ensure_ascii=False)

        # 9. Write resolved config.yaml
        config_out_path = self.run_dir / "config.yaml"
        with open(config_out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.config, f, sort_keys=False)

        self.logger.info("Experiment run %s completed in %.2fs", self.experiment_id, total_duration_sec)
        self.logger.info("Artifacts saved to: %s", self.run_dir)

        return summary_payload
