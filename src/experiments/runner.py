"""Unified Experiment Runner for SROIE Robustness Matrix across B0, B1, and B2."""

from __future__ import annotations

from datetime import datetime
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import yaml

from src.core.schemas import DegradationSpec, PreprocessingSpec
from src.datasets.sroie import SROIEAdapter
from src.degradation import get_degradation
from src.evaluation.kie_metrics import KIEEvaluator
from src.experiments.bootstrap import aggregate_condition_records
from src.experiments.conditions import (
    DEGRADATION_CODE_MAP,
    ExperimentCondition,
    build_conditions_matrix,
)
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
        debug_timing: bool = False,
    ) -> None:
        self.config = config
        self.smoke_mode = smoke_mode
        self.allow_test = allow_test
        self.allow_fixture = allow_fixture
        self.save_images = save_images
        self.debug_timing = debug_timing

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

        # Initialize Logger with immediate flushing
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        if self.save_images:
            self.images_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(f"runner_{self.experiment_id}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        class FlushingFileHandler(logging.FileHandler):
            def emit(self, record: logging.LogRecord) -> None:
                super().emit(record)
                self.flush()

        log_file = self.logs_dir / "run.log"
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
        file_handler = FlushingFileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)
        self.logger.propagate = False

        # Load calibrated degradation configuration
        self.degradation_config = self._load_degradation_configs()

    def _load_degradation_configs(self) -> Dict[str, Any]:
        """Load degradation parameters configuration (configs/degradation.yaml)."""
        repo_root = Path(__file__).resolve().parent.parent.parent
        deg_cfg_path = Path(self.config.get("degradation_config", "configs/degradation.yaml"))
        if not deg_cfg_path.is_absolute():
            deg_cfg_path = (repo_root / deg_cfg_path).resolve()
        if deg_cfg_path.is_file():
            with open(deg_cfg_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _resolve_degradation_params(self, degradation_type: str, severity: int) -> Dict[str, Any]:
        """Resolve calibrated degradation parameters for a given type and severity."""
        if severity == 0 or degradation_type in ("none", "control"):
            return {}
        degs = self.degradation_config.get("degradations", {})
        if degradation_type not in degs:
            raise KeyError(f"Degradation '{degradation_type}' not found in degradation configuration.")
        sev_dict = degs[degradation_type].get("severities", {})
        params = sev_dict.get(severity) if severity in sev_dict else sev_dict.get(str(severity))
        if params is None:
            raise KeyError(
                f"Severity '{severity}' for degradation '{degradation_type}' not found in degradation configuration."
            )
        return dict(params)

    def _load_b0_reference(self) -> Optional[Dict[str, Any]]:
        """Load frozen B0 reference summary if available."""
        repo_root = Path(__file__).resolve().parent.parent.parent
        b0_ref_path = self.config.get("b0_reference_path", "experiments/runs/b0_clean_validation_n126/summary.json")
        p = Path(b0_ref_path)
        if not p.is_absolute():
            p = (repo_root / p).resolve()
        if p.is_file():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    b0_data = json.load(f)
                conds = b0_data.get("conditions", {})
                if "D0_S0_P0" in conds:
                    git_commit = b0_data.get("git_commit")
                    manifest_path = p.parent / "manifest.json"
                    if not git_commit and manifest_path.is_file():
                        try:
                            with open(manifest_path, "r", encoding="utf-8") as mf:
                                git_commit = json.load(mf).get("git_commit")
                        except Exception:
                            pass
                    return {
                        "source": "frozen_b0_reference",
                        "experiment_id": b0_data.get("experiment_id"),
                        "config_hash": b0_data.get("config_hash"),
                        "git_commit": git_commit or "UNKNOWN_COMMIT",
                        "path": str(p),
                        "metrics": conds["D0_S0_P0"],
                    }
            except Exception as e:
                self.logger.warning("Could not load B0 reference from %s: %s", p, e)
        return None

    def _load_b1_reference(self) -> Optional[Dict[str, Any]]:
        """Load frozen B1 reference summary and metadata if configured."""
        repo_root = Path(__file__).resolve().parent.parent.parent
        b1_ref_path = self.config.get("b1_reference_path")
        if not b1_ref_path:
            return None
        p = Path(b1_ref_path)
        if not p.is_absolute():
            p = (repo_root / p).resolve()
        if not p.is_file():
            self.logger.warning("B1 reference path '%s' not found.", p)
            return None

        try:
            with open(p, "r", encoding="utf-8") as f:
                b1_data = json.load(f)

            # Metadata validation
            exp_id = b1_data.get("experiment_id")
            split = b1_data.get("split")
            data_status = b1_data.get("data_status")

            if exp_id != "b1_degraded_validation_n126_gpu":
                self.logger.warning(
                    "B1 reference experiment_id '%s' differs from expected 'b1_degraded_validation_n126_gpu'",
                    exp_id,
                )
            if split != "validation":
                self.logger.warning("B1 reference split '%s' differs from expected 'validation'", split)
            if data_status != "REAL_SROIE":
                self.logger.warning("B1 reference data_status '%s' differs from expected 'REAL_SROIE'", data_status)

            git_commit = b1_data.get("git_commit")
            manifest_path = p.parent / "manifest.json"
            if not git_commit and manifest_path.is_file():
                try:
                    with open(manifest_path, "r", encoding="utf-8") as mf:
                        git_commit = json.load(mf).get("git_commit")
                except Exception:
                    pass

            return {
                "source": "frozen_b1_reference",
                "experiment_id": exp_id,
                "config_hash": b1_data.get("config_hash"),
                "git_commit": git_commit or "UNKNOWN_COMMIT",
                "path": str(p),
                "conditions": b1_data.get("conditions", {}),
            }
        except Exception as e:
            self.logger.warning("Could not load B1 reference from %s: %s", p, e)
            return None

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
            if "execution_provider" in ocr_cfg:
                ocr_params["execution_provider"] = ocr_cfg["execution_provider"]

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

        # 1. Setup Engines and Adapter (measuring initialization time)
        t_engine_init_start = time.perf_counter()
        ocr_engine, kie_engine, evaluator, adapter = self._setup_engines()
        engine_init_seconds = round(time.perf_counter() - t_engine_init_start, 4)

        # 2. Resolve Split & Documents (measuring dataset load time)
        t_data_load_start = time.perf_counter()
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
        data_load_seconds = round(time.perf_counter() - t_data_load_start, 4)

        # 3. Build Condition Matrix
        conditions = build_conditions_matrix(self.config, smoke_mode=self.smoke_mode)

        # Log one-time initialization metrics
        self.logger.info(
            "[INIT] dataset_load_seconds=%.4f engine_initialization_seconds=%.4f number_of_documents=%d number_of_conditions=%d",
            data_load_seconds,
            engine_init_seconds,
            len(doc_ids),
            len(conditions),
        )
        self.logger.info(
            "Resolved split '%s' with %d documents (Data Status: %s)",
            resolved_split,
            len(doc_ids),
            data_status,
        )
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

                    # Helpers for stage timing and logging
                    def _log_start(stg: str) -> float:
                        self.logger.info("[START] doc=%s condition=%s stage=%s", doc_id, cond.condition_id, stg)
                        return time.perf_counter()

                    def _log_done(stg: str, t_st: float) -> None:
                        elapsed = time.perf_counter() - t_st
                        self.logger.info("[DONE]  doc=%s condition=%s stage=%s elapsed=%.4fs", doc_id, cond.condition_id, stg, elapsed)

                    try:
                        # Step A: Load Raw Image
                        t_st = _log_start("load_image")
                        raw_image = adapter.get_image(doc_id).copy()
                        _log_done("load_image", t_st)

                        # Step B: Apply Degradation
                        t_st = _log_start("degradation")
                        deg_params: Dict[str, Any] = {}
                        if cond.severity > 0 and cond.degradation_type != "none":
                            deg_params = self._resolve_degradation_params(cond.degradation_type, cond.severity)
                            deg_spec = DegradationSpec(
                                type=cond.degradation_type,
                                severity=cond.severity,
                                parameters=deg_params,
                                seed=derived_seed,
                            )
                            deg_engine = get_degradation(cond.degradation_type)
                            current_image, _ = deg_engine.apply(raw_image, deg_spec)
                        else:
                            current_image = raw_image
                        record["degradation_parameters"] = deg_params
                        _log_done("degradation", t_st)

                        # Step C: Apply Preprocessing
                        t_st = _log_start("preprocessing")
                        prep_methods = []
                        prep_params = {}
                        if cond.preprocessing_id not in ("none", "p0", "raw", ""):
                            prep_def = prep_pipelines_cfg.get(cond.preprocessing_id)
                            if prep_def is None:
                                for k, v in prep_pipelines_cfg.items():
                                    if k.lower() == cond.preprocessing_id.lower():
                                        prep_def = v
                                        break
                            if prep_def is None:
                                prep_def = {}

                            if isinstance(prep_def, dict):
                                prep_methods = list(prep_def.get("steps", prep_def.get("methods", [])))
                                prep_params = dict(prep_def.get("parameters", {}))
                            elif isinstance(prep_def, list):
                                prep_methods = list(prep_def)
                                prep_params = {}
                            elif isinstance(prep_def, str):
                                prep_methods = [prep_def]
                                prep_params = {}
                            else:
                                clean_name = (
                                    cond.preprocessing_id[2:]
                                    if cond.preprocessing_id.lower().startswith("p_")
                                    else cond.preprocessing_id
                                )
                                prep_methods = [clean_name]
                                prep_params = {}

                            prep_spec = PreprocessingSpec(
                                enabled=True,
                                methods=prep_methods,
                                parameters=prep_params,
                            )
                            from src.preprocessing.pipeline import PreprocessingPipeline
                            prep_pipeline = PreprocessingPipeline(steps=prep_methods)
                            current_image, _ = prep_pipeline.process(current_image, prep_spec)

                        record["preprocessing_steps"] = prep_methods
                        record["preprocessing_parameters"] = prep_params
                        _log_done("preprocessing", t_st)

                        # Save debug image if requested
                        if self.save_images:
                            img_out_path = self.images_dir / f"{cond.condition_id}_{doc_id}.jpg"
                            cv2.imwrite(str(img_out_path), current_image)

                        # Step D: OCR Inference (Strict GT Isolation)
                        t_st = _log_start("ocr")
                        try:
                            ocr_result = ocr_engine.recognize(current_image, doc_id)
                            _log_done("ocr", t_st)
                        except Exception as e:
                            _log_done("ocr", t_st)
                            record["status"] = "ocr_failed"
                            record["error_type"] = type(e).__name__
                            record["error_message"] = str(e)
                            self.logger.error("OCR failure for %s in %s: %s", doc_id, cond.condition_id, e)
                            per_condition_records[cond.condition_id].append(record)
                            t_w = _log_start("write_result")
                            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                            jsonl_file.flush()
                            _log_done("write_result", t_w)
                            continue

                        # Step E: KIE Inference (Strict GT Isolation: only ocr_result and doc_id)
                        t_st = _log_start("kie")
                        try:
                            kie_result = kie_engine.extract(ocr_result=ocr_result, document_id=doc_id)
                            _log_done("kie", t_st)
                        except Exception as e:
                            _log_done("kie", t_st)
                            record["status"] = "kie_failed"
                            record["error_type"] = type(e).__name__
                            record["error_message"] = str(e)
                            self.logger.error("KIE failure for %s in %s: %s", doc_id, cond.condition_id, e)
                            per_condition_records[cond.condition_id].append(record)
                            t_w = _log_start("write_result")
                            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                            jsonl_file.flush()
                            _log_done("write_result", t_w)
                            continue

                        # Step F: Load Ground Truth (Strictly AFTER inference)
                        t_st = _log_start("gt_load")
                        ocr_gt = adapter.get_ocr_ground_truth(doc_id)
                        kie_gt = adapter.get_kie_ground_truth(doc_id)
                        _log_done("gt_load", t_st)

                        # Step G: Evaluation
                        t_st = _log_start("evaluation")
                        try:
                            ocr_metrics = evaluator.evaluate_ocr(ocr_result, ocr_gt)
                            kie_metrics = evaluator.evaluate_kie(kie_result, kie_gt)
                            _log_done("evaluation", t_st)
                        except Exception as e:
                            _log_done("evaluation", t_st)
                            record["status"] = "evaluation_failed"
                            record["error_type"] = type(e).__name__
                            record["error_message"] = str(fatal_e if "fatal_e" in locals() else e)
                            self.logger.error("Evaluation failure for %s in %s: %s", doc_id, cond.condition_id, e)
                            per_condition_records[cond.condition_id].append(record)
                            t_w = _log_start("write_result")
                            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                            jsonl_file.flush()
                            _log_done("write_result", t_w)
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

                    t_w = _log_start("write_result")
                    per_condition_records[cond.condition_id].append(record)
                    jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    jsonl_file.flush()
                    _log_done("write_result", t_w)

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

        # 6.2 Compute Deltas Relative to B0 Reference
        b0_ref = self._load_b0_reference()
        ref_metrics: Optional[Dict[str, Any]] = None
        ref_provenance: Optional[Dict[str, Any]] = None

        if b0_ref is not None:
            ref_metrics = b0_ref["metrics"]
            ref_provenance = {
                "source": "frozen_b0_reference",
                "experiment_id": b0_ref.get("experiment_id"),
                "config_hash": b0_ref.get("config_hash"),
                "path": b0_ref.get("path"),
            }
        elif "D0_S0_P0" in condition_summaries:
            ref_metrics = condition_summaries["D0_S0_P0"]
            ref_provenance = {
                "source": "current_run_control",
                "condition_id": "D0_S0_P0",
            }

        if ref_metrics is not None:
            ref_ocr = ref_metrics.get("ocr", {})
            ref_kie = ref_metrics.get("kie", {})
            b0_cer_raw = ref_ocr.get("cer_raw", {}).get("mean", 0.0)
            b0_cer_norm = ref_ocr.get("cer_normalized", {}).get("mean", 0.0)
            b0_wer_raw = ref_ocr.get("wer_raw", {}).get("mean", 0.0)
            b0_wer_norm = ref_ocr.get("wer_normalized", {}).get("mean", 0.0)
            b0_ned_raw = ref_ocr.get("char_ned_raw", {}).get("mean", 0.0)
            b0_ned_norm = ref_ocr.get("char_ned_normalized", {}).get("mean", 0.0)
            b0_macro_f1_raw = ref_kie.get("macro_f1_raw", 0.0)
            b0_macro_f1_norm = ref_kie.get("macro_f1_normalized", 0.0)
            b0_hmean = ref_kie.get("sroie_official_compatible", {}).get("entity_hmean", 0.0)
            b0_doc_em_norm = ref_kie.get("normalized_doc_em_rate", 0.0)

            for cid, c_sum in condition_summaries.items():
                c_ocr = c_sum.get("ocr", {})
                c_kie = c_sum.get("kie", {})

                c_cer_raw = c_ocr.get("cer_raw", {}).get("mean", 0.0)
                c_cer_norm = c_ocr.get("cer_normalized", {}).get("mean", 0.0)
                c_wer_raw = c_ocr.get("wer_raw", {}).get("mean", 0.0)
                c_wer_norm = c_ocr.get("wer_normalized", {}).get("mean", 0.0)
                c_ned_raw = c_ocr.get("char_ned_raw", {}).get("mean", 0.0)
                c_ned_norm = c_ocr.get("char_ned_normalized", {}).get("mean", 0.0)
                c_macro_f1_raw = c_kie.get("macro_f1_raw", 0.0)
                c_macro_f1_norm = c_kie.get("macro_f1_normalized", 0.0)
                c_hmean = c_kie.get("sroie_official_compatible", {}).get("entity_hmean", 0.0)
                c_doc_em_norm = c_kie.get("normalized_doc_em_rate", 0.0)

                c_sum["delta_from_b0"] = {
                    "delta_cer_raw": round(c_cer_raw - b0_cer_raw, 4),
                    "delta_cer_normalized": round(c_cer_norm - b0_cer_norm, 4),
                    "delta_wer_raw": round(c_wer_raw - b0_wer_raw, 4),
                    "delta_wer_normalized": round(c_wer_norm - b0_wer_norm, 4),
                    "delta_char_ned_raw": round(c_ned_raw - b0_ned_raw, 4),
                    "delta_char_ned_normalized": round(c_ned_norm - b0_ned_norm, 4),
                    "delta_macro_f1_raw": round(c_macro_f1_raw - b0_macro_f1_raw, 4),
                    "delta_macro_f1_normalized": round(c_macro_f1_norm - b0_macro_f1_norm, 4),
                    "delta_entity_hmean": round(c_hmean - b0_hmean, 4),
                    "delta_normalized_doc_em": round(c_doc_em_norm - b0_doc_em_norm, 4),
                    "b0_reference_provenance": ref_provenance,
                }

        # 6.3 Compute Recovery Relative to Matching B1 Reference Condition
        b1_ref = self._load_b1_reference()
        b1_conditions = b1_ref.get("conditions", {}) if b1_ref else {}
        b1_provenance: Optional[Dict[str, Any]] = None
        if b1_ref is not None:
            b1_provenance = {
                "source": "frozen_b1_reference",
                "experiment_id": b1_ref.get("experiment_id"),
                "config_hash": b1_ref.get("config_hash"),
                "git_commit": b1_ref.get("git_commit"),
                "path": b1_ref.get("path"),
            }
            for cid, c_sum in condition_summaries.items():
                cond_info = c_sum.get("condition", {})
                deg_type = cond_info.get("degradation_type", "none")
                sev = cond_info.get("severity", 0)
                d_code = DEGRADATION_CODE_MAP.get(deg_type, deg_type).upper()
                matching_b1_id = f"{d_code}_S{sev}_P0"

                if matching_b1_id in b1_conditions:
                    b1_c = b1_conditions[matching_b1_id]
                    b1_ocr = b1_c.get("ocr", {})
                    b1_kie = b1_c.get("kie", {})

                    b1_cer_raw = b1_ocr.get("cer_raw", {}).get("mean", 0.0)
                    b1_cer_norm = b1_ocr.get("cer_normalized", {}).get("mean", 0.0)
                    b1_wer_raw = b1_ocr.get("wer_raw", {}).get("mean", 0.0)
                    b1_wer_norm = b1_ocr.get("wer_normalized", {}).get("mean", 0.0)
                    b1_ned_raw = b1_ocr.get("char_ned_raw", {}).get("mean", 0.0)
                    b1_ned_norm = b1_ocr.get("char_ned_normalized", {}).get("mean", 0.0)
                    b1_macro_f1_raw = b1_kie.get("macro_f1_raw", 0.0)
                    b1_macro_f1_norm = b1_kie.get("macro_f1_normalized", 0.0)
                    b1_hmean = b1_kie.get("sroie_official_compatible", {}).get("entity_hmean", 0.0)
                    b1_doc_em_norm = b1_kie.get("normalized_doc_em_rate", 0.0)

                    c_ocr = c_sum.get("ocr", {})
                    c_kie = c_sum.get("kie", {})
                    c_cer_raw = c_ocr.get("cer_raw", {}).get("mean", 0.0)
                    c_cer_norm = c_ocr.get("cer_normalized", {}).get("mean", 0.0)
                    c_wer_raw = c_ocr.get("wer_raw", {}).get("mean", 0.0)
                    c_wer_norm = c_ocr.get("wer_normalized", {}).get("mean", 0.0)
                    c_ned_raw = c_ocr.get("char_ned_raw", {}).get("mean", 0.0)
                    c_ned_norm = c_ocr.get("char_ned_normalized", {}).get("mean", 0.0)
                    c_macro_f1_raw = c_kie.get("macro_f1_raw", 0.0)
                    c_macro_f1_norm = c_kie.get("macro_f1_normalized", 0.0)
                    c_hmean = c_kie.get("sroie_official_compatible", {}).get("entity_hmean", 0.0)
                    c_doc_em_norm = c_kie.get("normalized_doc_em_rate", 0.0)

                    c_sum["recovery_from_b1"] = {
                        "reference_b1_condition_id": matching_b1_id,
                        "delta_cer_raw": round(c_cer_raw - b1_cer_raw, 4),
                        "delta_cer_normalized": round(c_cer_norm - b1_cer_norm, 4),
                        "delta_wer_raw": round(c_wer_raw - b1_wer_raw, 4),
                        "delta_wer_normalized": round(c_wer_norm - b1_wer_norm, 4),
                        "delta_char_ned_raw": round(c_ned_raw - b1_ned_raw, 4),
                        "delta_char_ned_normalized": round(c_ned_norm - b1_ned_norm, 4),
                        "delta_macro_f1_raw": round(c_macro_f1_raw - b1_macro_f1_raw, 4),
                        "delta_macro_f1_normalized": round(c_macro_f1_norm - b1_macro_f1_norm, 4),
                        "delta_entity_hmean": round(c_hmean - b1_hmean, 4),
                        "delta_normalized_doc_em": round(c_doc_em_norm - b1_doc_em_norm, 4),
                        "b1_reference_provenance": b1_provenance,
                    }

        # 7. Write summary.json
        summary_payload = {
            "title": "SROIE OCR/KIE Robustness Experiment Summary",
            "experiment_id": self.experiment_id,
            "config_hash": self.config_hash,
            "data_status": data_status,
            "split": resolved_split,
            "smoke_mode": self.smoke_mode,
            "execution_provider": getattr(ocr_engine, "execution_provider", "cpu"),
            "b0_reference": ref_provenance,
            "b1_reference": b1_provenance,
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
        resolved_degradation_parameters: Dict[str, Any] = {}
        for c in conditions:
            if c.severity > 0 and c.degradation_type != "none":
                resolved_degradation_parameters[c.condition_id] = self._resolve_degradation_params(
                    c.degradation_type, c.severity
                )

        resolved_prep_pipelines: Dict[str, Any] = {}
        for p_id, p_spec in prep_pipelines_cfg.items():
            if isinstance(p_spec, dict):
                resolved_prep_pipelines[p_id] = {
                    "steps": p_spec.get("steps", p_spec.get("methods", [])),
                    "parameters": p_spec.get("parameters", {}),
                }
            elif isinstance(p_spec, list):
                resolved_prep_pipelines[p_id] = {
                    "steps": p_spec,
                    "parameters": {},
                }
            else:
                resolved_prep_pipelines[p_id] = {
                    "steps": [p_spec],
                    "parameters": {},
                }

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
                "execution_provider": getattr(ocr_engine, "execution_provider", "cpu"),
                "onnxruntime_providers": getattr(ocr_engine, "get_model_manifest", lambda: {})().get("resolved", {}).get("detector", {}).get("providers", []),
            },
            "kie_engine": {
                "name": getattr(kie_engine, "model_name", "rule_based"),
                "fallback_largest_amount": self.config.get("kie", {}).get("engine", {}).get("fallback_largest_amount", False),
            },
            "degradation_config": {
                "path": str(self.config.get("degradation_config", "configs/degradation.yaml")),
                "parameter_status": self.degradation_config.get("parameter_status", "unknown"),
                "resolved_parameters": resolved_degradation_parameters,
            },
            "preprocessing_implementation": {
                "name": "src.preprocessing",
                "version": "1.0.0",
            },
            "preprocessing_pipelines": resolved_prep_pipelines,
            "b0_reference": ref_provenance,
            "b1_reference": b1_provenance,
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

        # 9. Write ocr_stack_manifest.json if available
        if hasattr(ocr_engine, "get_model_manifest"):
            try:
                ocr_manifest_path = self.run_dir / "ocr_stack_manifest.json"
                with open(ocr_manifest_path, "w", encoding="utf-8") as f:
                    json.dump(ocr_engine.get_model_manifest(), f, indent=2)
            except Exception as e:
                self.logger.warning("Could not export ocr_stack_manifest.json: %s", e)

        # 10. Write resolved config.yaml
        config_out_path = self.run_dir / "config.yaml"
        with open(config_out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.config, f, sort_keys=False)


        self.logger.info("Experiment run %s completed in %.2fs", self.experiment_id, total_duration_sec)
        self.logger.info("Artifacts saved to: %s", self.run_dir)

        # Explicitly close and remove all logger handlers to release file locks on Windows
        for handler in list(self.logger.handlers):
            try:
                handler.close()
            except Exception:
                pass
            self.logger.removeHandler(handler)

        return summary_payload
