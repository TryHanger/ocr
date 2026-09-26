"""Unified Experiment Runner and Experiment Matrix package."""

from src.experiments.bootstrap import aggregate_condition_records, compute_bootstrap_ci
from src.experiments.conditions import (
    ExperimentCondition,
    build_condition_id,
    build_conditions_matrix,
)
from src.experiments.config import compute_config_hash, load_experiment_config
from src.experiments.runner import UnifiedExperimentRunner
from src.experiments.seed import derive_seed
from src.experiments.split import resolve_dataset_root, resolve_split_documents

__all__ = [
    "derive_seed",
    "ExperimentCondition",
    "build_condition_id",
    "build_conditions_matrix",
    "resolve_dataset_root",
    "resolve_split_documents",
    "load_experiment_config",
    "compute_config_hash",
    "compute_bootstrap_ci",
    "aggregate_condition_records",
    "UnifiedExperimentRunner",
]
