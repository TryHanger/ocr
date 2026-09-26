"""Configuration loading, validation, and canonical hashing for experiments."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml


VOLATILE_CONFIG_KEYS = {
    "timestamp",
    "pid",
    "temp_path",
    "output_dir",
    "log_dir",
    "start_time",
    "end_time",
    "run_id",
}


def load_experiment_config(config_path: Union[Path, str]) -> Dict[str, Any]:
    """Load and parse YAML experiment configuration."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Experiment configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    return cfg


def canonicalize_for_hashing(obj: Any) -> Any:
    """Recursively strip volatile runtime fields and sort keys for stable hashing."""
    if isinstance(obj, dict):
        cleaned: Dict[str, Any] = {}
        for k, v in sorted(obj.items()):
            if k in VOLATILE_CONFIG_KEYS:
                continue
            cleaned[k] = canonicalize_for_hashing(v)
        return cleaned
    elif isinstance(obj, (list, tuple)):
        return [canonicalize_for_hashing(item) for item in obj]
    else:
        return obj


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute deterministic SHA256 hash of a canonicalized configuration dictionary.

    Guarantees:
    - Identical resolved configurations yield identical hash.
    - Volatile runtime fields (timestamps, output directories, PIDs) are excluded.
    - Key order differences do not affect the hash.
    """
    canonical_obj = canonicalize_for_hashing(config)
    serialized = json.dumps(canonical_obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
