"""Dataset split resolution, subset sampling, and frozen test split safety guards."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.datasets.sroie import SROIEAdapter


def resolve_dataset_root(candidate_roots: Optional[List[Path | str]] = None) -> Tuple[Path, str]:
    """Resolve active dataset root directory and determine data status.

    Returns:
        Tuple of (resolved_path, data_status).
        data_status is either 'REAL_SROIE' or 'SYNTHETIC_FIXTURE'.
    """
    candidates = candidate_roots or [
        Path("data/SROIE2019"),
        Path("data"),
    ]

    for cand in candidates:
        p = Path(cand) if cand else None
        if p and p.is_dir():
            train_img = p / "train" / "img"
            if train_img.is_dir() and len(list(train_img.glob("*.jpg"))) >= 100:
                return p.resolve(), "REAL_SROIE"
            # Nested SROIE2019/train/img
            nested_img = p / "SROIE2019" / "train" / "img"
            if nested_img.is_dir() and len(list(nested_img.glob("*.jpg"))) >= 100:
                return (p / "SROIE2019").resolve(), "REAL_SROIE"

    # Fallback to local test fixtures
    fixture_valid = Path("tests/fixtures/sroie_valid")
    if fixture_valid.is_dir() and (fixture_valid / "train" / "img").is_dir():
        return fixture_valid.resolve(), "SYNTHETIC_FIXTURE"

    fixture_sroie = Path("tests/fixtures/sroie")
    if fixture_sroie.is_dir() and (fixture_sroie / "train" / "img").is_dir():
        return fixture_sroie.resolve(), "SYNTHETIC_FIXTURE"

    return Path("tests/fixtures/sroie_valid").resolve(), "SYNTHETIC_FIXTURE"


def resolve_split_documents(
    adapter: SROIEAdapter,
    split: str = "development",
    subset_size: Optional[int] = None,
    subset_seed: int = 42,
    allow_test: bool = False,
    allow_fixture: bool = False,
) -> Tuple[List[str], str, str]:
    """Resolve document IDs for the requested split with safety guards and subsetting.

    Args:
        adapter: SROIE dataset adapter.
        split: One of 'development', 'validation', 'test'.
        subset_size: Optional maximum number of documents to select.
        subset_seed: Random seed for deterministic subset sampling.
        allow_test: Explicit safety override required to use 'test' split.
        allow_fixture: Explicit override allowing synthetic fixtures when real data absent.

    Returns:
        Tuple of (selected_doc_ids, resolved_split_name, data_status).

    Raises:
        ValueError: If split is invalid or if 'test' requested without allow_test.
        FileNotFoundError: If real SROIE data is missing and allow_fixture is False.
    """
    normalized_split = split.lower().strip()

    # 1. Enforce Test Split Guard
    if normalized_split in ("test", "test_split", "test_set"):
        if not allow_test:
            raise ValueError(
                "Test split (N=347) is strictly frozen for final evaluation and prohibited for "
                "development, calibration, or tuning. Pass --allow-test to explicitly override."
            )
        target_dataset_split = "test"
    elif normalized_split in ("dev", "development", "val", "validation", "train"):
        target_dataset_split = "train"
    else:
        raise ValueError(
            f"Unsupported split '{split}'. Supported splits: ['development', 'validation', 'test']"
        )

    # 2. Check Real Data Availability vs Fixture Mode
    # SROIE Kaggle artifact has >= 100 images in train
    all_train_ids = adapter.list_document_ids("train") if (adapter.root_dir / "train").exists() else []
    all_test_ids = adapter.list_document_ids("test") if (adapter.root_dir / "test").exists() else []
    is_real_data = len(all_train_ids) >= 500 or len(all_test_ids) >= 100

    if not is_real_data and not allow_fixture:
        raise FileNotFoundError(
            "REAL_DATA_REQUIRED: SROIE dataset not found in candidate paths ('data/SROIE2019', 'data'). "
            "Pass --allow-fixture to explicitly permit synthetic test fixtures for engineering verification."
        )

    data_status = "REAL_SROIE" if is_real_data else "PENDING_REAL_DATA"

    # 3. Resolve Split Document IDs
    if normalized_split in ("test", "test_split", "test_set"):
        split_doc_ids = list(all_test_ids)
    elif normalized_split in ("dev", "development"):
        if len(all_train_ids) >= 626:
            # Deterministic split: seed 42, 500 dev / 126 val
            rng = np.random.default_rng(42)
            perm = rng.permutation(len(all_train_ids))
            dev_indices = sorted(perm[:500])
            split_doc_ids = [all_train_ids[i] for i in dev_indices]
        else:
            # Fixture or partial data
            split_doc_ids = list(all_train_ids)
    elif normalized_split in ("val", "validation"):
        if len(all_train_ids) >= 626:
            # Deterministic split: seed 42, 500 dev / 126 val
            rng = np.random.default_rng(42)
            perm = rng.permutation(len(all_train_ids))
            val_indices = sorted(perm[500:626])
            split_doc_ids = [all_train_ids[i] for i in val_indices]
        elif len(all_train_ids) >= 126:
            rng = np.random.default_rng(42)
            perm = rng.permutation(len(all_train_ids))
            val_indices = sorted(perm[-126:])
            split_doc_ids = [all_train_ids[i] for i in val_indices]
        else:
            split_doc_ids = list(all_train_ids)
    else:  # 'train' full
        split_doc_ids = list(all_train_ids)

    # 4. Apply Deterministic Subsetting
    if subset_size is not None and 0 < subset_size < len(split_doc_ids):
        sub_rng = np.random.default_rng(subset_seed)
        perm = sub_rng.permutation(len(split_doc_ids))
        chosen_indices = sorted(perm[:subset_size])
        final_doc_ids = [split_doc_ids[i] for i in chosen_indices]
    else:
        final_doc_ids = split_doc_ids

    return final_doc_ids, normalized_split, data_status
