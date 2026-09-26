"""Deterministic seed derivation for experiment conditions and documents."""

from __future__ import annotations

import hashlib


def derive_seed(
    experiment_seed: int,
    document_id: str,
    degradation_type: str,
    severity: int,
    preprocessing_id: str,
) -> int:
    """Derive a deterministic, stable 32-bit unsigned integer seed.

    Guarantees:
    - Stable across processes and platforms.
    - Independent of Python's randomized hash seed.
    - No global mutable RNG dependence.
    - Distinct document/condition combinations produce distinct seeds.

    Formula:
        SHA256(f"{experiment_seed}:{document_id}:{degradation_type}:{severity}:{preprocessing_id}")
        converted from first 8 hex characters into a 32-bit uint.
    """
    key = f"{experiment_seed}:{document_id}:{degradation_type}:{severity}:{preprocessing_id}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    # 8 hex characters = 32 bits, range [0, 4294967295]
    seed_uint32 = int(digest[:8], 16)
    return seed_uint32
