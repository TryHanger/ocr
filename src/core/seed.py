"""Seed utility for reproducible experiments across random generators."""

import os
import random
import numpy as np


def set_seed(seed: int) -> int:
    """Set global random seed for Python random, NumPy, and environment hash seed.

    Args:
        seed: Integer seed value.

    Returns:
        The integer seed value that was configured.

    Raises:
        TypeError: If seed is not an integer.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"Seed must be an integer, got {type(seed).__name__}: {seed}")

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed
