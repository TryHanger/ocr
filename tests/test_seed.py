"""Unit tests for random seed utility and determinism."""

import os
import random
import numpy as np
import pytest

from src.core.seed import set_seed


def test_set_seed_returns_seed():
    result = set_seed(42)
    assert result == 42


def test_python_random_determinism():
    set_seed(12345)
    seq1 = [random.random() for _ in range(10)]

    set_seed(12345)
    seq2 = [random.random() for _ in range(10)]

    assert seq1 == seq2


def test_numpy_random_determinism():
    set_seed(999)
    arr1 = np.random.rand(5, 5)

    set_seed(999)
    arr2 = np.random.rand(5, 5)

    assert np.array_equal(arr1, arr2)


def test_different_seeds_produce_different_sequences():
    set_seed(1)
    seq1 = [random.random() for _ in range(10)]

    set_seed(2)
    seq2 = [random.random() for _ in range(10)]

    assert seq1 != seq2


def test_python_hash_seed_env():
    set_seed(777)
    assert os.environ.get("PYTHONHASHSEED") == "777"


@pytest.mark.parametrize("invalid_seed", [3.14, "42", None, True, False, [42]])
def test_invalid_seed_types_raise_type_error(invalid_seed):
    with pytest.raises(TypeError):
        set_seed(invalid_seed)
