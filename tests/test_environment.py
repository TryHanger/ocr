"""Smoke tests verifying that the basic Python environment has required packages."""

def test_numpy_import():
    import numpy as np
    arr = np.array([1, 2, 3])
    assert arr.sum() == 6

def test_opencv_import():
    import cv2
    import numpy as np
    # Create a blank 10x10 RGB image and test basic cvtColor
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert gray.shape == (10, 10)

def test_pyyaml_import():
    import yaml
    payload = yaml.safe_load("key: value\nlist:\n  - 1\n  - 2")
    assert payload["key"] == "value"
    assert payload["list"] == [1, 2]

def test_pytest_available():
    import pytest
    assert pytest is not None
