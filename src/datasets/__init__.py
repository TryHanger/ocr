"""Dataset adapters package for OCR and KIE benchmark datasets."""

from __future__ import annotations

from src.datasets.base import BaseDatasetAdapter
from src.datasets.sroie import SROIEAdapter

__all__ = ["BaseDatasetAdapter", "SROIEAdapter"]
