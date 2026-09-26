"""SROIE dataset adapter for ICDAR 2019 Task 1-3 Cleaned (urbikn/sroie-datasetv2)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import cv2
import numpy as np

from src.core.contracts import BaseDatasetAdapter
from src.core.schemas import (
    BoundingBox,
    DocumentMetadata,
    KIEGroundTruth,
    OCRToken,
    OCRGroundTruth,
)


class SROIEAdapter(BaseDatasetAdapter):
    """Adapter for accessing SROIE 2019 receipts dataset in canonical Kaggle v2 layout."""

    SUPPORTED_SPLITS: Tuple[str, ...] = ("train", "test")
    REQUIRED_KIE_KEYS: Tuple[str, ...] = ("company", "date", "address", "total")
    IMAGE_EXTENSIONS: Tuple[str, ...] = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    )

    def __init__(self, root_dir: str | Path) -> None:
        """Initialize SROIE adapter with root dataset directory path.

        Args:
            root_dir: Path to SROIE dataset directory. Supports both root
                containing train/test directly and root containing SROIE2019/train.
        """
        raw_path = Path(root_dir)
        if not raw_path.exists():
            raise FileNotFoundError(f"Dataset root directory does not exist: {raw_path}")

        # Support both direct layout and nested SROIE2019/ subdirectory layout
        if (raw_path / "train").exists() or (raw_path / "test").exists():
            self.root_dir = raw_path.resolve()
        elif (raw_path / "SROIE2019" / "train").exists() or (raw_path / "SROIE2019" / "test").exists():
            self.root_dir = (raw_path / "SROIE2019").resolve()
        else:
            self.root_dir = raw_path.resolve()

        self._index: Dict[str, Dict[str, Any]] = {}
        self._split_ids: Dict[str, List[str]] = {s: [] for s in self.SUPPORTED_SPLITS}
        self._build_index()

    def _build_index(self) -> None:
        """Scan directory tree and construct document index."""
        for split in self.SUPPORTED_SPLITS:
            split_dir = self.root_dir / split
            if not split_dir.exists():
                continue

            img_dir = split_dir / "img"
            box_dir = split_dir / "box"
            entities_dir = split_dir / "entities"

            img_map: Dict[str, Path] = {}
            if img_dir.is_dir():
                for p in sorted(img_dir.iterdir()):
                    if (
                        p.is_file()
                        and p.suffix.lower() in self.IMAGE_EXTENSIONS
                        and not p.name.startswith(".")
                    ):
                        img_map[p.stem] = p

            box_map: Dict[str, Path] = {}
            if box_dir.is_dir():
                for p in sorted(box_dir.iterdir()):
                    if (
                        p.is_file()
                        and p.suffix.lower() == ".txt"
                        and not p.name.startswith(".")
                    ):
                        box_map[p.stem] = p

            entities_map: Dict[str, Path] = {}
            if entities_dir.is_dir():
                for p in sorted(entities_dir.iterdir()):
                    if (
                        p.is_file()
                        and p.suffix.lower() in (".txt", ".json")
                        and not p.name.startswith(".")
                    ):
                        entities_map[p.stem] = p

            all_stems = sorted(
                set(img_map.keys()) | set(box_map.keys()) | set(entities_map.keys())
            )
            for stem in all_stems:
                self._index[stem] = {
                    "split": split,
                    "img_path": img_map.get(stem),
                    "box_path": box_map.get(stem),
                    "entities_path": entities_map.get(stem),
                }

            # If images exist, list_document_ids defaults to image IDs, else all discovered stems
            if img_map:
                self._split_ids[split] = sorted(img_map.keys())
            else:
                self._split_ids[split] = all_stems

    def list_document_ids(self, split: str) -> List[str]:
        """List document IDs for a given split in deterministic sorted order.

        Args:
            split: 'train' or 'test'.

        Returns:
            Sorted list of document IDs without file extensions.

        Raises:
            ValueError: If split is not supported.
            FileNotFoundError: If the split directory does not exist on disk.
        """
        if split not in self.SUPPORTED_SPLITS:
            raise ValueError(
                f"Unsupported split '{split}'. Expected one of {self.SUPPORTED_SPLITS}."
            )

        split_dir = self.root_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split directory '{split}' not found at '{split_dir}'."
            )

        return list(self._split_ids.get(split, []))

    def get_document_info(self, document_id: str) -> Dict[str, Any]:
        """Return raw path mapping and split for a document.

        Args:
            document_id: Document ID.

        Returns:
            Dictionary with 'split', 'img_path', 'box_path', 'entities_path'.

        Raises:
            KeyError: If document ID is not found in the dataset.
        """
        if document_id not in self._index:
            raise KeyError(
                f"Document '{document_id}' not found in dataset at '{self.root_dir}'."
            )
        return dict(self._index[document_id])

    def get_metadata(self, document_id: str) -> DocumentMetadata:
        """Get standardized metadata for a document.

        Args:
            document_id: Document ID.

        Returns:
            Standardized DocumentMetadata instance.

        Raises:
            KeyError: If document ID is not in index.
            FileNotFoundError: If image file is missing.
            ValueError: If image file cannot be read.
        """
        entry = self.get_document_info(document_id)
        img_path = entry.get("img_path")
        if img_path is None or not img_path.exists():
            raise FileNotFoundError(
                f"Image file for document '{document_id}' not found."
            )

        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(
                f"Failed to read image for document '{document_id}' at '{img_path}'."
            )

        h, w = img.shape[:2]
        return DocumentMetadata(
            document_id=document_id,
            image_path=str(img_path),
            width=int(w),
            height=int(h),
            split=entry["split"],
            metadata={},
        )

    def get_image(self, document_id: str) -> np.ndarray:
        """Load document image in canonical RGB representation.

        Args:
            document_id: Document ID.

        Returns:
            Image as numpy ndarray of shape (H, W, 3) and dtype uint8.

        Raises:
            KeyError: If document ID is not in index.
            FileNotFoundError: If image file is missing.
            ValueError: If image is empty, malformed, or has non-positive dimensions.
            TypeError: If image dtype is not uint8.
        """
        entry = self.get_document_info(document_id)
        img_path = entry.get("img_path")
        if img_path is None or not img_path.exists():
            raise FileNotFoundError(
                f"Image file for document '{document_id}' not found."
            )

        bgr = cv2.imread(str(img_path))
        if bgr is None or bgr.size == 0 or bgr.shape[0] <= 0 or bgr.shape[1] <= 0:
            raise ValueError(
                f"Image for document '{document_id}' at '{img_path}' could not be read or is empty."
            )

        if bgr.dtype != np.uint8:
            raise TypeError(
                f"Image dtype must be uint8, got {bgr.dtype} for document '{document_id}'"
            )

        if len(bgr.shape) != 3 or bgr.shape[2] != 3:
            raise ValueError(
                f"Image must have 3 channels (H, W, 3), got shape {bgr.shape} for document '{document_id}'"
            )

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    def get_ocr_ground_truth(self, document_id: str) -> OCRGroundTruth:
        """Load immutable raw OCR ground truth for a document.

        Preserves raw transcript without text normalization, lowercasing,
        or semantic punctuation removal. Preserves line annotation order.

        Args:
            document_id: Document ID.

        Returns:
            Standardized OCRGroundTruth instance.

        Raises:
            KeyError: If document ID is not in index.
            FileNotFoundError: If OCR box annotation file is missing.
            ValueError: If OCR file is empty, or line format/coordinates are malformed.
        """
        entry = self.get_document_info(document_id)
        box_path = entry.get("box_path")
        if box_path is None or not box_path.exists():
            raise FileNotFoundError(
                f"OCR annotation file for document '{document_id}' not found."
            )

        with open(box_path, "r", encoding="utf-8-sig") as f:
            content = f.read()

        if not content.strip():
            raise ValueError(
                f"OCR annotation file for document '{document_id}' is empty."
            )

        tokens: List[OCRToken] = []
        polygons: List[List[float]] = []

        for line_num, line in enumerate(content.splitlines(), start=1):
            line_str = line.rstrip("\r\n")
            if not line_str.strip():
                continue

            all_parts = line_str.split(",")
            if len(all_parts) < 9:
                raise ValueError(
                    f"Malformed OCR annotation at line {line_num} in document '{document_id}': "
                    f"expected exactly 8 coordinates and text, got {len(all_parts)} parts in line: '{line_str}'"
                )

            try:
                coords = [float(x.strip()) for x in all_parts[:8]]
            except ValueError as e:
                raise ValueError(
                    f"Non-numeric OCR coordinate at line {line_num} in document '{document_id}': {all_parts[:8]}"
                ) from e

            if not all(math.isfinite(c) for c in coords):
                raise ValueError(
                    f"Non-finite OCR coordinate at line {line_num} in document '{document_id}': {coords}"
                )

            # Reject lines with > 8 coordinates:
            # If line has more than 9 comma-separated parts and the 9th item is numeric,
            # it indicates a 9th coordinate before the transcription.
            if len(all_parts) > 9:
                try:
                    float(all_parts[8].strip())
                    is_ninth_coord = True
                except ValueError:
                    is_ninth_coord = False

                if is_ninth_coord:
                    raise ValueError(
                        f"Malformed OCR annotation at line {line_num} in document '{document_id}': "
                        f"line contains more than 8 coordinates (found 9th numeric coordinate '{all_parts[8].strip()}')"
                    )

            # Raw transcript preserved verbatim (joining any remaining comma-split segments)
            raw_transcript = ",".join(all_parts[8:])

            xs = [coords[0], coords[2], coords[4], coords[6]]
            ys = [coords[1], coords[3], coords[5], coords[7]]
            x_min = float(min(xs))
            y_min = float(min(ys))
            x_max = float(max(xs))
            y_max = float(max(ys))

            # Raw unclipped BoundingBox derived directly from polygon
            bbox = BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
            tokens.append(OCRToken(text=raw_transcript, bbox=bbox, confidence=None))
            polygons.append(coords)

        full_text = "\n".join(t.text for t in tokens)
        return OCRGroundTruth(
            document_id=document_id,
            text=full_text,
            tokens=tokens,
            metadata={"polygons": polygons},
        )

    def get_kie_ground_truth(self, document_id: str) -> KIEGroundTruth:
        """Load immutable raw KIE ground truth for a document.

        Parses JSON entity annotation. Preserves raw string values without
        lowercasing, date parsing, float conversion, or heuristic cleaning.

        Args:
            document_id: Document ID.

        Returns:
            Standardized KIEGroundTruth instance.

        Raises:
            KeyError: If document ID is not in index.
            FileNotFoundError: If KIE annotation file is missing.
            ValueError: If JSON is malformed or required keys are missing.
            TypeError: If entity field values are not strings.
        """
        entry = self.get_document_info(document_id)
        entities_path = entry.get("entities_path")
        if entities_path is None or not entities_path.exists():
            raise FileNotFoundError(
                f"KIE annotation file for document '{document_id}' not found."
            )

        with open(entities_path, "r", encoding="utf-8-sig") as f:
            content = f.read()

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Malformed KIE JSON in document '{document_id}': {e}"
            ) from e

        if not isinstance(data, dict):
            raise ValueError(
                f"KIE annotation for '{document_id}' must be a JSON object, got {type(data).__name__}"
            )

        missing_keys = [k for k in self.REQUIRED_KIE_KEYS if k not in data]
        if missing_keys:
            raise ValueError(
                f"Document '{document_id}' KIE is missing required keys: {missing_keys}"
            )

        for k in self.REQUIRED_KIE_KEYS:
            val = data[k]
            if not isinstance(val, str):
                raise TypeError(
                    f"Document '{document_id}' KIE field '{k}' must be str, got {type(val).__name__}: {val!r}"
                )

        # Raw GT immutability: verbatim copy of required keys
        raw_fields = {k: data[k] for k in self.REQUIRED_KIE_KEYS}
        raw_metadata = {k: v for k, v in data.items() if k not in self.REQUIRED_KIE_KEYS}

        return KIEGroundTruth(
            document_id=document_id,
            fields=raw_fields,
            metadata=raw_metadata,
        )
