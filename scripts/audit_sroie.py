"""SROIE dataset integrity audit script (validating checks C-01 to C-10)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Ensure project root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from typing import Any, Dict, List, Optional, Set, Tuple
import cv2
import numpy as np
import yaml

from src.datasets.sroie import SROIEAdapter


def load_config(config_path: Optional[str | Path]) -> Dict[str, Any]:
    """Load YAML configuration if exists, else return default dict."""
    if config_path is None:
        default_cfg = Path("configs/sroie.yaml")
        if default_cfg.exists():
            config_path = default_cfg
        else:
            return {}

    cfg_file = Path(config_path)
    if not cfg_file.exists():
        return {}

    with open(cfg_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def audit_sroie_dataset(
    root_dir: str | Path,
    expected_train_count: int = 626,
    expected_test_count: int = 347,
    strict_counts: bool = False,
    min_area_pixels: int = 10000,
) -> Dict[str, Any]:
    """Execute integrity audit C-01 to C-10 on an SROIE dataset.

    Args:
        root_dir: Dataset root directory path.
        expected_train_count: Canonical train document count.
        expected_test_count: Canonical test document count.
        strict_counts: Whether count mismatch is treated as an error.
        min_area_pixels: Minimum image area (H * W) requirement.

    Returns:
        Structured audit report dictionary.
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root directory does not exist: {root}")

    # Support nested SROIE2019/ subdirectory if present
    if (root / "SROIE2019" / "train").exists() or (root / "SROIE2019" / "test").exists():
        root = root / "SROIE2019"

    errors: List[str] = []
    warnings: List[str] = []

    splits = ("train", "test")
    split_ids: Dict[str, Dict[str, Set[str]]] = {}
    split_file_maps: Dict[str, Dict[str, Dict[str, Path]]] = {}
    image_shapes: Dict[str, Tuple[int, int]] = {}

    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    # 1. Discover files and collect IDs
    for split in splits:
        split_dir = root / split
        split_ids[split] = {"img": set(), "box": set(), "entities": set()}
        split_file_maps[split] = {"img": {}, "box": {}, "entities": {}}

        if not split_dir.exists():
            errors.append(f"Missing split directory: '{split_dir}'")
            continue

        for folder, key in (("img", "img"), ("box", "box"), ("entities", "entities")):
            target_dir = split_dir / folder
            if not target_dir.exists():
                errors.append(f"Missing '{folder}' directory in split '{split}': '{target_dir}'")
                continue

            for p in sorted(target_dir.iterdir()):
                if not p.is_file() or p.name.startswith("."):
                    continue

                if key == "img":
                    if p.suffix.lower() in image_exts:
                        if p.stem in split_file_maps[split]["img"]:
                            errors.append(
                                f"Duplicate image stem '{p.stem}' in '{split}/img': {p.name}"
                            )
                        split_file_maps[split]["img"][p.stem] = p
                        split_ids[split]["img"].add(p.stem)
                elif key == "box":
                    if p.suffix.lower() == ".txt":
                        split_file_maps[split]["box"][p.stem] = p
                        split_ids[split]["box"].add(p.stem)
                elif key == "entities":
                    if p.suffix.lower() in (".txt", ".json"):
                        split_file_maps[split]["entities"][p.stem] = p
                        split_ids[split]["entities"].add(p.stem)

    # C-09: Uniqueness of IDs within and between splits
    c09_passed = True
    train_all = (
        split_ids.get("train", {}).get("img", set())
        | split_ids.get("train", {}).get("box", set())
        | split_ids.get("train", {}).get("entities", set())
    )
    test_all = (
        split_ids.get("test", {}).get("img", set())
        | split_ids.get("test", {}).get("box", set())
        | split_ids.get("test", {}).get("entities", set())
    )
    split_leakage = train_all & test_all
    if split_leakage:
        c09_passed = False
        errors.append(
            f"[C-09] Split isolation violation! Document IDs present in both train and test: {sorted(split_leakage)}"
        )

    # C-10: Triplet consistency
    c10_passed = True
    discrepancies: Dict[str, Dict[str, List[str]]] = {}
    paired_ids: Dict[str, Set[str]] = {}

    for split in splits:
        img_set = split_ids.get(split, {}).get("img", set())
        box_set = split_ids.get(split, {}).get("box", set())
        ent_set = split_ids.get(split, {}).get("entities", set())

        paired = img_set & box_set & ent_set
        paired_ids[split] = paired

        missing_ocr = sorted(img_set - box_set)
        missing_kie = sorted(img_set - ent_set)
        missing_img = sorted((box_set | ent_set) - img_set)
        extra_ocr = sorted(box_set - img_set)
        extra_kie = sorted(ent_set - img_set)
        extra_img = sorted(img_set - (box_set & ent_set))

        discrepancies[split] = {
            "missing_images": missing_img,
            "missing_ocr": missing_ocr,
            "missing_kie": missing_kie,
            "extra_images": extra_img,
            "extra_ocr": extra_ocr,
            "extra_kie": extra_kie,
        }

        if missing_img or missing_ocr or missing_kie or extra_ocr or extra_kie:
            c10_passed = False
            if missing_img:
                errors.append(f"[C-10] Split '{split}' missing images for IDs: {missing_img}")
            if missing_ocr:
                errors.append(f"[C-10] Split '{split}' missing OCR annotations for IDs: {missing_ocr}")
            if missing_kie:
                errors.append(f"[C-10] Split '{split}' missing KIE annotations for IDs: {missing_kie}")
            if extra_ocr:
                warnings.append(f"[C-10] Split '{split}' has orphan OCR annotations: {extra_ocr}")
            if extra_kie:
                warnings.append(f"[C-10] Split '{split}' has orphan KIE annotations: {extra_kie}")

    # C-01, C-02, C-03: Image checks
    c01_passed = True
    c02_passed = True
    c03_passed = True

    for split in splits:
        for doc_id, img_path in sorted(split_file_maps[split]["img"].items()):
            # C-01: File exists and size > 0
            if not img_path.exists() or img_path.stat().st_size == 0:
                c01_passed = False
                errors.append(f"[C-01] Image file missing or empty: '{img_path}'")
                continue

            # C-02: OpenCV readable, uint8, 3 channels
            img = cv2.imread(str(img_path))
            if img is None:
                c02_passed = False
                errors.append(f"[C-02] Failed to decode image with OpenCV: '{img_path}'")
                continue

            if img.dtype != np.uint8:
                c02_passed = False
                errors.append(f"[C-02] Image dtype is not uint8 ({img.dtype}): '{img_path}'")

            if len(img.shape) != 3 or img.shape[2] != 3:
                c02_passed = False
                errors.append(f"[C-02] Image is not 3-channel RGB/BGR (shape {img.shape}): '{img_path}'")

            # C-03: Geometric dimensions
            h, w = img.shape[:2]
            image_shapes[doc_id] = (h, w)
            if h <= 0 or w <= 0 or (h * w) < min_area_pixels:
                c03_passed = False
                errors.append(
                    f"[C-03] Image dimensions {w}x{h} (area {w * h}) below minimum threshold ({min_area_pixels}): '{img_path}'"
                )

    # C-04, C-05, C-06: OCR checks
    c04_passed = True
    c05_passed = True
    c06_passed = True

    for split in splits:
        for doc_id, box_path in sorted(split_file_maps[split]["box"].items()):
            # C-04: Exists and non-empty
            if not box_path.exists() or box_path.stat().st_size == 0:
                c04_passed = False
                errors.append(f"[C-04] OCR annotation file is empty or missing: '{box_path}'")
                continue

            try:
                with open(box_path, "r", encoding="utf-8-sig") as f:
                    content = f.read()
            except Exception as e:
                c04_passed = False
                errors.append(f"[C-04] Failed to read OCR file '{box_path}': {e}")
                continue

            if not content.strip():
                c04_passed = False
                errors.append(f"[C-04] OCR annotation file content is blank: '{box_path}'")
                continue

            # C-05: Polygon correctness
            lines = content.splitlines()
            h_img, w_img = image_shapes.get(doc_id, (None, None))

            for line_idx, line in enumerate(lines, start=1):
                raw_line = line.rstrip("\r\n")
                if not raw_line.strip():
                    continue

                parts = raw_line.split(",", 8)
                if len(parts) < 9:
                    c05_passed = False
                    errors.append(
                        f"[C-05] Document '{doc_id}' line {line_idx}: expected 8 coords + text, got {len(parts)} parts: '{raw_line}'"
                    )
                    continue

                try:
                    coords = [float(x.strip()) for x in parts[:8]]
                except ValueError:
                    c05_passed = False
                    errors.append(
                        f"[C-05] Document '{doc_id}' line {line_idx}: non-numeric coordinates: {parts[:8]}"
                    )
                    continue

                if not all(math.isfinite(c) for c in coords):
                    c05_passed = False
                    errors.append(
                        f"[C-05] Document '{doc_id}' line {line_idx}: non-finite coordinates: {coords}"
                    )
                    continue

                # C-06: Bounding box geometry
                xs = [coords[0], coords[2], coords[4], coords[6]]
                ys = [coords[1], coords[3], coords[5], coords[7]]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)

                if x_max < x_min or y_max < y_min:
                    c06_passed = False
                    errors.append(
                        f"[C-06] Document '{doc_id}' line {line_idx}: invalid geometry x_max < x_min or y_max < y_min"
                    )

                # Check if coordinates lie within image frame
                if w_img is not None and h_img is not None:
                    if x_min < 0 or y_min < 0 or x_max > w_img or y_max > h_img:
                        warnings.append(
                            f"[C-06] Document '{doc_id}' line {line_idx}: polygon extends outside image frame "
                            f"(box=[{x_min:.1f}, {y_min:.1f}, {x_max:.1f}, {y_max:.1f}], image={w_img}x{h_img})"
                        )

    # C-07, C-08: KIE checks
    c07_passed = True
    c08_passed = True
    required_kie_keys = ("company", "date", "address", "total")

    for split in splits:
        for doc_id, ent_path in sorted(split_file_maps[split]["entities"].items()):
            # C-07: Exists and valid JSON
            if not ent_path.exists() or ent_path.stat().st_size == 0:
                c07_passed = False
                errors.append(f"[C-07] KIE annotation file is missing or empty: '{ent_path}'")
                continue

            try:
                with open(ent_path, "r", encoding="utf-8-sig") as f:
                    data = json.load(f)
            except Exception as e:
                c07_passed = False
                errors.append(f"[C-07] Malformed KIE JSON in '{ent_path}': {e}")
                continue

            if not isinstance(data, dict):
                c07_passed = False
                errors.append(f"[C-07] KIE root in '{ent_path}' is not a JSON object")
                continue

            # C-08: Required keys and string values
            missing_keys = [k for k in required_kie_keys if k not in data]
            if missing_keys:
                c08_passed = False
                errors.append(f"[C-08] Document '{doc_id}' KIE missing keys: {missing_keys}")

            for k in required_kie_keys:
                if k in data and not isinstance(data[k], str):
                    c08_passed = False
                    errors.append(
                        f"[C-08] Document '{doc_id}' KIE key '{k}' must be string, got {type(data[k]).__name__}"
                    )

    # Document counts evaluation
    paired_train_count = len(paired_ids.get("train", set()))
    paired_test_count = len(paired_ids.get("test", set()))

    if strict_counts:
        if paired_train_count != expected_train_count:
            errors.append(
                f"Train split paired count mismatch: expected {expected_train_count}, got {paired_train_count}"
            )
        if paired_test_count != expected_test_count:
            errors.append(
                f"Test split paired count mismatch: expected {expected_test_count}, got {paired_test_count}"
            )
    else:
        if paired_train_count != expected_train_count:
            warnings.append(
                f"Train split paired count {paired_train_count} differs from canonical {expected_train_count} (subset mode)"
            )
        if paired_test_count != expected_test_count:
            warnings.append(
                f"Test split paired count {paired_test_count} differs from canonical {expected_test_count} (subset mode)"
            )

    checks = {
        "C-01": {
            "name": "image_file_exists",
            "passed": c01_passed,
            "status": "PASSED" if c01_passed else "FAILED",
        },
        "C-02": {
            "name": "raster_readability",
            "passed": c02_passed,
            "status": "PASSED" if c02_passed else "FAILED",
        },
        "C-03": {
            "name": "geometric_dimensions",
            "passed": c03_passed,
            "status": "PASSED" if c03_passed else "FAILED",
        },
        "C-04": {
            "name": "ocr_file_exists",
            "passed": c04_passed,
            "status": "PASSED" if c04_passed else "FAILED",
        },
        "C-05": {
            "name": "ocr_polygons_valid",
            "passed": c05_passed,
            "status": "PASSED" if c05_passed else "FAILED",
        },
        "C-06": {
            "name": "bbox_geometry_bounds",
            "passed": c06_passed,
            "status": "PASSED" if c06_passed else "FAILED",
        },
        "C-07": {
            "name": "kie_file_exists_and_valid_json",
            "passed": c07_passed,
            "status": "PASSED" if c07_passed else "FAILED",
        },
        "C-08": {
            "name": "kie_required_keys_and_string_values",
            "passed": c08_passed,
            "status": "PASSED" if c08_passed else "FAILED",
        },
        "C-09": {
            "name": "id_uniqueness_and_split_isolation",
            "passed": c09_passed,
            "status": "PASSED" if c09_passed else "FAILED",
        },
        "C-10": {
            "name": "cross_file_triplet_consistency",
            "passed": c10_passed,
            "status": "PASSED" if c10_passed else "FAILED",
        },
    }

    report = {
        "dataset": "sroie-datasetv2",
        "train_count": paired_train_count,
        "test_count": paired_test_count,
        "expected_train_count": expected_train_count,
        "expected_test_count": expected_test_count,
        "summary": {
            "train_images": len(split_ids.get("train", {}).get("img", set())),
            "train_ocr": len(split_ids.get("train", {}).get("box", set())),
            "train_kie": len(split_ids.get("train", {}).get("entities", set())),
            "test_images": len(split_ids.get("test", {}).get("img", set())),
            "test_ocr": len(split_ids.get("test", {}).get("box", set())),
            "test_kie": len(split_ids.get("test", {}).get("entities", set())),
            "paired_train_documents": paired_train_count,
            "paired_test_documents": paired_test_count,
        },
        "checks": checks,
        "split_discrepancies": discrepancies,
        "errors": errors,
        "warnings": warnings,
    }

    return report


def main() -> int:
    """CLI entrypoint for SROIE dataset integrity audit."""
    parser = argparse.ArgumentParser(
        description="Run SROIE dataset integrity audit (checks C-01 to C-10)."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Path to SROIE dataset root directory.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sroie.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/audit/sroie_integrity.json",
        help="Path to output JSON audit report.",
    )
    parser.add_argument(
        "--strict-counts",
        action="store_true",
        help="Fail audit if document counts do not exactly match 626 train / 347 test.",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_cfg = cfg.get("dataset", {})

    root_dir = args.root or dataset_cfg.get("root_dir")
    if not root_dir:
        print(
            "Error: Dataset root must be specified via --root or defined in config file.",
            file=sys.stderr,
        )
        return 1

    expected_train = dataset_cfg.get("expected_train_count", 626)
    expected_test = dataset_cfg.get("expected_test_count", 347)
    min_area = dataset_cfg.get("min_dimension_pixels", 100) ** 2

    print("=" * 60)
    print("SROIE DATASET INTEGRITY AUDIT")
    print("=" * 60)
    print(f"Target Root: {root_dir}")
    print(f"Strict Counts: {args.strict_counts}")
    print("-" * 60)

    try:
        report = audit_sroie_dataset(
            root_dir=root_dir,
            expected_train_count=expected_train,
            expected_test_count=expected_test,
            strict_counts=args.strict_counts,
            min_area_pixels=min_area,
        )
    except Exception as e:
        print(f"FATAL AUDIT ERROR: {e}", file=sys.stderr)
        return 1

    # Print summary to console
    summary = report["summary"]
    print("Document Counts:")
    print(f"  Train Images: {summary['train_images']}")
    print(f"  Train OCR:    {summary['train_ocr']}")
    print(f"  Train KIE:    {summary['train_kie']}")
    print(f"  Paired Train: {summary['paired_train_documents']} (Expected: {expected_train})")
    print(f"  Test Images:  {summary['test_images']}")
    print(f"  Test OCR:     {summary['test_ocr']}")
    print(f"  Test KIE:     {summary['test_kie']}")
    print(f"  Paired Test:  {summary['paired_test_documents']} (Expected: {expected_test})")
    print("-" * 60)

    print("Integrity Checks:")
    for code, info in report["checks"].items():
        print(f"  [{code}] {info['name']:<35} : {info['status']}")

    print("-" * 60)
    print(f"Errors   : {len(report['errors'])}")
    print(f"Warnings : {len(report['warnings'])}")

    if report["errors"]:
        print("\nErrors encountered:")
        for err in report["errors"][:10]:
            print(f"  ! {err}")
        if len(report["errors"]) > 10:
            print(f"  ... and {len(report['errors']) - 10} more errors.")

    if report["warnings"]:
        print("\nWarnings:")
        for warn in report["warnings"][:5]:
            print(f"  * {warn}")
        if len(report["warnings"]) > 5:
            print(f"  ... and {len(report['warnings']) - 5} more warnings.")

    # Save machine-readable JSON report
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nAudit report saved to: {out_path.resolve()}")
    print("=" * 60)

    if report["errors"]:
        print("AUDIT FAILED with errors.")
        return 1
    else:
        print("AUDIT PASSED successfully.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
