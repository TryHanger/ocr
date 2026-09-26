"""Unit tests for scripts/audit_sroie.py dataset integrity audit."""

import json
from pathlib import Path
import pytest

from scripts.audit_sroie import audit_sroie_dataset, load_config, main


FIXTURE_ROOT = Path("tests/fixtures/sroie")
VALID_FIXTURE_ROOT = Path("tests/fixtures/sroie_valid")


def test_audit_valid_dataset():
    report = audit_sroie_dataset(VALID_FIXTURE_ROOT)
    assert report["dataset"] == "sroie-datasetv2"
    assert report["mode"] == "synthetic_fixture"
    assert VALID_FIXTURE_ROOT.as_posix() in report["source"]
    assert len(report["errors"]) == 0

    # Checks C-01 to C-10 all PASSED
    for code in (
        "C-01",
        "C-02",
        "C-03",
        "C-04",
        "C-05",
        "C-06",
        "C-07",
        "C-08",
        "C-09",
        "C-10",
    ):
        assert report["checks"][code]["passed"] is True
        assert report["checks"][code]["status"] == "PASSED"

    summary = report["summary"]
    assert summary["paired_train_documents"] == 1
    assert summary["paired_test_documents"] == 1


def test_audit_corrupted_dataset():
    report = audit_sroie_dataset(FIXTURE_ROOT)
    assert len(report["errors"]) > 0

    # C-05 (malformed OCR line) and C-08 (missing KIE keys) and C-10 (missing pair) failed
    assert report["checks"]["C-05"]["passed"] is False
    assert report["checks"]["C-08"]["passed"] is False
    assert report["checks"]["C-10"]["passed"] is False

    # Out of bounds should generate warning in C-06
    assert any("[C-06]" in w for w in report["warnings"])
    assert "doc_missing_pair" in report["split_discrepancies"]["train"]["missing_ocr"]


def test_audit_missing_root():
    with pytest.raises(FileNotFoundError, match="Dataset root directory does not exist"):
        audit_sroie_dataset("non/existent/root")


def test_audit_split_isolation_leakage(tmp_path: Path):
    root = tmp_path / "leakage_dataset"
    for split in ("train", "test"):
        (root / split / "img").mkdir(parents=True)
        (root / split / "box").mkdir(parents=True)
        (root / split / "entities").mkdir(parents=True)

    # Same doc_leak in train and test
    (root / "train" / "box" / "doc_leak.txt").write_text("10,10,20,10,20,20,10,20,T\n", encoding="utf-8")
    (root / "test" / "box" / "doc_leak.txt").write_text("10,10,20,10,20,20,10,20,T\n", encoding="utf-8")

    report = audit_sroie_dataset(root)
    assert report["checks"]["C-09"]["passed"] is False
    assert any("[C-09]" in err for err in report["errors"])


def test_audit_strict_vs_subset_counts():
    # In strict mode, if train != 626 or test != 347, errors should be recorded
    report_strict = audit_sroie_dataset(VALID_FIXTURE_ROOT, strict_counts=True)
    assert len(report_strict["errors"]) > 0
    assert any("paired count mismatch" in err for err in report_strict["errors"])
    assert report_strict["mode"] == "canonical_strict"

    # In subset mode, smaller document counts generate warnings, NOT errors
    report_subset = audit_sroie_dataset(VALID_FIXTURE_ROOT, strict_counts=False, mode="synthetic_fixture")
    assert len(report_subset["errors"]) == 0
    assert any("differs from canonical" in w for w in report_subset["warnings"])
    assert report_subset["mode"] == "synthetic_fixture"


def test_load_config_custom(tmp_path: Path):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("dataset:\n  expected_train_count: 500\n", encoding="utf-8")
    cfg = load_config(cfg_file)
    assert cfg["dataset"]["expected_train_count"] == 500


def test_load_config_nonexistent():
    cfg = load_config("nonexistent/cfg.yaml")
    assert cfg == {}


def test_cli_main_valid(monkeypatch, tmp_path: Path):
    out_file = tmp_path / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "audit_sroie.py",
            "--root",
            str(VALID_FIXTURE_ROOT),
            "--output",
            str(out_file),
            "--mode",
            "synthetic_fixture",
        ],
    )
    code = main()
    assert code == 0
    assert out_file.exists()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["dataset"] == "sroie-datasetv2"
    assert data["mode"] == "synthetic_fixture"
    assert VALID_FIXTURE_ROOT.as_posix() in data["source"]
    assert len(data["errors"]) == 0


def test_cli_main_corrupted(monkeypatch, tmp_path: Path):
    out_file = tmp_path / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "audit_sroie.py",
            "--root",
            str(FIXTURE_ROOT),
            "--output",
            str(out_file),
        ],
    )
    code = main()
    assert code == 1
    assert out_file.exists()


def test_cli_main_no_root(monkeypatch):
    monkeypatch.setattr("sys.argv", ["audit_sroie.py", "--config", "nonexistent.yaml"])
    code = main()
    assert code == 1


def test_audit_missing_split_and_subdirs(tmp_path: Path):
    root = tmp_path / "broken_layout"
    (root / "train").mkdir(parents=True)
    # train has no img/box/entities subdirs, and no test dir at all
    report = audit_sroie_dataset(root)
    assert any("Missing split directory" in err for err in report["errors"])
    assert any("Missing 'img' directory" in err for err in report["errors"])


def test_audit_image_failures(monkeypatch, tmp_path: Path):
    root = tmp_path / "img_failures"
    for split in ("train", "test"):
        (root / split / "img").mkdir(parents=True)
        (root / split / "box").mkdir(parents=True)
        (root / split / "entities").mkdir(parents=True)

    # Empty image file (C-01)
    (root / "train" / "img" / "empty.jpg").touch()

    # Corrupted image bytes (C-02)
    (root / "train" / "img" / "corrupted.jpg").write_bytes(b"invalid data")

    # Image with non-positive dimensions (C-03)
    # Mock cv2.imread returning shape with 0 height
    import cv2
    import numpy as np
    valid_stub = root / "train" / "img" / "bad_dims.jpg"
    cv2.imwrite(str(valid_stub), np.zeros((10, 10, 3), dtype=np.uint8))
    monkeypatch.setattr(cv2, "imread", lambda p: np.zeros((0, 0, 3), dtype=np.uint8) if "bad_dims" in str(p) else None)

    report = audit_sroie_dataset(root)
    assert report["checks"]["C-01"]["passed"] is False
    assert report["checks"]["C-02"]["passed"] is False
    assert report["checks"]["C-03"]["passed"] is False


def test_audit_small_valid_images_not_rejected(tmp_path: Path):
    # Small image (e.g. 20x30 = 600 px area, well below 10,000) must NOT be rejected
    import cv2
    import numpy as np

    root = tmp_path / "small_valid"
    for split in ("train", "test"):
        (root / split / "img").mkdir(parents=True)
        (root / split / "box").mkdir(parents=True)
        (root / split / "entities").mkdir(parents=True)

    small_img = np.full((30, 20, 3), 200, dtype=np.uint8)
    cv2.imwrite(str(root / "train" / "img" / "small_doc.jpg"), small_img)
    (root / "train" / "box" / "small_doc.txt").write_text("1,1,10,1,10,10,1,10,SMALL\n", encoding="utf-8")
    (root / "train" / "entities" / "small_doc.txt").write_text(
        json.dumps({"company": "S", "date": "D", "address": "A", "total": "T"}),
        encoding="utf-8",
    )

    report = audit_sroie_dataset(root)
    assert report["checks"]["C-03"]["passed"] is True


def test_audit_ocr_failures_including_more_than_eight_coords(tmp_path: Path):
    root = tmp_path / "ocr_failures"
    for split in ("train", "test"):
        (root / split / "img").mkdir(parents=True)
        (root / split / "box").mkdir(parents=True)
        (root / split / "entities").mkdir(parents=True)

    # Empty OCR file (C-04)
    (root / "train" / "box" / "empty_box.txt").touch()

    # Blank content OCR file (C-04)
    (root / "train" / "box" / "blank_box.txt").write_text("   \n\n", encoding="utf-8")

    # Fewer than 8 coordinates (C-05)
    (root / "train" / "box" / "bad_few.txt").write_text("10,20,30,TEXT\n", encoding="utf-8")

    # Non-numeric coordinate (C-05)
    (root / "train" / "box" / "bad_num.txt").write_text("10,20,abc,40,50,60,70,80,TEXT\n", encoding="utf-8")

    # Non-finite coordinate (C-05)
    (root / "train" / "box" / "bad_inf.txt").write_text("10,20,inf,40,50,60,70,80,TEXT\n", encoding="utf-8")

    # More than 8 coordinates (C-05)
    (root / "train" / "box" / "bad_many.txt").write_text(
        "10,20,30,40,50,60,70,80,90,100,TEXT\n", encoding="utf-8"
    )

    # Inverted bbox x_max < x_min (C-06)
    (root / "train" / "box" / "bad_geom.txt").write_text("50,10,20,10,20,20,50,20,TEXT\n", encoding="utf-8")

    report = audit_sroie_dataset(root)
    assert report["checks"]["C-04"]["passed"] is False
    assert report["checks"]["C-05"]["passed"] is False
    assert any("line contains more than 8 coordinates" in err for err in report["errors"])


def test_audit_kie_failures(tmp_path: Path):
    root = tmp_path / "kie_failures"
    for split in ("train", "test"):
        (root / split / "img").mkdir(parents=True)
        (root / split / "box").mkdir(parents=True)
        (root / split / "entities").mkdir(parents=True)

    # Empty KIE file (C-07)
    (root / "train" / "entities" / "empty_kie.txt").touch()

    # Malformed JSON (C-07)
    (root / "train" / "entities" / "malformed.txt").write_text("{invalid json", encoding="utf-8")

    # Non-dict JSON (C-07)
    (root / "train" / "entities" / "array.txt").write_text("[\"test\"]", encoding="utf-8")

    # Non-string value for required key (C-08)
    (root / "train" / "entities" / "bad_type.txt").write_text(
        json.dumps({"company": 123, "date": "01/01/2020", "address": "city", "total": "100"}),
        encoding="utf-8"
    )

    report = audit_sroie_dataset(root)
    assert report["checks"]["C-07"]["passed"] is False
    assert report["checks"]["C-08"]["passed"] is False


def test_cli_main_exception_handling(monkeypatch):
    monkeypatch.setattr("sys.argv", ["audit_sroie.py", "--root", "non/existent/root"])
    code = main()
    assert code == 1


