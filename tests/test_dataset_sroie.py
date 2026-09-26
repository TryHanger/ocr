"""Unit tests for SROIEAdapter dataset adapter."""

import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from src.datasets.sroie import SROIEAdapter
from src.core.schemas import DocumentMetadata, OCRGroundTruth, KIEGroundTruth


FIXTURE_ROOT = Path("tests/fixtures/sroie")
VALID_FIXTURE_ROOT = Path("tests/fixtures/sroie_valid")


def test_init_valid_roots():
    adapter = SROIEAdapter(FIXTURE_ROOT)
    assert adapter.root_dir == FIXTURE_ROOT.resolve()

    adapter_valid = SROIEAdapter(VALID_FIXTURE_ROOT)
    assert adapter_valid.root_dir == VALID_FIXTURE_ROOT.resolve()


def test_init_nested_sroie2019_dir(tmp_path: Path):
    nested_root = tmp_path / "dataset"
    (nested_root / "SROIE2019" / "train" / "img").mkdir(parents=True)
    adapter = SROIEAdapter(nested_root)
    assert adapter.root_dir == (nested_root / "SROIE2019").resolve()


def test_init_nonexistent_root():
    with pytest.raises(FileNotFoundError, match="Dataset root directory does not exist"):
        SROIEAdapter("non/existent/path")


def test_list_document_ids_deterministic_and_clean():
    adapter = SROIEAdapter(FIXTURE_ROOT)
    train_ids = adapter.list_document_ids("train")
    test_ids = adapter.list_document_ids("test")

    # Deterministic sorted order
    assert train_ids == sorted(train_ids)
    assert test_ids == sorted(test_ids)

    # Document IDs must have no extension
    for doc_id in train_ids + test_ids:
        assert not doc_id.endswith(".jpg")
        assert not doc_id.endswith(".txt")
        assert not doc_id.endswith(".json")

    # Expected IDs
    assert "doc_valid_1" in train_ids
    assert "doc_valid_2" in train_ids
    assert "doc_test_1" in test_ids
    assert "doc_test_2" in test_ids


def test_list_document_ids_invalid_split():
    adapter = SROIEAdapter(FIXTURE_ROOT)
    with pytest.raises(ValueError, match="Unsupported split 'val'"):
        adapter.list_document_ids("val")
    with pytest.raises(ValueError, match="Unsupported split 'validation'"):
        adapter.list_document_ids("validation")


def test_list_document_ids_missing_split_directory(tmp_path: Path):
    root = tmp_path / "empty_dataset"
    root.mkdir()
    adapter = SROIEAdapter(root)
    with pytest.raises(FileNotFoundError, match="Split directory 'train' not found"):
        adapter.list_document_ids("train")


def test_get_metadata_valid():
    adapter = SROIEAdapter(FIXTURE_ROOT)
    meta = adapter.get_metadata("doc_valid_1")
    assert isinstance(meta, DocumentMetadata)
    assert meta.document_id == "doc_valid_1"
    assert meta.width == 120
    assert meta.height == 120
    assert meta.split == "train"
    assert "doc_valid_1.jpg" in meta.image_path


def test_get_metadata_nonexistent_doc():
    adapter = SROIEAdapter(FIXTURE_ROOT)
    with pytest.raises(KeyError, match="Document 'unknown_doc' not found"):
        adapter.get_metadata("unknown_doc")


def test_get_image_valid_rgb():
    adapter = SROIEAdapter(FIXTURE_ROOT)
    img = adapter.get_image("doc_valid_1")
    assert isinstance(img, np.ndarray)
    assert img.dtype == np.uint8
    assert img.shape == (120, 120, 3)


def test_get_image_missing_file(tmp_path: Path):
    root = tmp_path / "dataset"
    (root / "train" / "img").mkdir(parents=True)
    # create doc in index without actual image file
    adapter = SROIEAdapter(root)
    adapter._index["ghost_doc"] = {
        "split": "train",
        "img_path": root / "train" / "img" / "ghost_doc.jpg",
        "box_path": None,
        "entities_path": None,
    }
    with pytest.raises(FileNotFoundError, match="Image file for document 'ghost_doc' not found"):
        adapter.get_image("ghost_doc")


def test_get_image_corrupted_or_invalid_dimensions(tmp_path: Path):
    root = tmp_path / "dataset"
    (root / "train" / "img").mkdir(parents=True)
    corrupted_img = root / "train" / "img" / "corrupted.jpg"
    corrupted_img.write_bytes(b"not an image")

    adapter = SROIEAdapter(root)
    with pytest.raises(ValueError, match="could not be read or is empty"):
        adapter.get_image("corrupted")


def test_get_ocr_ground_truth_valid_and_immutability():
    adapter = SROIEAdapter(FIXTURE_ROOT)
    ocr_gt = adapter.get_ocr_ground_truth("doc_valid_1")

    assert isinstance(ocr_gt, OCRGroundTruth)
    assert ocr_gt.document_id == "doc_valid_1"
    assert len(ocr_gt.tokens) == 4

    # Raw transcript immutability checks
    expected_lines = [
        "BOOK STORE SDN BHD",
        "25/12/2018",
        "KUALA LUMPUR, MALAYSIA",
        "TOTAL 45.00",
    ]
    for token, expected_text in zip(ocr_gt.tokens, expected_lines):
        # Must preserve exact casing, punctuation (comma, slash), spacing
        assert token.text == expected_text
        assert token.bbox.width > 0
        assert token.bbox.height > 0

    assert ocr_gt.text == "\n".join(expected_lines)
    # Check polygons preserved in metadata
    assert "polygons" in ocr_gt.metadata
    assert len(ocr_gt.metadata["polygons"]) == 4
    assert ocr_gt.metadata["polygons"][0] == [10.0, 10.0, 100.0, 10.0, 100.0, 30.0, 10.0, 30.0]


def test_get_ocr_ground_truth_out_of_bounds_handling():
    adapter = SROIEAdapter(FIXTURE_ROOT)
    ocr_gt = adapter.get_ocr_ground_truth("doc_oob")
    # Coordinates in file: -10, 10, 150, 10, 150, 40, -10, 40
    # Must preserve exact raw unclipped coordinates: source coordinates == adapter coordinates
    token = ocr_gt.tokens[0]
    assert token.bbox.x_min == -10.0
    assert token.bbox.x_max == 150.0
    assert token.bbox.y_min == 10.0
    assert token.bbox.y_max == 40.0
    # Raw polygon preserved verbatim
    assert ocr_gt.metadata["polygons"][0] == [-10.0, 10.0, 150.0, 10.0, 150.0, 40.0, -10.0, 40.0]


def test_get_ocr_ground_truth_fewer_than_eight_coords_rejected(tmp_path: Path):
    root = tmp_path / "dataset"
    (root / "train" / "box").mkdir(parents=True)
    # 7 coordinates + text
    (root / "train" / "box" / "few_coords.txt").write_text(
        "10,20,30,40,50,60,70,TEXT\n", encoding="utf-8"
    )
    adapter = SROIEAdapter(root)
    with pytest.raises(ValueError, match="expected exactly 8 coordinates"):
        adapter.get_ocr_ground_truth("few_coords")


def test_get_ocr_ground_truth_more_than_eight_coords_rejected(tmp_path: Path):
    root = tmp_path / "dataset"
    (root / "train" / "box").mkdir(parents=True)
    # 10 coordinates + text
    (root / "train" / "box" / "many_coords.txt").write_text(
        "10,20,30,40,50,60,70,80,90,100,TEXT\n", encoding="utf-8"
    )
    adapter = SROIEAdapter(root)
    with pytest.raises(ValueError, match="line contains more than 8 coordinates"):
        adapter.get_ocr_ground_truth("many_coords")


def test_get_ocr_ground_truth_exactly_eight_coords_accepted(tmp_path: Path):
    root = tmp_path / "dataset"
    (root / "train" / "box").mkdir(parents=True)
    # Exactly 8 coordinates + text (and text containing commas)
    (root / "train" / "box" / "exact_coords.txt").write_text(
        "10,20,30,20,30,40,10,40,CITY, COUNTRY, 12345\n", encoding="utf-8"
    )
    adapter = SROIEAdapter(root)
    gt = adapter.get_ocr_ground_truth("exact_coords")
    assert len(gt.tokens) == 1
    assert gt.tokens[0].text == "CITY, COUNTRY, 12345"
    assert gt.tokens[0].bbox.x_min == 10.0
    assert gt.tokens[0].bbox.x_max == 30.0


def test_get_ocr_ground_truth_missing_file():
    adapter = SROIEAdapter(FIXTURE_ROOT)
    # doc_missing_pair has image and KIE, but no OCR box file
    with pytest.raises(FileNotFoundError, match="OCR annotation file for document 'doc_missing_pair' not found"):
        adapter.get_ocr_ground_truth("doc_missing_pair")


def test_get_ocr_ground_truth_empty_file(tmp_path: Path):
    root = tmp_path / "dataset"
    (root / "train" / "box").mkdir(parents=True)
    (root / "train" / "box" / "empty_doc.txt").write_text("", encoding="utf-8")
    adapter = SROIEAdapter(root)
    with pytest.raises(ValueError, match="OCR annotation file for document 'empty_doc' is empty"):
        adapter.get_ocr_ground_truth("empty_doc")




def test_get_ocr_ground_truth_non_numeric_coords(tmp_path: Path):
    root = tmp_path / "dataset"
    (root / "train" / "box").mkdir(parents=True)
    (root / "train" / "box" / "non_num.txt").write_text(
        "10,20,abc,40,50,60,70,80,TEXT\n", encoding="utf-8"
    )
    adapter = SROIEAdapter(root)
    with pytest.raises(ValueError, match="Non-numeric OCR coordinate"):
        adapter.get_ocr_ground_truth("non_num")


def test_get_ocr_ground_truth_non_finite_coords(tmp_path: Path):
    root = tmp_path / "dataset"
    (root / "train" / "box").mkdir(parents=True)
    (root / "train" / "box" / "inf_doc.txt").write_text(
        "10,20,inf,40,50,60,70,80,TEXT\n", encoding="utf-8"
    )
    adapter = SROIEAdapter(root)
    with pytest.raises(ValueError, match="Non-finite OCR coordinate"):
        adapter.get_ocr_ground_truth("inf_doc")


def test_get_kie_ground_truth_valid_and_immutability():
    adapter = SROIEAdapter(FIXTURE_ROOT)
    kie_gt = adapter.get_kie_ground_truth("doc_valid_1")

    assert isinstance(kie_gt, KIEGroundTruth)
    assert kie_gt.document_id == "doc_valid_1"

    # Raw string immutability checks: no lowercase, no float parsing, no date conversion
    assert kie_gt.fields["company"] == "BOOK STORE SDN BHD"
    assert kie_gt.fields["date"] == "25/12/2018"
    assert kie_gt.fields["address"] == "KUALA LUMPUR, MALAYSIA"
    assert kie_gt.fields["total"] == "45.00"  # string, NOT float 45.0


def test_get_kie_ground_truth_supports_json_extension():
    adapter = SROIEAdapter(FIXTURE_ROOT)
    kie_gt = adapter.get_kie_ground_truth("doc_valid_2")
    assert kie_gt.fields["company"] == "RESTAURANT XYZ"
    assert kie_gt.fields["total"] == "RM 120.50"


def test_get_kie_ground_truth_missing_keys():
    adapter = SROIEAdapter(FIXTURE_ROOT)
    # doc_bad_kie is missing "address" and "total"
    with pytest.raises(ValueError, match="missing required keys"):
        adapter.get_kie_ground_truth("doc_bad_kie")


def test_get_kie_ground_truth_malformed_json(tmp_path: Path):
    root = tmp_path / "dataset"
    (root / "train" / "entities").mkdir(parents=True)
    (root / "train" / "entities" / "bad_json.txt").write_text("{not a json}", encoding="utf-8")
    adapter = SROIEAdapter(root)
    with pytest.raises(ValueError, match="Malformed KIE JSON"):
        adapter.get_kie_ground_truth("bad_json")


def test_get_kie_ground_truth_non_dict_json(tmp_path: Path):
    root = tmp_path / "dataset"
    (root / "train" / "entities").mkdir(parents=True)
    (root / "train" / "entities" / "array_json.txt").write_text("[\"item\"]", encoding="utf-8")
    adapter = SROIEAdapter(root)
    with pytest.raises(ValueError, match="must be a JSON object"):
        adapter.get_kie_ground_truth("array_json")


def test_get_kie_ground_truth_non_string_values(tmp_path: Path):
    root = tmp_path / "dataset"
    (root / "train" / "entities").mkdir(parents=True)
    (root / "train" / "entities" / "num_field.txt").write_text(
        json.dumps({
            "company": "SHOP",
            "date": "01/01/2020",
            "address": "CITY",
            "total": 50.0  # numeric instead of string
        }),
        encoding="utf-8"
    )
    adapter = SROIEAdapter(root)
    with pytest.raises(TypeError, match="must be str"):
        adapter.get_kie_ground_truth("num_field")


def test_get_metadata_missing_image_file(tmp_path: Path):
    root = tmp_path / "dataset"
    (root / "train" / "box").mkdir(parents=True)
    (root / "train" / "box" / "orphan_doc.txt").write_text("10,10,20,10,20,20,10,20,T\n", encoding="utf-8")
    adapter = SROIEAdapter(root)
    with pytest.raises(FileNotFoundError, match="Image file for document 'orphan_doc' not found"):
        adapter.get_metadata("orphan_doc")


def test_get_metadata_unreadable_image(tmp_path: Path):
    root = tmp_path / "dataset"
    (root / "train" / "img").mkdir(parents=True)
    (root / "train" / "img" / "bad_img.jpg").write_bytes(b"garbage")
    adapter = SROIEAdapter(root)
    with pytest.raises(ValueError, match="Failed to read image for document 'bad_img'"):
        adapter.get_metadata("bad_img")


def test_get_image_invalid_dtype_or_channels(monkeypatch, tmp_path: Path):
    root = tmp_path / "dataset"
    (root / "train" / "img").mkdir(parents=True)
    img_path = root / "train" / "img" / "test.jpg"
    img_path.write_bytes(b"placeholder")
    adapter = SROIEAdapter(root)

    # Mock cv2.imread returning float64
    monkeypatch.setattr(cv2, "imread", lambda *args, **kwargs: np.zeros((10, 10, 3), dtype=np.float32))
    with pytest.raises(TypeError, match="Image dtype must be uint8"):
        adapter.get_image("test")

    # Mock cv2.imread returning 2-channel or 1-channel (grayscale)
    monkeypatch.setattr(cv2, "imread", lambda *args, **kwargs: np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError, match="Image must have 3 channels"):
        adapter.get_image("test")


def test_get_ocr_ground_truth_skips_blank_lines(tmp_path: Path):
    root = tmp_path / "dataset"
    (root / "train" / "box").mkdir(parents=True)
    (root / "train" / "box" / "blank_lines.txt").write_text(
        "10,10,20,10,20,20,10,20,LINE1\n\n   \n10,30,20,30,20,40,10,40,LINE2\n",
        encoding="utf-8",
    )
    adapter = SROIEAdapter(root)
    gt = adapter.get_ocr_ground_truth("blank_lines")
    assert len(gt.tokens) == 2
    assert gt.text == "LINE1\nLINE2"


def test_get_kie_ground_truth_missing_entities_file(tmp_path: Path):
    root = tmp_path / "dataset"
    (root / "train" / "box").mkdir(parents=True)
    (root / "train" / "box" / "no_kie.txt").write_text("10,10,20,10,20,20,10,20,T\n", encoding="utf-8")
    adapter = SROIEAdapter(root)
    with pytest.raises(FileNotFoundError, match="KIE annotation file for document 'no_kie' not found"):
        adapter.get_kie_ground_truth("no_kie")

