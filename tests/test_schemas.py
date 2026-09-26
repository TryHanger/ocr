"""Unit tests for core schemas, DTOs, validation and serialization."""

import pytest
from src.core.schemas import (
    BoundingBox,
    DegradationSpec,
    DocumentMetadata,
    ExperimentResult,
    KIEGroundTruth,
    KIEResult,
    OCRGroundTruth,
    OCRResult,
    OCRToken,
    PreprocessingSpec,
)


# ============================================================================
# BoundingBox Tests
# ============================================================================

def test_bounding_box_valid():
    bbox = BoundingBox(x_min=10, y_min=20, x_max=50, y_max=60)
    assert bbox.x_min == 10
    assert bbox.y_min == 20
    assert bbox.x_max == 50
    assert bbox.y_max == 60
    assert bbox.width == 40
    assert bbox.height == 40
    assert bbox.area == 1600


def test_bounding_box_float_valid():
    bbox = BoundingBox(x_min=0.5, y_min=1.5, x_max=10.5, y_max=21.5)
    assert bbox.width == 10.0
    assert bbox.height == 20.0
    assert bbox.area == 200.0


def test_bounding_box_zero_area():
    bbox = BoundingBox(x_min=5, y_min=5, x_max=5, y_max=5)
    assert bbox.width == 0
    assert bbox.height == 0
    assert bbox.area == 0


@pytest.mark.parametrize("x_min,y_min,x_max,y_max", [
    (-1, 0, 10, 10),
    (0, -1, 10, 10),
    (15, 0, 10, 10),   # x_max < x_min
    (0, 25, 10, 20),   # y_max < y_min
])
def test_bounding_box_invalid_coords(x_min, y_min, x_max, y_max):
    with pytest.raises(ValueError):
        BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


@pytest.mark.parametrize("val", ["10", True, False, None, float("nan"), float("inf")])
def test_bounding_box_invalid_types(val):
    with pytest.raises((TypeError, ValueError)):
        BoundingBox(x_min=val, y_min=0, x_max=10, y_max=10)


def test_bounding_box_serialization():
    bbox = BoundingBox(x_min=10.0, y_min=20.0, x_max=30.0, y_max=40.0)
    data = bbox.to_dict()
    assert data == {"x_min": 10.0, "y_min": 20.0, "x_max": 30.0, "y_max": 40.0}
    restored = BoundingBox.from_dict(data)
    assert restored == bbox


def test_bounding_box_from_dict_missing_keys():
    with pytest.raises(KeyError):
        BoundingBox.from_dict({"x_min": 0, "y_min": 0, "x_max": 10})


# ============================================================================
# OCRToken Tests
# ============================================================================

def test_ocr_token_valid():
    bbox = BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10)
    token = OCRToken(text="Receipt", bbox=bbox, confidence=0.95)
    assert token.text == "Receipt"
    assert token.bbox == bbox
    assert token.confidence == 0.95


def test_ocr_token_none_confidence():
    bbox = BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10)
    token = OCRToken(text="Total", bbox=bbox, confidence=None)
    assert token.confidence is None


def test_ocr_token_auto_convert_bbox_dict():
    token = OCRToken(text="Date", bbox={"x_min": 1, "y_min": 2, "x_max": 3, "y_max": 4})
    assert isinstance(token.bbox, BoundingBox)
    assert token.bbox.x_min == 1


def test_ocr_token_invalid_text():
    bbox = BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10)
    with pytest.raises(TypeError):
        OCRToken(text=123, bbox=bbox)


def test_ocr_token_invalid_bbox_type():
    with pytest.raises(TypeError):
        OCRToken(text="hello", bbox="not-a-bbox")


@pytest.mark.parametrize("conf", [-0.1, 1.01, "0.5", True, float("nan")])
def test_ocr_token_invalid_confidence(conf):
    bbox = BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10)
    with pytest.raises((ValueError, TypeError)):
        OCRToken(text="Tax", bbox=bbox, confidence=conf)


def test_ocr_token_serialization():
    token = OCRToken(text="Total", bbox=BoundingBox(0, 0, 10, 10), confidence=0.88)
    data = token.to_dict()
    assert data["text"] == "Total"
    assert data["confidence"] == 0.88
    assert data["bbox"]["x_min"] == 0

    restored = OCRToken.from_dict(data)
    assert restored.text == token.text
    assert restored.confidence == token.confidence
    assert restored.bbox == token.bbox


def test_ocr_token_from_dict_missing_keys():
    with pytest.raises(KeyError):
        OCRToken.from_dict({"text": "test"})


# ============================================================================
# OCRResult Tests
# ============================================================================

def test_ocr_result_valid_empty_tokens():
    res = OCRResult(document_id="doc_1", full_text="", tokens=[])
    assert res.document_id == "doc_1"
    assert res.tokens == []
    assert res.processing_time_ms is None


def test_ocr_result_valid_with_tokens_and_metadata():
    bbox = BoundingBox(0, 0, 10, 10)
    t1 = OCRToken("Hello", bbox, 0.9)
    res = OCRResult(
        document_id="doc_2",
        full_text="Hello",
        tokens=[t1, {"text": "World", "bbox": bbox.to_dict(), "confidence": 0.8}],
        metadata={"dpi": 300},
        processing_time_ms=125.5,
        model_name="PaddleOCR",
        model_version="2.7.0",
    )
    assert len(res.tokens) == 2
    assert isinstance(res.tokens[1], OCRToken)
    assert res.processing_time_ms == 125.5
    assert res.model_name == "PaddleOCR"


@pytest.mark.parametrize("doc_id", ["", "   ", None, 123])
def test_ocr_result_invalid_document_id(doc_id):
    with pytest.raises((ValueError, TypeError)):
        OCRResult(document_id=doc_id, full_text="Text", tokens=[])


def test_ocr_result_invalid_full_text():
    with pytest.raises(TypeError):
        OCRResult(document_id="doc_1", full_text=None, tokens=[])


def test_ocr_result_invalid_tokens_type():
    with pytest.raises(TypeError):
        OCRResult(document_id="doc_1", full_text="Text", tokens="not-a-list")


def test_ocr_result_invalid_token_element():
    with pytest.raises(TypeError):
        OCRResult(document_id="doc_1", full_text="Text", tokens=["not-a-token"])


def test_ocr_result_invalid_metadata():
    with pytest.raises(TypeError):
        OCRResult(document_id="doc_1", full_text="Text", tokens=[], metadata="not-a-dict")


@pytest.mark.parametrize("ms", [-1.0, "100", float("nan")])
def test_ocr_result_invalid_processing_time(ms):
    with pytest.raises((ValueError, TypeError)):
        OCRResult(document_id="doc_1", full_text="Text", tokens=[], processing_time_ms=ms)


def test_ocr_result_serialization():
    bbox = BoundingBox(0, 0, 10, 10)
    t = OCRToken("Hello", bbox, 0.9)
    res = OCRResult(
        document_id="doc_1",
        full_text="Hello",
        tokens=[t],
        metadata={"source": "test"},
        processing_time_ms=50.0,
        model_name="Paddle",
    )
    data = res.to_dict()
    assert data["document_id"] == "doc_1"
    assert data["tokens"][0]["text"] == "Hello"

    restored = OCRResult.from_dict(data)
    assert restored.document_id == res.document_id
    assert restored.full_text == res.full_text
    assert len(restored.tokens) == 1
    assert restored.tokens[0].text == "Hello"
    assert restored.processing_time_ms == 50.0


def test_ocr_result_from_dict_missing_keys():
    with pytest.raises(KeyError):
        OCRResult.from_dict({"document_id": "doc_1"})


# ============================================================================
# KIEResult Tests
# ============================================================================

def test_kie_result_valid_arbitrary_fields():
    kie = KIEResult(
        document_id="doc_10",
        fields={"invoice_no": "INV-001", "total": "125.00", "custom_field": [1, 2]},
        confidences={"invoice_no": 0.99, "total": 0.85},
        metadata={"method": "regex"},
        processing_time_ms=12.0,
        model_name="RegexKIE",
    )
    assert kie.fields["invoice_no"] == "INV-001"
    assert kie.confidences["total"] == 0.85
    assert kie.processing_time_ms == 12.0


def test_kie_result_empty_fields():
    kie = KIEResult(document_id="doc_11", fields={})
    assert kie.fields == {}
    assert kie.confidences is None


@pytest.mark.parametrize("doc_id", ["", "  ", None, 123])
def test_kie_result_invalid_document_id(doc_id):
    with pytest.raises((ValueError, TypeError)):
        KIEResult(document_id=doc_id, fields={})


def test_kie_result_invalid_fields_type():
    with pytest.raises(TypeError):
        KIEResult(document_id="doc_1", fields="not-a-dict")


def test_kie_result_invalid_metadata():
    with pytest.raises(TypeError):
        KIEResult(document_id="doc_1", fields={}, metadata="not-a-dict")


def test_kie_result_invalid_confidences_type():
    with pytest.raises(TypeError):
        KIEResult(document_id="doc_1", fields={}, confidences="not-a-dict")


@pytest.mark.parametrize("c_val", [-0.1, 1.5, "high", True])
def test_kie_result_invalid_confidence_values(c_val):
    with pytest.raises((ValueError, TypeError)):
        KIEResult(document_id="doc_1", fields={"total": "10"}, confidences={"total": c_val})


def test_kie_result_invalid_processing_time():
    with pytest.raises(ValueError):
        KIEResult(document_id="doc_1", fields={}, processing_time_ms=-5.0)


def test_kie_result_serialization():
    kie = KIEResult(
        document_id="doc_1",
        fields={"total": "100.0"},
        confidences={"total": 0.95},
        metadata={"k": "v"},
        processing_time_ms=10.0,
        model_name="model1",
    )
    data = kie.to_dict()
    assert data["fields"]["total"] == "100.0"
    restored = KIEResult.from_dict(data)
    assert restored.document_id == kie.document_id
    assert restored.fields == kie.fields
    assert restored.confidences == kie.confidences


def test_kie_result_from_dict_missing_keys():
    with pytest.raises(KeyError):
        KIEResult.from_dict({"document_id": "doc_1"})


# ============================================================================
# DocumentMetadata Tests
# ============================================================================

@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_document_metadata_valid_splits(split):
    doc = DocumentMetadata(
        document_id="d1",
        image_path="/path/to/img.png",
        width=800,
        height=600,
        split=split,
        metadata={"source": "sroie"},
    )
    assert doc.split == split
    assert doc.width == 800
    assert doc.height == 600


@pytest.mark.parametrize("split", ["val", "dev", "train_val", "TEST"])
def test_document_metadata_invalid_split(split):
    with pytest.raises(ValueError):
        DocumentMetadata(
            document_id="d1",
            image_path="/path/to/img.png",
            width=800,
            height=600,
            split=split,
        )


@pytest.mark.parametrize("w,h", [(0, 100), (100, 0), (-10, 100), (100, -10), (True, 100), (100, "200")])
def test_document_metadata_invalid_dims(w, h):
    with pytest.raises((ValueError, TypeError)):
        DocumentMetadata(
            document_id="d1",
            image_path="/path/to/img.png",
            width=w,
            height=h,
            split="train",
        )


@pytest.mark.parametrize("doc_id,img_path", [("", "/path"), ("d1", ""), ("   ", "/path")])
def test_document_metadata_invalid_strings(doc_id, img_path):
    with pytest.raises(ValueError):
        DocumentMetadata(
            document_id=doc_id,
            image_path=img_path,
            width=100,
            height=100,
            split="train",
        )


def test_document_metadata_invalid_metadata():
    with pytest.raises(TypeError):
        DocumentMetadata(
            document_id="d1",
            image_path="/path",
            width=100,
            height=100,
            split="train",
            metadata="not-dict",
        )


def test_document_metadata_serialization():
    doc = DocumentMetadata(
        document_id="d1",
        image_path="/img.png",
        width=1024,
        height=768,
        split="test",
        metadata={"dpi": 300},
    )
    data = doc.to_dict()
    assert data["width"] == 1024
    restored = DocumentMetadata.from_dict(data)
    assert restored.document_id == doc.document_id
    assert restored.split == doc.split
    assert restored.metadata == doc.metadata


def test_document_metadata_from_dict_missing_keys():
    with pytest.raises(KeyError):
        DocumentMetadata.from_dict({"document_id": "d1", "image_path": "/img.png"})


# ============================================================================
# DegradationSpec Tests
# ============================================================================

@pytest.mark.parametrize("sev", [0, 1, 2, 3, 4])
def test_degradation_spec_valid_severities(sev):
    spec = DegradationSpec(type="gaussian_blur", severity=sev, parameters={"sigma": 1.5}, seed=100)
    assert spec.severity == sev
    assert spec.type == "gaussian_blur"
    assert spec.parameters == {"sigma": 1.5}
    assert spec.seed == 100


@pytest.mark.parametrize("sev", [-1, 5, 10, "2", True, 2.5])
def test_degradation_spec_invalid_severity(sev):
    with pytest.raises((ValueError, TypeError)):
        DegradationSpec(type="blur", severity=sev)


@pytest.mark.parametrize("deg_type", ["", "   ", None, 123])
def test_degradation_spec_invalid_type(deg_type):
    with pytest.raises((ValueError, TypeError)):
        DegradationSpec(type=deg_type, severity=1)


@pytest.mark.parametrize("seed", ["42", True, 3.14, None])
def test_degradation_spec_invalid_seed(seed):
    with pytest.raises(TypeError):
        DegradationSpec(type="blur", severity=1, seed=seed)


def test_degradation_spec_invalid_params():
    with pytest.raises(TypeError):
        DegradationSpec(type="blur", severity=1, parameters="not-dict")


def test_degradation_spec_serialization():
    spec = DegradationSpec(type="motion_blur", severity=3, parameters={"kernel_size": 9}, seed=7)
    data = spec.to_dict()
    assert data["type"] == "motion_blur"
    assert data["severity"] == 3
    restored = DegradationSpec.from_dict(data)
    assert restored.type == spec.type
    assert restored.severity == spec.severity
    assert restored.parameters == spec.parameters
    assert restored.seed == spec.seed


def test_degradation_spec_from_dict_missing_keys():
    with pytest.raises(KeyError):
        DegradationSpec.from_dict({"type": "blur"})


# ============================================================================
# PreprocessingSpec Tests
# ============================================================================

def test_preprocessing_spec_default():
    spec = PreprocessingSpec()
    assert spec.enabled is True
    assert spec.methods == []
    assert spec.parameters == {}


def test_preprocessing_spec_custom():
    spec = PreprocessingSpec(
        enabled=True,
        methods=["deskew", "denoise"],
        parameters={"deskew_angle_max": 45},
    )
    assert spec.enabled is True
    assert spec.methods == ["deskew", "denoise"]
    assert spec.parameters["deskew_angle_max"] == 45


@pytest.mark.parametrize("enabled", ["True", 1, None])
def test_preprocessing_spec_invalid_enabled(enabled):
    with pytest.raises(TypeError):
        PreprocessingSpec(enabled=enabled)


def test_preprocessing_spec_invalid_methods_type():
    with pytest.raises(TypeError):
        PreprocessingSpec(methods="deskew")


@pytest.mark.parametrize("m_val", ["", "   ", 123, None])
def test_preprocessing_spec_invalid_method_element(m_val):
    with pytest.raises((ValueError, TypeError)):
        PreprocessingSpec(methods=["deskew", m_val])


def test_preprocessing_spec_invalid_parameters():
    with pytest.raises(TypeError):
        PreprocessingSpec(parameters=["not-dict"])


def test_preprocessing_spec_serialization():
    spec = PreprocessingSpec(enabled=False, methods=["binarize"], parameters={"thresh": 128})
    data = spec.to_dict()
    assert data["enabled"] is False
    assert data["methods"] == ["binarize"]
    restored = PreprocessingSpec.from_dict(data)
    assert restored.enabled == spec.enabled
    assert restored.methods == spec.methods
    assert restored.parameters == spec.parameters


# ============================================================================
# ExperimentResult Tests
# ============================================================================

def test_experiment_result_valid():
    deg = DegradationSpec(type="blur", severity=2)
    prep = PreprocessingSpec(enabled=True, methods=["sharpen"])
    exp = ExperimentResult(
        experiment_id="exp_001",
        document_id="doc_1",
        degradation=deg,
        preprocessing=prep,
        ocr_model="PaddleOCR",
        kie_model="RegexKIE",
        metrics={"cer": 0.05, "wer": 0.12, "f1": 0.88},
        metadata={"git_hash": "a4cde0e"},
    )
    assert exp.experiment_id == "exp_001"
    assert exp.metrics["cer"] == 0.05
    assert exp.metadata["git_hash"] == "a4cde0e"


def test_experiment_result_dict_nested_auto_conversion():
    exp = ExperimentResult(
        experiment_id="exp_002",
        document_id="doc_2",
        degradation={"type": "noise", "severity": 1},
        preprocessing={"enabled": False},
        ocr_model="Tesseract",
        kie_model="LayoutLM",
    )
    assert isinstance(exp.degradation, DegradationSpec)
    assert isinstance(exp.preprocessing, PreprocessingSpec)


@pytest.mark.parametrize("exp_id,doc_id,ocr,kie", [
    ("", "d1", "ocr", "kie"),
    ("e1", "", "ocr", "kie"),
    ("e1", "d1", "", "kie"),
    ("e1", "d1", "ocr", ""),
])
def test_experiment_result_invalid_ids(exp_id, doc_id, ocr, kie):
    with pytest.raises(ValueError):
        ExperimentResult(
            experiment_id=exp_id,
            document_id=doc_id,
            degradation=DegradationSpec(type="blur", severity=0),
            preprocessing=PreprocessingSpec(),
            ocr_model=ocr,
            kie_model=kie,
        )


def test_experiment_result_invalid_types():
    deg = DegradationSpec(type="blur", severity=0)
    prep = PreprocessingSpec()
    with pytest.raises(TypeError):
        ExperimentResult("e1", "d1", degradation="not-deg", preprocessing=prep, ocr_model="o", kie_model="k")
    with pytest.raises(TypeError):
        ExperimentResult("e1", "d1", degradation=deg, preprocessing="not-prep", ocr_model="o", kie_model="k")
    with pytest.raises(TypeError):
        ExperimentResult("e1", "d1", degradation=deg, preprocessing=prep, ocr_model="o", kie_model="k", metrics="str")
    with pytest.raises(TypeError):
        ExperimentResult("e1", "d1", degradation=deg, preprocessing=prep, ocr_model="o", kie_model="k", metadata="str")


def test_experiment_result_serialization():
    deg = DegradationSpec(type="blur", severity=1)
    prep = PreprocessingSpec(enabled=True, methods=["denoise"])
    exp = ExperimentResult(
        experiment_id="exp_001",
        document_id="doc_1",
        degradation=deg,
        preprocessing=prep,
        ocr_model="Paddle",
        kie_model="Rules",
        metrics={"cer": 0.02},
    )
    data = exp.to_dict()
    assert data["degradation"]["type"] == "blur"
    assert data["preprocessing"]["methods"] == ["denoise"]

    restored = ExperimentResult.from_dict(data)
    assert restored.experiment_id == exp.experiment_id
    assert restored.degradation == deg
    assert restored.preprocessing == prep
    assert restored.metrics == exp.metrics


def test_experiment_result_from_dict_missing_keys():
    with pytest.raises(KeyError):
        ExperimentResult.from_dict({"experiment_id": "exp_1"})


# ============================================================================
# OCRGroundTruth Tests
# ============================================================================

def test_ocr_ground_truth_valid():
    gt = OCRGroundTruth(document_id="d1", text="Clean Ground Truth Text")
    assert gt.document_id == "d1"
    assert gt.text == "Clean Ground Truth Text"
    assert gt.tokens is None


def test_ocr_ground_truth_with_tokens():
    bbox = BoundingBox(0, 0, 5, 5)
    t = OCRToken("Clean", bbox)
    gt = OCRGroundTruth(document_id="d1", text="Clean", tokens=[t, {"text": "Text", "bbox": bbox.to_dict()}])
    assert len(gt.tokens) == 2
    assert isinstance(gt.tokens[1], OCRToken)


@pytest.mark.parametrize("doc_id", ["", "  ", None, 123])
def test_ocr_ground_truth_invalid_doc_id(doc_id):
    with pytest.raises((ValueError, TypeError)):
        OCRGroundTruth(document_id=doc_id, text="Some text")


def test_ocr_ground_truth_invalid_text():
    with pytest.raises(TypeError):
        OCRGroundTruth(document_id="d1", text=123)


def test_ocr_ground_truth_invalid_tokens():
    with pytest.raises(TypeError):
        OCRGroundTruth(document_id="d1", text="abc", tokens="not-list")
    with pytest.raises(TypeError):
        OCRGroundTruth(document_id="d1", text="abc", tokens=["not-token"])


def test_ocr_ground_truth_invalid_metadata():
    with pytest.raises(TypeError):
        OCRGroundTruth(document_id="d1", text="abc", metadata="not-dict")


def test_ocr_ground_truth_serialization():
    bbox = BoundingBox(0, 0, 5, 5)
    t = OCRToken("Clean", bbox)
    gt = OCRGroundTruth(document_id="d1", text="Clean", tokens=[t], metadata={"verified": True})
    data = gt.to_dict()
    assert data["tokens"][0]["text"] == "Clean"
    assert data["metadata"]["verified"] is True

    restored = OCRGroundTruth.from_dict(data)
    assert restored.document_id == gt.document_id
    assert restored.text == gt.text
    assert len(restored.tokens) == 1
    assert restored.tokens[0].bbox == bbox


def test_ocr_ground_truth_from_dict_missing_keys():
    with pytest.raises(KeyError):
        OCRGroundTruth.from_dict({"document_id": "d1"})


# ============================================================================
# KIEGroundTruth Tests
# ============================================================================

def test_kie_ground_truth_valid():
    gt = KIEGroundTruth(
        document_id="d1",
        fields={"company": "Acme Corp", "total": "500.00"},
        metadata={"annotator": "A1"},
    )
    assert gt.document_id == "d1"
    assert gt.fields["company"] == "Acme Corp"
    assert gt.metadata["annotator"] == "A1"


@pytest.mark.parametrize("doc_id", ["", "  ", None, 123])
def test_kie_ground_truth_invalid_doc_id(doc_id):
    with pytest.raises((ValueError, TypeError)):
        KIEGroundTruth(document_id=doc_id, fields={})


def test_kie_ground_truth_invalid_fields():
    with pytest.raises(TypeError):
        KIEGroundTruth(document_id="d1", fields="not-dict")


def test_kie_ground_truth_invalid_metadata():
    with pytest.raises(TypeError):
        KIEGroundTruth(document_id="d1", fields={}, metadata="not-dict")


def test_kie_ground_truth_serialization():
    gt = KIEGroundTruth(
        document_id="d1",
        fields={"total": "125.0"},
        metadata={"dataset": "sroie"},
    )
    data = gt.to_dict()
    assert data["fields"]["total"] == "125.0"
    restored = KIEGroundTruth.from_dict(data)
    assert restored.document_id == gt.document_id
    assert restored.fields == gt.fields
    assert restored.metadata == gt.metadata


def test_kie_ground_truth_from_dict_missing_keys():
    with pytest.raises(KeyError):
        KIEGroundTruth.from_dict({"document_id": "d1"})
