"""Unit tests for abstract module contracts and pipeline interfaces."""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import pytest

from src.core.contracts import (
    BaseDegradation,
    BaseEvaluator,
    BaseKIEEngine,
    BaseOCREngine,
    BasePreprocessor,
)
from src.core.schemas import (
    DegradationSpec,
    KIEGroundTruth,
    KIEResult,
    OCRGroundTruth,
    OCRResult,
    PreprocessingSpec,
)


def test_cannot_instantiate_abstract_contracts():
    with pytest.raises(TypeError):
        BaseDegradation()  # type: ignore[abstract]
    with pytest.raises(TypeError):
        BasePreprocessor()  # type: ignore[abstract]
    with pytest.raises(TypeError):
        BaseOCREngine()  # type: ignore[abstract]
    with pytest.raises(TypeError):
        BaseKIEEngine()  # type: ignore[abstract]
    with pytest.raises(TypeError):
        BaseEvaluator()  # type: ignore[abstract]


def test_concrete_degradation_implementation():
    class DummyDegradation(BaseDegradation):
        def apply(
            self, image: np.ndarray, spec: DegradationSpec
        ) -> Tuple[np.ndarray, DegradationSpec]:
            super().apply(image, spec)
            return image, spec

    degrader = DummyDegradation()
    img = np.zeros((10, 10), dtype=np.uint8)
    spec = DegradationSpec(type="none", severity=0)
    out_img, out_spec = degrader.apply(img, spec)
    assert np.array_equal(out_img, img)
    assert out_spec == spec


def test_concrete_preprocessor_implementation():
    class DummyPreprocessor(BasePreprocessor):
        def process(
            self, image: np.ndarray, spec: PreprocessingSpec
        ) -> Tuple[np.ndarray, PreprocessingSpec]:
            super().process(image, spec)
            return image, spec

    preprocessor = DummyPreprocessor()
    img = np.zeros((10, 10), dtype=np.uint8)
    spec = PreprocessingSpec(enabled=True)
    out_img, out_spec = preprocessor.process(img, spec)
    assert np.array_equal(out_img, img)
    assert out_spec == spec


def test_concrete_ocr_engine_implementation():
    class DummyOCR(BaseOCREngine):
        def recognize(self, image: np.ndarray, document_id: str) -> OCRResult:
            super().recognize(image, document_id)
            return OCRResult(document_id=document_id, full_text="", tokens=[])

    ocr = DummyOCR()
    img = np.zeros((10, 10), dtype=np.uint8)
    res = ocr.recognize(img, "doc_test")
    assert res.document_id == "doc_test"
    assert res.tokens == []


def test_concrete_kie_engine_implementation():
    class DummyKIE(BaseKIEEngine):
        def extract(
            self, ocr_result: OCRResult, image: Optional[np.ndarray] = None
        ) -> KIEResult:
            super().extract(ocr_result, image)
            return KIEResult(document_id=ocr_result.document_id, fields={"total": "100"})

    kie = DummyKIE()
    ocr_res = OCRResult(document_id="doc_test", full_text="TOTAL 100", tokens=[])
    res = kie.extract(ocr_res)
    assert res.document_id == "doc_test"
    assert res.fields["total"] == "100"


def test_concrete_evaluator_implementation():
    class DummyEvaluator(BaseEvaluator):
        def evaluate_ocr(
            self, prediction: OCRResult, ground_truth: OCRGroundTruth
        ) -> Dict[str, Any]:
            super().evaluate_ocr(prediction, ground_truth)
            return {"cer": 0.0}

        def evaluate_kie(
            self, prediction: KIEResult, ground_truth: KIEGroundTruth
        ) -> Dict[str, Any]:
            super().evaluate_kie(prediction, ground_truth)
            return {"f1": 1.0}

    evaluator = DummyEvaluator()
    ocr_pred = OCRResult("d1", "text", [])
    ocr_gt = OCRGroundTruth("d1", "text")
    kie_pred = KIEResult("d1", {"f": "1"})
    kie_gt = KIEGroundTruth("d1", {"f": "1"})

    assert evaluator.evaluate_ocr(ocr_pred, ocr_gt) == {"cer": 0.0}
    assert evaluator.evaluate_kie(kie_pred, kie_gt) == {"f1": 1.0}
