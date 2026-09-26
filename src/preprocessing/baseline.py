"""B0/B1/B2 Analytical Baseline Runner Abstraction.

Implements the comparative evaluation of:
- B0: Original image -> OCR -> Evaluation
- B1: Degraded image -> OCR -> Evaluation
- B2: Degraded image -> Preprocessed image -> OCR -> Evaluation
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional
import numpy as np

from src.core.contracts import BaseOCREngine
from src.core.schemas import DegradationSpec, PreprocessingSpec
from src.degradation.pipeline import get_degradation
from src.evaluation.ocr_metrics import OCREvaluationResult, evaluate_ocr
from src.preprocessing.pipeline import PreprocessingPipeline


@dataclass
class BaselineRunOutput:
    """Evaluation output for a single baseline branch (B0, B1, or B2)."""

    branch: str
    full_text: str
    tokens_count: int
    processing_time_ms: Optional[float]
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BaselineComparisonResult:
    """Comparative outcome for B0, B1, and B2 baselines on a single document."""

    document_id: str
    b0: BaselineRunOutput
    b1: BaselineRunOutput
    b2: BaselineRunOutput
    degradation_spec: Dict[str, Any]
    preprocessing_spec: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "b0": self.b0.to_dict(),
            "b1": self.b1.to_dict(),
            "b2": self.b2.to_dict(),
            "degradation_spec": self.degradation_spec,
            "preprocessing_spec": self.preprocessing_spec,
        }


def run_b0_b1_b2_comparison(
    image: np.ndarray,
    document_id: str,
    gt_text: str,
    degradation_spec: DegradationSpec,
    preprocessing_spec: PreprocessingSpec,
    ocr_engine: BaseOCREngine,
) -> BaselineComparisonResult:
    """Execute B0, B1, and B2 branches on a single document image.

    Args:
        image: Original RGB uint8 document image.
        document_id: Identifier of the document.
        gt_text: Raw Ground Truth text for evaluation.
        degradation_spec: Synthetic degradation specification.
        preprocessing_spec: Preprocessing pipeline specification.
        ocr_engine: Initialized OCR engine.

    Returns:
        BaselineComparisonResult containing detailed metrics for B0, B1, and B2.
    """
    # 1. B0: Original -> OCR
    ocr_b0 = ocr_engine.recognize(image, document_id)
    eval_b0 = evaluate_ocr(pred=ocr_b0.full_text, gt=gt_text)
    out_b0 = BaselineRunOutput(
        branch="B0_original",
        full_text=ocr_b0.full_text,
        tokens_count=len(ocr_b0.tokens),
        processing_time_ms=ocr_b0.processing_time_ms,
        metrics=eval_b0.to_dict(),
    )

    # 2. B1: Degraded -> OCR
    deg_primitive = get_degradation(degradation_spec.type)
    deg_image, applied_deg = deg_primitive.apply(image, degradation_spec)
    ocr_b1 = ocr_engine.recognize(deg_image, document_id)
    eval_b1 = evaluate_ocr(pred=ocr_b1.full_text, gt=gt_text)
    out_b1 = BaselineRunOutput(
        branch="B1_degraded",
        full_text=ocr_b1.full_text,
        tokens_count=len(ocr_b1.tokens),
        processing_time_ms=ocr_b1.processing_time_ms,
        metrics=eval_b1.to_dict(),
    )

    # 3. B2: Degraded -> Preprocessed -> OCR
    prep_pipeline = PreprocessingPipeline()
    prep_image, applied_prep = prep_pipeline.process(deg_image, preprocessing_spec)
    ocr_b2 = ocr_engine.recognize(prep_image, document_id)
    eval_b2 = evaluate_ocr(pred=ocr_b2.full_text, gt=gt_text)
    out_b2 = BaselineRunOutput(
        branch="B2_degraded_preprocessed",
        full_text=ocr_b2.full_text,
        tokens_count=len(ocr_b2.tokens),
        processing_time_ms=ocr_b2.processing_time_ms,
        metrics=eval_b2.to_dict(),
    )

    return BaselineComparisonResult(
        document_id=document_id,
        b0=out_b0,
        b1=out_b1,
        b2=out_b2,
        degradation_spec=degradation_spec.to_dict(),
        preprocessing_spec=preprocessing_spec.to_dict(),
    )
