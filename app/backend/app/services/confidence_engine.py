"""Confidence Engine for multi-level scoring."""

from typing import Dict, List, Optional
from app.schemas.kie import NormalizedFieldDTO
from app.schemas.ocr import NormalizedTokenDTO


class ConfidenceEngine:
    """Calculates granular and aggregate confidence scores across tokens, fields, and documents."""

    @staticmethod
    def calculate_ocr_confidence(tokens: List[NormalizedTokenDTO]) -> float:
        """Aggregate average confidence across recognized OCR tokens."""
        if not tokens:
            return 0.0
        confs = [t.confidence for t in tokens if t.confidence is not None]
        return round(float(sum(confs) / max(len(confs), 1)), 4)

    @staticmethod
    def calculate_field_confidence(field: NormalizedFieldDTO) -> float:
        """Extract and validate normalized confidence for a field."""
        return round(max(0.0, min(1.0, float(field.confidence))), 4)

    @classmethod
    def calculate_document_confidence(
        cls,
        fields: Dict[str, NormalizedFieldDTO],
        ocr_tokens: Optional[List[NormalizedTokenDTO]] = None,
        image_quality_score: Optional[float] = None,
        required_fields: Optional[List[str]] = None,
    ) -> float:
        """Compute holistic document-level confidence score.

        Formula:
        Weighted combination:
        - 60% average confidence of required extracted fields
        - 25% average OCR token confidence
        - 15% image quality score (if available, otherwise redistributed to fields/tokens)
        """
        if not fields:
            return 0.0

        # Calculate field component
        if required_fields:
            field_confs = [
                cls.calculate_field_confidence(fields[f])
                for f in required_fields
                if f in fields
            ]
        else:
            field_confs = [cls.calculate_field_confidence(f) for f in fields.values()]

        field_avg = float(sum(field_confs) / max(len(field_confs), 1)) if field_confs else 0.0

        # Token component
        token_avg = cls.calculate_ocr_confidence(ocr_tokens) if ocr_tokens else field_avg

        # Quality component
        if image_quality_score is not None:
            doc_conf = (0.60 * field_avg) + (0.25 * token_avg) + (0.15 * image_quality_score)
        else:
            doc_conf = (0.70 * field_avg) + (0.30 * token_avg)

        return round(max(0.0, min(1.0, float(doc_conf))), 4)
