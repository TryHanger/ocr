from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.core.enums import AutomationDecision, DocumentType, ReviewReason
from app.schemas.kie import NormalizedFieldDTO
from app.schemas.quality import QualityResultDTO
from app.services.document_validator import DocumentValidator


class DecisionEngine:
    """Evaluates business rules, thresholds, and quality gates to make automation decisions."""

    DOCUMENT_POLICIES = {
        DocumentType.RECEIPT.value: {
            "required_fields": ["company", "date", "total"],
            "thresholds": {
                "company": settings.POLICY_RECEIPT_COMPANY_THRESHOLD,
                "date": settings.POLICY_RECEIPT_DATE_THRESHOLD,
                "address": settings.POLICY_RECEIPT_ADDRESS_THRESHOLD,
                "total": settings.POLICY_RECEIPT_TOTAL_THRESHOLD,
            },
            "min_quality_score": settings.POLICY_MIN_IMAGE_QUALITY_SCORE,
        }
    }

    @classmethod
    def get_policy(cls, document_type: str) -> Dict:
        return cls.DOCUMENT_POLICIES.get(
            document_type,
            cls.DOCUMENT_POLICIES[DocumentType.RECEIPT.value],
        )

    @classmethod
    def evaluate(
        cls,
        document_type: str,
        fields: Dict[str, NormalizedFieldDTO],
        quality: Optional[QualityResultDTO] = None,
        document_confidence: float = 0.0,
    ) -> Tuple[AutomationDecision, ReviewReason, str, Optional[Dict[str, Any]]]:
        """Evaluate whether document qualifies for AUTOMATIC or requires MANUAL_REVIEW.

        Returns:
            (decision, review_reason, explanation, review_details)
        """
        policy = cls.get_policy(document_type)
        thresholds: Dict[str, float] = policy.get("thresholds", {})
        min_quality: float = policy.get("min_quality_score", 0.50)

        # 1. Quality Gate Check
        if quality is not None and quality.quality_score < min_quality:
            expl = f"Image quality score ({quality.quality_score:.2f}) below threshold ({min_quality:.2f}). Profile: {quality.profile_summary}"
            details = {
                "reasons": [
                    {
                        "code": ReviewReason.LOW_IMAGE_QUALITY.value,
                        "field": None,
                        "confidence": quality.quality_score,
                        "threshold": min_quality,
                        "message": expl,
                    }
                ],
                "validation_errors": [],
                "fields": {},
            }
            return (
                AutomationDecision.MANUAL_REVIEW,
                ReviewReason.LOW_IMAGE_QUALITY,
                expl,
                details,
            )

        # 2. Domain Validation Check (Required fields, formats, numeric, date)
        val_result = DocumentValidator.validate(document_type, fields)
        if not val_result.is_valid:
            failed_reason = (
                ReviewReason(val_result.failed_reason)
                if val_result.failed_reason and val_result.failed_reason in ReviewReason._value2member_map_
                else ReviewReason.VALIDATION_FAILED
            )
            expl = val_result.errors[0].message if val_result.errors else "Document validation failed."
            return (
                AutomationDecision.MANUAL_REVIEW,
                failed_reason,
                expl,
                val_result.review_details,
            )

        # 3. Field Confidence Threshold Check
        low_conf_reasons = []
        low_conf_fields = {}
        for name, field in fields.items():
            threshold = thresholds.get(name, 0.85)
            if field.confidence < threshold:
                expl_item = f"Field '{name}' confidence ({field.confidence:.2f}) below threshold ({threshold:.2f})."
                low_conf_reasons.append(
                    {
                        "code": ReviewReason.LOW_CONFIDENCE.value,
                        "field": name,
                        "confidence": round(field.confidence, 4),
                        "threshold": threshold,
                        "message": expl_item,
                    }
                )
                low_conf_fields[name] = {
                    "status": "review",
                    "confidence": round(field.confidence, 4),
                    "threshold": threshold,
                }

        if low_conf_reasons:
            if len(low_conf_reasons) == 1:
                expl = low_conf_reasons[0]["message"]
            else:
                f_names = ", ".join(f"'{r['field']}'" for r in low_conf_reasons)
                expl = f"Fields {f_names} confidence below required threshold."

            details = {
                "reasons": low_conf_reasons,
                "validation_errors": [],
                "fields": low_conf_fields,
            }
            return (
                AutomationDecision.MANUAL_REVIEW,
                ReviewReason.LOW_CONFIDENCE,
                expl,
                details,
            )

        # All gates passed: AUTOMATIC approval
        return (
            AutomationDecision.AUTOMATIC,
            ReviewReason.NONE,
            "All required fields present, confidence thresholds satisfied, quality gate passed.",
            None,
        )
