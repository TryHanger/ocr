"""Standalone domain validation service for document fields and business rules."""

from datetime import datetime
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.core.enums import DocumentType, ReviewReason


class ValidationError(BaseModel):
    """Specific field or document-level validation error."""

    field: str
    code: str
    message: str


class ValidationResult(BaseModel):
    """Complete validation outcome with machine-readable diagnostic details."""

    is_valid: bool
    errors: List[ValidationError] = Field(default_factory=list)
    field_statuses: Dict[str, str] = Field(default_factory=dict)
    failed_reason: Optional[str] = None
    review_details: Dict[str, Any] = Field(default_factory=dict)


class DocumentValidator:
    """Evaluates business rules, required fields, and format constraints without invoking ML models."""

    REQUIRED_FIELDS_BY_TYPE = {
        DocumentType.RECEIPT.value: ["company", "date", "total"],
        DocumentType.INVOICE.value: ["company", "date", "total"],
        DocumentType.ACT.value: ["company", "date"],
        DocumentType.CONTRACT.value: ["company", "date"],
    }

    # Date regex patterns covering ISO, slashes, dashes, dots, and alphanumeric month receipts
    DATE_PATTERNS = [
        re.compile(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}$"),          # YYYY-MM-DD
        re.compile(r"^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}$"),          # DD/MM/YYYY or DD-MM-YY
        re.compile(r"^\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}$"),        # 25 DEC 2018
        re.compile(r"^[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4}$"),      # DEC 25, 2018
    ]

    @classmethod
    def _clean_numeric_string(cls, val: str) -> Optional[float]:
        """Verify string represents a valid decimal number. Rejects strings with embedded letters like '12O.50'."""
        if not val:
            return None
        s = val.strip()
        # Strip allowed currency prefixes or suffixes ($ € £ RM USD EUR)
        s = re.sub(r"^([\$€£]|RM|USD|EUR)\s*", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"\s*([\$€£]|RM|USD|EUR)$", "", s, flags=re.IGNORECASE).strip()
        # Handle comma vs dot separators
        if "," in s and "." not in s:
            s = s.replace(",", ".")
        elif "," in s and "." in s:
            s = s.replace(",", "")  # thousand separator
        # Must strictly contain only digits and optional decimal point
        if not re.match(r"^\d+(\.\d+)?$", s):
            return None
        try:
            return float(s)
        except ValueError:
            return None

    @classmethod
    def _is_valid_date(cls, val: str) -> bool:
        """Check if date matches any standard document date pattern."""
        s = val.strip()
        if len(s) < 4:
            return False
        return any(p.match(s) for p in cls.DATE_PATTERNS)

    @classmethod
    def validate(
        cls,
        document_type: str,
        fields: Dict[str, Any],
    ) -> ValidationResult:
        """Execute all business validation rules against the current active field values.

        Args:
            document_type: Document category (receipt, invoice, etc.)
            fields: Mapping of field_name to string value or field entity.
        """
        # Normalize fields mapping to string values
        normalized_values: Dict[str, str] = {}
        for k, v in fields.items():
            if isinstance(v, str):
                normalized_values[k] = v.strip()
            elif hasattr(v, "corrected_value") and v.corrected_value is not None:
                normalized_values[k] = str(v.corrected_value).strip()
            elif hasattr(v, "value"):
                normalized_values[k] = str(v.value or "").strip()
            else:
                normalized_values[k] = str(v or "").strip()

        errors: List[ValidationError] = []
        field_statuses: Dict[str, str] = {name: "VALID" for name in normalized_values.keys()}
        required = cls.REQUIRED_FIELDS_BY_TYPE.get(
            document_type,
            cls.REQUIRED_FIELDS_BY_TYPE[DocumentType.RECEIPT.value],
        )

        # 1. Required Fields Check
        for req in required:
            val = normalized_values.get(req, "")
            if not val:
                errors.append(
                    ValidationError(
                        field=req,
                        code=ReviewReason.MISSING_REQUIRED_FIELD.value,
                        message=f"Mandatory field '{req}' is missing or empty.",
                    )
                )
                field_statuses[req] = "INVALID"

        # 2. Company Name Check
        if "company" in normalized_values and normalized_values["company"]:
            company_val = normalized_values["company"]
            if len(company_val) < 2:
                errors.append(
                    ValidationError(
                        field="company",
                        code=ReviewReason.EMPTY_VALUE.value,
                        message="Company name must contain at least 2 characters.",
                    )
                )
                field_statuses["company"] = "INVALID"

        # 3. Numeric Total Amount Check
        if "total" in normalized_values and normalized_values["total"]:
            total_raw = normalized_values["total"]
            parsed_total = cls._clean_numeric_string(total_raw)
            if parsed_total is None:
                errors.append(
                    ValidationError(
                        field="total",
                        code=ReviewReason.INVALID_NUMBER.value,
                        message=f"Total amount '{total_raw}' is not a valid numeric value.",
                    )
                )
                field_statuses["total"] = "INVALID"
            elif parsed_total <= 0.0:
                errors.append(
                    ValidationError(
                        field="total",
                        code=ReviewReason.DOCUMENT_INTEGRITY.value,
                        message=f"Total amount '{total_raw}' must be greater than zero.",
                    )
                )
                field_statuses["total"] = "INVALID"

        # 4. Date Format Check
        if "date" in normalized_values and normalized_values["date"]:
            date_raw = normalized_values["date"]
            if not cls._is_valid_date(date_raw):
                errors.append(
                    ValidationError(
                        field="date",
                        code=ReviewReason.INVALID_DATE.value,
                        message=f"Date '{date_raw}' does not match a recognizable date format.",
                    )
                )
                field_statuses["date"] = "INVALID"

        # Construct structured review_details
        is_valid = len(errors) == 0
        failed_reason: Optional[str] = errors[0].code if errors else None

        review_details = {
            "reasons": [
                {
                    "code": err.code,
                    "field": err.field,
                    "message": err.message,
                }
                for err in errors
            ],
            "validation_errors": [err.model_dump() for err in errors],
            "fields": {
                name: {
                    "status": field_statuses[name].lower(),
                }
                for name in field_statuses
            },
        }

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            field_statuses=field_statuses,
            failed_reason=failed_reason,
            review_details=review_details,
        )
