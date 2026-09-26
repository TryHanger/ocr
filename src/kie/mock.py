"""Mock KIE engine for testing and isolated verification."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from src.core.contracts import BaseKIEEngine
from src.core.schemas import KIEResult, OCRResult


class MockKIEEngine(BaseKIEEngine):
    """Deterministic mock KIE engine for unit and pipeline tests."""

    def __init__(
        self,
        canned_fields: Optional[Dict[str, Dict[str, Any]]] = None,
        default_fields: Optional[Dict[str, Any]] = None,
        model_name: str = "mock_kie_engine",
    ) -> None:
        """Initialize MockKIEEngine.

        Args:
            canned_fields: Mapping of document_id to extracted field dictionary.
            default_fields: Fallback fields to return if document_id not in canned_fields.
            model_name: Engine identifier string.
        """
        self.canned_fields = canned_fields or {}
        self.default_fields = default_fields or {
            "company": "MOCK STORE SDN BHD",
            "date": "01/01/2020",
            "address": "MOCK ADDRESS, MALAYSIA",
            "total": "10.00",
        }
        self.model_name = model_name

    def extract(self, ocr_result: OCRResult, document_id: str) -> KIEResult:
        """Return canned or default fields for the given document_id."""
        start_time = time.perf_counter()

        if not isinstance(ocr_result, OCRResult):
            raise TypeError(f"ocr_result must be OCRResult, got {type(ocr_result).__name__}")
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("document_id must be a non-empty string")
        if ocr_result.document_id != document_id:
            raise ValueError(
                f"document_id mismatch: OCRResult has '{ocr_result.document_id}', argument has '{document_id}'"
            )

        fields = self.canned_fields.get(document_id, self.default_fields)
        confidences = {k: 1.0 for k in fields}
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return KIEResult(
            document_id=document_id,
            fields=dict(fields),
            confidences=confidences,
            metadata={"status": "MOCK_SUCCESS", "field_provenance": {k: [0] for k in fields}},
            processing_time_ms=round(elapsed_ms, 2),
            model_name=self.model_name,
        )
