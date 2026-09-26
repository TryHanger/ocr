"""Rule-based Key Information Extraction (KIE) engine for SROIE receipts."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Union
import yaml

from src.core.contracts import BaseKIEEngine
from src.core.schemas import BoundingBox, KIEResult, OCRResult, OCRToken
from src.kie.rules.address import extract_address_candidate
from src.kie.rules.candidate import FieldCandidate
from src.kie.rules.company import extract_company_candidate
from src.kie.rules.date import extract_date_candidate
from src.kie.rules.total import extract_total_candidate


class RuleBasedKIEEngine(BaseKIEEngine):
    """Deterministic, rule-based KIE extractor for receipt key information.

    Extracts: company, date, address, total from OCRResult tokens and text.
    Zero GT leakage: receives strictly OCRResult and document_id.
    """

    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], str]] = None,
    ) -> None:
        """Initialize RuleBasedKIEEngine with configuration.

        Args:
            config: Path to YAML config file or dictionary of configuration parameters.
        """
        if isinstance(config, str):
            if os.path.isfile(config):
                with open(config, "r", encoding="utf-8") as f:
                    self.config: Dict[str, Any] = yaml.safe_load(f) or {}
            else:
                self.config = {}
        elif isinstance(config, dict):
            self.config = config
        else:
            # Default fallback to configs/kie.yaml if present
            default_cfg_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "kie.yaml")
            default_cfg_path = os.path.abspath(default_cfg_path)
            if os.path.isfile(default_cfg_path):
                with open(default_cfg_path, "r", encoding="utf-8") as f:
                    self.config = yaml.safe_load(f) or {}
            else:
                self.config = {}

        engine_cfg = self.config.get("engine", {})
        self.model_name = engine_cfg.get("model_name", "sroie_rule_based_kie")
        self.model_version = engine_cfg.get("model_version", "1.0.0")
        self.min_confidence = float(engine_cfg.get("min_confidence_threshold", 0.2))

    def extract(self, ocr_result: OCRResult, document_id: str) -> KIEResult:
        """Extract structured fields from OCR tokens.

        Args:
            ocr_result: Standardized OCR tokens and full text.
            document_id: Identifier of the document being processed.

        Returns:
            Standardized KIEResult containing extracted field mappings and confidences.
        """
        start_time = time.perf_counter()

        if not isinstance(ocr_result, OCRResult):
            raise TypeError(f"ocr_result must be OCRResult, got {type(ocr_result).__name__}")
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("document_id must be a non-empty string")
        if ocr_result.document_id != document_id:
            raise ValueError(
                f"document_id mismatch: OCRResult has '{ocr_result.document_id}', argument has '{document_id}'"
            )

        tokens = ocr_result.tokens
        # If tokens are empty but full_text has content, split into line tokens
        if not tokens and ocr_result.full_text.strip():
            tokens = [
                OCRToken(text=line.strip(), bbox=BoundingBox(0, 0, 10, 10), confidence=1.0)
                for line in ocr_result.full_text.splitlines()
                if line.strip()
            ]

        # Handle empty OCR output
        if not tokens:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return KIEResult(
                document_id=document_id,
                fields={},
                confidences={},
                metadata={"status": "EMPTY_OCR", "field_provenance": {}, "rules_applied": {}},
                processing_time_ms=round(elapsed_ms, 2),
                model_name=self.model_name,
            )

        # Extract field candidates
        company_cand = extract_company_candidate(tokens, self.config)
        date_cand = extract_date_candidate(tokens, self.config)
        address_cand = extract_address_candidate(tokens, company_cand, self.config)
        total_cand = extract_total_candidate(tokens, self.config)

        candidates: Dict[str, Optional[FieldCandidate]] = {
            "company": company_cand,
            "date": date_cand,
            "address": address_cand,
            "total": total_cand,
        }

        extracted_fields: Dict[str, str] = {}
        extracted_confidences: Dict[str, float] = {}
        field_provenance: Dict[str, List[int]] = {}
        rules_applied: Dict[str, str] = {}

        for field_name, cand in candidates.items():
            if cand is not None and cand.confidence >= self.min_confidence:
                extracted_fields[field_name] = cand.text
                extracted_confidences[field_name] = cand.confidence
                field_provenance[field_name] = cand.token_indices
                rules_applied[field_name] = cand.rule_name

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return KIEResult(
            document_id=document_id,
            fields=extracted_fields,
            confidences=extracted_confidences,
            metadata={
                "status": "SUCCESS",
                "field_provenance": field_provenance,
                "rules_applied": rules_applied,
            },
            processing_time_ms=round(elapsed_ms, 2),
            model_name=self.model_name,
        )
