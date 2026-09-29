"""KIE Adapter connecting to RuleBasedKIEEngine with provenance resolution."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from typing import Any, Dict, List, Optional

from app.adapters.base import KIEProvider
from app.core.logging import logger
from app.schemas.kie import NormalizedFieldDTO, NormalizedKIEResult
from app.schemas.ocr import NormalizedOCRResult

_kie_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="kie_inference")


class ResearchRuleBasedKIEAdapter(KIEProvider):
    """Bridges production API to src.kie.rule_based.RuleBasedKIEEngine."""

    def __init__(self) -> None:
        self._engine: Optional[Any] = None
        self._initialized = False

    def _get_engine(self) -> Any:
        if not self._initialized:
            try:
                from src.kie.rule_based import RuleBasedKIEEngine

                self._engine = RuleBasedKIEEngine()
                self._initialized = True
                logger.info("Initialized Research RuleBasedKIEEngine successfully")
            except Exception as e:
                logger.warning(
                    f"Could not initialize research RuleBasedKIEEngine ({e}). Operating in fallback/mock mode."
                )
                self._engine = None
                self._initialized = True
        return self._engine

    def _sync_extract(self, ocr_result: NormalizedOCRResult, document_id: str) -> NormalizedKIEResult:
        t0 = time.perf_counter()
        engine = self._get_engine()

        if engine is not None:
            # Build research schema OCRResult to satisfy research contract without GT leakage
            from src.core.schemas import BoundingBox, OCRResult as ResearchOCRResult, OCRToken as ResearchOCRToken

            research_tokens: List[ResearchOCRToken] = []
            for t in ocr_result.tokens:
                bbox_obj = BoundingBox(
                    x_min=t.bbox[0],
                    y_min=t.bbox[1],
                    x_max=t.bbox[2],
                    y_max=t.bbox[3],
                )
                research_tokens.append(
                    ResearchOCRToken(text=t.text, bbox=bbox_obj, confidence=t.confidence)
                )

            research_ocr = ResearchOCRResult(
                document_id=document_id,
                full_text=ocr_result.full_text,
                tokens=research_tokens,
            )

            kie_raw = engine.extract(research_ocr, document_id)

            provenance = kie_raw.metadata.get("field_provenance", {})
            rules_applied = kie_raw.metadata.get("rules_applied", {})

            normalized_fields: Dict[str, NormalizedFieldDTO] = {}
            for field_name, value in kie_raw.fields.items():
                conf = float(kie_raw.confidences.get(field_name, 0.0) if kie_raw.confidences else 0.0)
                token_indices = provenance.get(field_name, [])

                # Calculate union bounding box from tokens
                union_bbox: Optional[List[float]] = None
                if token_indices and ocr_result.tokens:
                    valid_boxes = [
                        ocr_result.tokens[idx].bbox
                        for idx in token_indices
                        if idx < len(ocr_result.tokens)
                    ]
                    if valid_boxes:
                        u_x1 = min(b[0] for b in valid_boxes)
                        u_y1 = min(b[1] for b in valid_boxes)
                        u_x2 = max(b[2] for b in valid_boxes)
                        u_y2 = max(b[3] for b in valid_boxes)
                        union_bbox = [round(u_x1, 5), round(u_y1, 5), round(u_x2, 5), round(u_y2, 5)]

                normalized_fields[field_name] = NormalizedFieldDTO(
                    field_name=field_name,
                    value=str(value),
                    confidence=conf,
                    bbox=union_bbox,
                    source_token_indices=token_indices,
                    validation_status="VALID" if str(value).strip() else "INVALID",
                    rule_applied=rules_applied.get(field_name),
                )

            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return NormalizedKIEResult(
                document_id=document_id,
                fields=normalized_fields,
                processing_time_ms=elapsed_ms,
                model_name="RuleBasedKIE SROIE",
                metadata={"status": kie_raw.metadata.get("status", "SUCCESS")},
            )
        else:
            # Deterministic mock fallback
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            mock_fields = {
                "company": NormalizedFieldDTO(
                    field_name="company",
                    value="SAMPLE COMPANY SDN BHD",
                    confidence=0.92,
                    bbox=[0.1, 0.05, 0.8, 0.15],
                    validation_status="VALID",
                ),
                "date": NormalizedFieldDTO(
                    field_name="date",
                    value="2026-09-28",
                    confidence=0.91,
                    bbox=[0.1, 0.2, 0.5, 0.25],
                    validation_status="VALID",
                ),
                "total": NormalizedFieldDTO(
                    field_name="total",
                    value="100.00",
                    confidence=0.96,
                    bbox=[0.1, 0.8, 0.9, 0.9],
                    validation_status="VALID",
                ),
            }
            return NormalizedKIEResult(
                document_id=document_id,
                fields=mock_fields,
                processing_time_ms=elapsed_ms,
                model_name="MockKIE",
                metadata={"fallback": True},
            )

    async def extract(
        self, ocr_result: NormalizedOCRResult, document_id: str
    ) -> NormalizedKIEResult:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _kie_executor, self._sync_extract, ocr_result, document_id
        )


default_kie_adapter = ResearchRuleBasedKIEAdapter()
