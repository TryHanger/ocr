"""OCR Adapter connecting to RapidOCREngine with coordinate normalization."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
import time
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

from app.adapters.base import OCRProvider
from app.core.config import settings
from app.core.logging import logger
from app.schemas.ocr import NormalizedOCRResult, NormalizedTokenDTO

# Shared executor for heavy CPU/GPU inference tasks
_inference_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ocr_inference")


class ResearchRapidOCRAdapter(OCRProvider):
    """Bridges production API to src.ocr.rapid_ocr.RapidOCREngine."""

    def __init__(self, execution_provider: str = settings.OCR_EXECUTION_PROVIDER) -> None:
        self.execution_provider = execution_provider
        self._engine: Optional[Any] = None
        self._initialized = False

    def _get_engine(self) -> Any:
        if not self._initialized:
            try:
                from src.ocr.rapid_ocr import RapidOCREngine

                self._engine = RapidOCREngine(execution_provider=self.execution_provider)
                self._initialized = True
                logger.info(
                    "Initialized Research RapidOCREngine successfully",
                    extra={"execution_provider": self.execution_provider},
                )
            except Exception as e:
                logger.warning(
                    f"Could not initialize research RapidOCREngine ({e}). Operating in fallback/mock mode."
                )
                self._engine = None
                self._initialized = True
        return self._engine

    def _sync_process(
        self, image_bytes: bytes, document_id: str, execution_provider: str
    ) -> NormalizedOCRResult:
        t0 = time.perf_counter()

        # Decode image to RGB uint8
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Failed to decode image bytes for document {document_id}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        height, width = img_rgb.shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid image dimensions: {width}x{height}")

        engine = self._get_engine()

        if engine is not None:
            # Call research engine contract: recognize(image, document_id) -> OCRResult
            research_result = engine.recognize(img_rgb, document_id)

            normalized_tokens: List[NormalizedTokenDTO] = []
            for t in research_result.tokens:
                # Normalize pixel coordinates to [0.0, 1.0] range
                x1 = max(0.0, min(1.0, float(t.bbox.x_min) / float(width)))
                y1 = max(0.0, min(1.0, float(t.bbox.y_min) / float(height)))
                x2 = max(0.0, min(1.0, float(t.bbox.x_max) / float(width)))
                y2 = max(0.0, min(1.0, float(t.bbox.y_max) / float(height)))

                # Ensure non-inverted
                if x2 < x1:
                    x1, x2 = x2, x1
                if y2 < y1:
                    y1, y2 = y2, y1

                normalized_tokens.append(
                    NormalizedTokenDTO(
                        text=t.text,
                        confidence=float(t.confidence or 0.0),
                        bbox=[round(x1, 5), round(y1, 5), round(x2, 5), round(y2, 5)],
                        page_number=1,
                    )
                )

            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return NormalizedOCRResult(
                document_id=document_id,
                full_text=research_result.full_text,
                tokens=normalized_tokens,
                processing_time_ms=elapsed_ms,
                model_name="RapidOCR PP-OCRv6",
                metadata={
                    "width": width,
                    "height": height,
                    "engine_metadata": research_result.metadata,
                },
            )
        else:
            # Deterministic mock fallback if RapidOCR dependencies are unavailable
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            mock_tokens = [
                NormalizedTokenDTO(
                    text="RECEIPT SAMPLE",
                    confidence=0.98,
                    bbox=[0.1, 0.05, 0.9, 0.15],
                    page_number=1,
                ),
                NormalizedTokenDTO(
                    text="TOTAL 100.00",
                    confidence=0.95,
                    bbox=[0.1, 0.8, 0.9, 0.9],
                    page_number=1,
                ),
            ]
            return NormalizedOCRResult(
                document_id=document_id,
                full_text="RECEIPT SAMPLE\nTOTAL 100.00",
                tokens=mock_tokens,
                processing_time_ms=elapsed_ms,
                model_name="MockOCR",
                metadata={"width": width, "height": height, "fallback": True},
            )

    async def process(
        self,
        image_bytes: bytes,
        document_id: str,
        execution_provider: str = settings.OCR_EXECUTION_PROVIDER,
    ) -> NormalizedOCRResult:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _inference_executor, self._sync_process, image_bytes, document_id, execution_provider
        )


default_ocr_adapter = ResearchRapidOCRAdapter()
