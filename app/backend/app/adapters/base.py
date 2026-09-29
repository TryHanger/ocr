"""Protocols and interfaces for ML, OCR, KIE, Quality, and Preprocessing adapters."""

from typing import Any, Dict, List, Protocol, Tuple, runtime_checkable

from app.schemas.kie import NormalizedKIEResult
from app.schemas.ocr import NormalizedOCRResult
from app.schemas.quality import QualityResultDTO
from app.schemas.research import ResearchSummaryResponse


@runtime_checkable
class OCRProvider(Protocol):
    """Abstract contract for OCR text detection and recognition engines."""

    async def process(
        self,
        image_bytes: bytes,
        document_id: str,
        execution_provider: str = "cpu",
    ) -> NormalizedOCRResult:
        ...


@runtime_checkable
class KIEProvider(Protocol):
    """Abstract contract for Key Information Extraction semantic engines."""

    async def extract(
        self,
        ocr_result: NormalizedOCRResult,
        document_id: str,
    ) -> NormalizedKIEResult:
        ...


@runtime_checkable
class QualityAnalyzer(Protocol):
    """Abstract contract for document scan quality evaluation."""

    async def analyze(
        self,
        image_bytes: bytes,
        document_id: str,
    ) -> QualityResultDTO:
        ...


@runtime_checkable
class PreprocessingProvider(Protocol):
    """Abstract contract for adaptive preprocessing pipelines."""

    async def process(
        self,
        image_bytes: bytes,
        quality: QualityResultDTO,
    ) -> Tuple[bytes, List[Dict[str, Any]]]:
        ...


@runtime_checkable
class ResearchMetricsProvider(Protocol):
    """Abstract contract for reading and presenting research experiment benchmarks."""

    async def get_summary(self) -> ResearchSummaryResponse:
        ...
