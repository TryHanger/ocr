"""Smart Preprocessing Adapter delegating to research-backed PreprocessingPolicy."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

from app.adapters.base import PreprocessingProvider
from app.schemas.quality import QualityResultDTO
from app.services.preprocessing_policy import PreprocessingPolicy

_prep_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="prep_pipeline")


class ResearchPreprocessingAdapter(PreprocessingProvider):
    """Applies smart preprocessing based on research findings and quality recommendations."""

    def _sync_process(
        self, image_bytes: bytes, quality: QualityResultDTO
    ) -> Tuple[bytes, List[Dict[str, Any]]]:
        return PreprocessingPolicy.execute_plan(image_bytes, quality)

    async def process(
        self, image_bytes: bytes, quality: QualityResultDTO
    ) -> Tuple[bytes, List[Dict[str, Any]]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _prep_executor, self._sync_process, image_bytes, quality
        )


default_prep_adapter = ResearchPreprocessingAdapter()
