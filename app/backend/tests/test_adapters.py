"""Tests for OCR and KIE Adapters ensuring coordinate normalization."""

import io
from PIL import Image, ImageDraw
import pytest

from app.adapters.kie_adapter import ResearchRuleBasedKIEAdapter
from app.adapters.ocr_adapter import ResearchRapidOCRAdapter
from app.adapters.quality_adapter import QualityAnalysisAdapter


def _make_test_image() -> bytes:
    img = Image.new("RGB", (400, 600), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), "SUPERMARKET SDN BHD", fill=(0, 0, 0))
    draw.text((20, 50), "DATE: 2026-09-28", fill=(0, 0, 0))
    draw.text((20, 500), "TOTAL: 88.00", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_ocr_adapter_normalization():
    adapter = ResearchRapidOCRAdapter(execution_provider="cpu")
    img_bytes = _make_test_image()

    res = await adapter.process(img_bytes, "test_doc_01")
    assert res.document_id == "test_doc_01"
    assert res.full_text != ""
    assert len(res.tokens) > 0

    for token in res.tokens:
        assert 0.0 <= token.confidence <= 1.0
        x1, y1, x2, y2 = token.bbox
        assert 0.0 <= x1 <= 1.0
        assert 0.0 <= y1 <= 1.0
        assert 0.0 <= x2 <= 1.0
        assert 0.0 <= y2 <= 1.0
        assert x2 >= x1
        assert y2 >= y1


@pytest.mark.asyncio
async def test_kie_adapter_and_provenance():
    ocr_adapter = ResearchRapidOCRAdapter(execution_provider="cpu")
    kie_adapter = ResearchRuleBasedKIEAdapter()
    img_bytes = _make_test_image()

    ocr_res = await ocr_adapter.process(img_bytes, "test_doc_02")
    kie_res = await kie_adapter.extract(ocr_res, "test_doc_02")

    assert kie_res.document_id == "test_doc_02"
    assert isinstance(kie_res.fields, dict)

    for field_name, f_dto in kie_res.fields.items():
        assert 0.0 <= f_dto.confidence <= 1.0
        if f_dto.bbox is not None:
            x1, y1, x2, y2 = f_dto.bbox
            assert 0.0 <= x1 <= 1.0
            assert 0.0 <= y1 <= 1.0
            assert 0.0 <= x2 <= 1.0
            assert 0.0 <= y2 <= 1.0


@pytest.mark.asyncio
async def test_quality_adapter():
    quality_adapter = QualityAnalysisAdapter()
    img_bytes = _make_test_image()

    q_res = await quality_adapter.analyze(img_bytes, "test_doc_03")
    assert 0.0 <= q_res.quality_score <= 1.0
    assert q_res.blur_score > 0
    assert q_res.contrast_score > 0
    assert "needs_deskew" in q_res.recommendations
