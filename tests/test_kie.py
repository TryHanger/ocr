"""Unit and integration tests for KIE Baseline Engine and heuristic rules."""

from __future__ import annotations

import inspect
import pytest

from src.core.contracts import BaseKIEEngine
from src.core.schemas import BoundingBox, KIEResult, OCRResult, OCRToken
from src.kie.factory import get_kie_engine
from src.kie.mock import MockKIEEngine
from src.kie.rule_based import RuleBasedKIEEngine
from src.kie.rules.address import extract_address_candidate
from src.kie.rules.candidate import FieldCandidate
from src.kie.rules.company import extract_company_candidate
from src.kie.rules.date import extract_date_candidate
from src.kie.rules.total import extract_total_candidate


def _make_token(text: str, ymin: int = 10, ymax: int = 30) -> OCRToken:
    """Helper to create dummy OCRToken."""
    return OCRToken(
        text=text,
        bbox=BoundingBox(10, ymin, 100, ymax),
        confidence=0.95,
    )


# --- 1. Candidate Extraction Rules Tests ---

def test_extract_company_candidate_legal_keywords():
    tokens = [
        _make_token("INVOICE 12345", 10, 20),
        _make_token("POPULAR BOOK STORE SDN BHD", 25, 45),
        _make_token("TEL: 03-12345678", 50, 65),
    ]
    cand = extract_company_candidate(tokens)
    assert cand is not None
    assert cand.field_name == "company"
    assert "POPULAR BOOK STORE SDN BHD" in cand.text
    assert cand.confidence > 0.7
    assert cand.token_indices == [1]


def test_extract_company_candidate_negative_filtering():
    tokens = [
        _make_token("TAX INVOICE", 10, 20),
        _make_token("CASHIER: ALICE", 25, 35),
    ]
    cand = extract_company_candidate(tokens)
    assert cand is None


def test_extract_date_candidate_regexes():
    # DMY pattern
    tokens_dmy = [_make_token("Date: 25/12/2018")]
    cand_dmy = extract_date_candidate(tokens_dmy)
    assert cand_dmy is not None
    assert cand_dmy.text == "25/12/2018"
    assert cand_dmy.confidence > 0.7

    # YMD pattern
    tokens_ymd = [_make_token("TARIKH: 2019-05-14")]
    cand_ymd = extract_date_candidate(tokens_ymd)
    assert cand_ymd is not None
    assert cand_ymd.text == "2019-05-14"

    # Named month pattern
    tokens_named = [_make_token("15-Jan-2020")]
    cand_named = extract_date_candidate(tokens_named)
    assert cand_named is not None
    assert "15-Jan-2020" in cand_named.text


def test_extract_date_candidate_invalid_calendar():
    # Invalid month 15 and invalid day 40
    tokens = [_make_token("40/15/2018")]
    cand = extract_date_candidate(tokens)
    assert cand is None


def test_extract_total_candidate_proximity():
    tokens = [
        _make_token("ITEM 1 10.00"),
        _make_token("SUBTOTAL 10.00"),
        _make_token("ROUNDING 0.05"),
        _make_token("TOTAL 10.05"),
        _make_token("CASH 20.00"),
        _make_token("CHANGE 9.95"),
    ]
    cand = extract_total_candidate(tokens)
    assert cand is not None
    assert cand.field_name == "total"
    assert cand.text == "10.05"
    assert cand.confidence >= 0.85
    assert cand.rule_name == "keyword_proximity"


def test_extract_total_fallback_toggle():
    tokens = [
        _make_token("ITEM 1 15.00"),
        _make_token("ITEM 2 25.00"),
    ]
    # Default: fallback_largest_amount is False
    cand_default = extract_total_candidate(tokens, config={"engine": {"fallback_largest_amount": False}})
    assert cand_default is None

    # Ablation: fallback_largest_amount is True
    cand_fallback = extract_total_candidate(
        tokens,
        config={"engine": {"fallback_largest_amount": True, "fallback_penalty": 0.5}},
    )
    assert cand_fallback is not None
    assert cand_fallback.text == "25.00"
    assert cand_fallback.confidence == pytest.approx(0.5, abs=1e-4)
    assert cand_fallback.rule_name == "fallback_largest_amount"


def test_extract_address_candidate_multi_line():
    comp_cand = FieldCandidate("company", "STORE BHD", 0.9, token_indices=[0])
    tokens = [
        _make_token("STORE BHD"),
        _make_token("LOT 123, JALAN AMPANG"),
        _make_token("50450 KUALA LUMPUR, MALAYSIA"),
        _make_token("TEL: 03-99998888"),
        _make_token("TOTAL 50.00"),
    ]
    cand = extract_address_candidate(tokens, company_candidate=comp_cand)
    assert cand is not None
    assert "JALAN AMPANG" in cand.text
    assert "KUALA LUMPUR" in cand.text
    assert "MALAYSIA" in cand.text
    assert cand.token_indices == [1, 2]


# --- 2. RuleBasedKIEEngine Contract and API Tests ---

def test_kie_engine_contract_signature():
    """Verify strictly OCRResult-only API: extract(ocr_result, document_id) -> KIEResult."""
    sig = inspect.signature(RuleBasedKIEEngine.extract)
    param_names = list(sig.parameters.keys())
    assert param_names == ["self", "ocr_result", "document_id"]
    assert "image" not in param_names


def test_rule_based_kie_empty_ocr():
    engine = RuleBasedKIEEngine()
    empty_ocr = OCRResult(document_id="empty_doc", full_text="", tokens=[])
    res = engine.extract(empty_ocr, "empty_doc")
    assert isinstance(res, KIEResult)
    assert res.document_id == "empty_doc"
    assert res.fields == {}
    assert res.confidences == {}
    assert res.metadata["status"] == "EMPTY_OCR"


def test_rule_based_kie_input_validation():
    engine = RuleBasedKIEEngine()
    ocr = OCRResult(document_id="d1", full_text="test", tokens=[])

    with pytest.raises(TypeError, match="ocr_result must be OCRResult"):
        engine.extract("not_ocr_result", "d1")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="document_id must be a non-empty string"):
        engine.extract(ocr, "")

    with pytest.raises(ValueError, match="document_id mismatch"):
        engine.extract(ocr, "d2")


def test_rule_based_kie_full_receipt_extraction():
    engine = RuleBasedKIEEngine()
    tokens = [
        _make_token("BOOK STORE SDN BHD"),
        _make_token("LOT 1, JLN SULTAN"),
        _make_token("50000 KUALA LUMPUR"),
        _make_token("DATE: 25/12/2018"),
        _make_token("TOTAL 45.00"),
    ]
    ocr = OCRResult(
        document_id="receipt_01",
        full_text="\n".join(t.text for t in tokens),
        tokens=tokens,
    )
    res = engine.extract(ocr, "receipt_01")
    assert isinstance(res, KIEResult)
    assert res.document_id == "receipt_01"
    assert "company" in res.fields
    assert "BOOK STORE SDN BHD" in res.fields["company"]
    assert res.fields["date"] == "25/12/2018"
    assert "KUALA LUMPUR" in res.fields["address"]
    assert res.fields["total"] == "45.00"

    # Provenance tracking
    assert "field_provenance" in res.metadata
    assert res.metadata["field_provenance"]["total"] == [4]


def test_rule_based_kie_determinism():
    engine = RuleBasedKIEEngine()
    tokens = [
        _make_token("MY CAFE SDN BHD"),
        _make_token("PENANG, MALAYSIA"),
        _make_token("01/01/2021"),
        _make_token("TOTAL RM 100.00"),
    ]
    ocr = OCRResult(
        document_id="det_doc",
        full_text="\n".join(t.text for t in tokens),
        tokens=tokens,
    )
    res1 = engine.extract(ocr, "det_doc")
    res2 = engine.extract(ocr, "det_doc")
    assert res1.fields == res2.fields
    assert res1.confidences == res2.confidences
    assert res1.metadata["field_provenance"] == res2.metadata["field_provenance"]


# --- 3. Mock and Factory Tests ---

def test_mock_kie_engine():
    mock = MockKIEEngine(canned_fields={"d1": {"company": "ACME", "total": "5.00"}})
    ocr1 = OCRResult(document_id="d1", full_text="", tokens=[])
    res1 = mock.extract(ocr1, "d1")
    assert res1.fields == {"company": "ACME", "total": "5.00"}

    ocr2 = OCRResult(document_id="unknown_doc", full_text="", tokens=[])
    res2 = mock.extract(ocr2, "unknown_doc")
    assert "company" in res2.fields


def test_factory_get_kie_engine():
    engine1 = get_kie_engine("rule_based")
    assert isinstance(engine1, RuleBasedKIEEngine)

    engine2 = get_kie_engine("mock")
    assert isinstance(engine2, MockKIEEngine)

    with pytest.raises(ValueError, match="Unsupported KIE engine"):
        get_kie_engine("unsupported_engine")


def test_extract_address_fallback_scan():
    # When company candidate index is after address tokens
    comp_cand = FieldCandidate("company", "LATE STORE", 0.9, token_indices=[5])
    tokens = [
        _make_token("TEL: 03-12345678"),
        _make_token("JALAN PETALING, KUALA LUMPUR"),
        _make_token("MALAYSIA"),
        _make_token("TOTAL 10.00"),
    ]
    cand = extract_address_candidate(tokens, company_candidate=comp_cand)
    assert cand is not None
    assert "JALAN PETALING" in cand.text


def test_rule_based_kie_text_only_tokens():
    engine = RuleBasedKIEEngine()
    ocr = OCRResult(
        document_id="text_only_doc",
        full_text="SUPERMARKET SDN BHD\n10/10/2020\nTOTAL 99.90",
        tokens=[],
    )
    res = engine.extract(ocr, "text_only_doc")
    assert "company" in res.fields
    assert "SUPERMARKET SDN BHD" in res.fields["company"]
    assert res.fields["date"] == "10/10/2020"
    assert res.fields["total"] == "99.90"

