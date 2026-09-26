"""Company field extraction rule for SROIE receipts."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.core.schemas import OCRToken
from src.kie.rules.candidate import FieldCandidate


def extract_company_candidate(
    tokens: List[OCRToken],
    config: Optional[Dict[str, Any]] = None,
) -> Optional[FieldCandidate]:
    """Extract candidate for 'company' field from OCR tokens.

    Heuristic principles:
    1. Company name appears in receipt header (first few lines / top vertical area).
    2. Often contains legal entity keywords (e.g. SDN BHD, BHD, LTD, ENTERPRISE, RESTAURANT).
    3. Excludes metadata lines (INVOICE, RECEIPT, TEL, FAX, GST NO, DATE, CASHIER).
    4. Heavy digits ratio is penalized.
    """
    if not tokens:
        return None

    cfg = config or {}
    comp_cfg = cfg.get("fields", {}).get("company", {})
    max_line_index = comp_cfg.get("header_max_line_index", 5)
    legal_keywords = [
        k.upper() for k in comp_cfg.get("legal_keywords", ["SDN BHD", "SDN. BHD.", "BHD", "LTD", "ENTERPRISE", "RESTAURANT"])
    ]
    negative_keywords = [
        k.upper() for k in comp_cfg.get("negative_keywords", ["INVOICE", "RECEIPT", "TAX", "TOTAL", "TEL", "FAX", "CASHIER", "GST NO"])
    ]
    min_len = comp_cfg.get("min_length", 3)
    max_len = comp_cfg.get("max_length", 100)

    # Group tokens by vertical lines or take first N tokens
    # In OCRResult, tokens are in reading order.
    candidates: List[FieldCandidate] = []

    # Consider tokens within the top window
    limit = min(len(tokens), max_line_index * 2)  # up to first few tokens
    for idx in range(limit):
        token = tokens[idx]
        text = token.text.strip()
        if len(text) < min_len or len(text) > max_len:
            continue

        text_upper = text.upper()

        # Check negative keywords
        if any(neg in text_upper for neg in negative_keywords):
            continue

        # Digits ratio check
        digit_count = sum(1 for c in text if c.isdigit())
        if len(text) > 0 and (digit_count / len(text)) > 0.35:
            continue

        # Position score (earlier in header is favored)
        pos_score = max(0.4, 0.85 - (idx * 0.08))

        # Legal keyword bonus
        has_legal = any(legal in text_upper for legal in legal_keywords)
        bonus = 0.25 if has_legal else 0.0

        # All-caps bonus
        caps_bonus = 0.05 if text.isupper() and len(text) > 4 else 0.0

        score = min(1.0, pos_score + bonus + caps_bonus)

        candidate = FieldCandidate(
            field_name="company",
            text=text,
            confidence=round(score, 4),
            token_indices=[idx],
            rule_name="header_legal_heuristic",
            metadata={"position_index": idx, "has_legal_keyword": has_legal},
        )
        candidates.append(candidate)

    if not candidates:
        return None

    # Return candidate with highest confidence
    # If equal confidence, stable sort preserves first encountered
    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return candidates[0]
