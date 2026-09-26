"""Address field extraction rule for SROIE receipts."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from src.core.schemas import OCRToken
from src.kie.rules.candidate import FieldCandidate


def extract_address_candidate(
    tokens: List[OCRToken],
    company_candidate: Optional[FieldCandidate] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[FieldCandidate]:
    """Extract candidate for 'address' field from OCR tokens.

    Heuristic principles:
    1. Usually located below the company name in the receipt header.
    2. Contains location indicators (JALAN, JLN, LOT, LORONG, STREET, ROAD, TAMAN, BANDAR,
       state/country names like SELANGOR, KUALA LUMPUR, MALAYSIA, PENANG) or 5-digit postcode.
    3. Excludes metadata lines (TEL, FAX, EMAIL, WWW, GST NO, CASHIER, DATE, TOTAL).
    4. May span multiple consecutive header lines.
    """
    if not tokens:
        return None

    cfg = config or {}
    addr_cfg = cfg.get("fields", {}).get("address", {})
    max_lines = addr_cfg.get("max_lines", 6)
    location_keywords = [
        k.upper()
        for k in addr_cfg.get(
            "location_keywords",
            ["NO.", "LOT", "JALAN", "JLN", "LORONG", "STREET", "ROAD", "TAMAN", "BANDAR", "SELANGOR", "KUALA LUMPUR", "MALAYSIA", "PENANG"],
        )
    ]
    postcode_pattern = re.compile(addr_cfg.get("postcode_regex", r"\b\d{5}\b"))
    negative_keywords = [
        k.upper()
        for k in addr_cfg.get(
            "negative_keywords",
            ["TEL", "FAX", "EMAIL", "WWW", "HTTP", "GST NO", "REG NO", "ROC", "TAX INVOICE", "CASHIER", "TOTAL"],
        )
    ]

    # Determine starting index based on company candidate token index
    start_idx = 0
    if company_candidate and company_candidate.token_indices:
        start_idx = max(company_candidate.token_indices) + 1

    # Window of tokens to search: header region after company
    search_limit = min(len(tokens), start_idx + max_lines * 3)

    collected_tokens: List[tuple[int, str]] = []  # (index, text)
    matched_location_features = 0

    for idx in range(start_idx, search_limit):
        token = tokens[idx]
        text = token.text.strip()
        text_upper = text.upper()

        # Stop if we hit transaction items or total
        if any(kw in text_upper for kw in ["TOTAL", "SUBTOTAL", "AMOUNT DUE", "RECEIPT", "INVOICE NO"]):
            break

        # Skip negative keywords
        if any(neg in text_upper for neg in negative_keywords):
            continue

        # Check for address signals
        has_loc_kw = any(kw in text_upper for kw in location_keywords)
        has_postcode = bool(postcode_pattern.search(text))

        if has_loc_kw or has_postcode:
            collected_tokens.append((idx, text))
            matched_location_features += (1 if has_loc_kw else 0) + (1 if has_postcode else 0)
        elif collected_tokens:
            # If we already started collecting an address block and this line looks like
            # continuation (not financial, not phone/tax, reasonable length)
            digit_count = sum(1 for c in text if c.isdigit())
            is_phone = "TEL" in text_upper or (len(text) > 0 and digit_count / len(text) > 0.6)
            if not is_phone and len(text) > 3 and len(collected_tokens) < max_lines:
                collected_tokens.append((idx, text))

    if not collected_tokens:
        # Fallback: scan any token across the entire receipt that has explicit address keywords
        for idx, token in enumerate(tokens):
            text = token.text.strip()
            text_upper = text.upper()
            if any(neg in text_upper for neg in negative_keywords):
                continue
            has_loc_kw = any(kw in text_upper for kw in location_keywords)
            has_postcode = bool(postcode_pattern.search(text))
            if has_loc_kw or has_postcode:
                collected_tokens.append((idx, text))
                matched_location_features += 1
                if len(collected_tokens) >= 3:
                    break

    if not collected_tokens:
        return None

    # Join collected lines cleanly
    parts = [t[1] for t in collected_tokens]
    # Join with space if ends with comma, otherwise comma-space if sensible
    joined_text = parts[0]
    for p in parts[1:]:
        if joined_text.endswith(",") or joined_text.endswith("-"):
            joined_text = f"{joined_text} {p}"
        elif p.startswith(","):
            joined_text = f"{joined_text}{p}"
        else:
            joined_text = f"{joined_text}, {p}"

    score = min(1.0, 0.65 + min(0.30, matched_location_features * 0.10))
    token_indices = [t[0] for t in collected_tokens]

    return FieldCandidate(
        field_name="address",
        text=joined_text,
        confidence=round(score, 4),
        token_indices=token_indices,
        rule_name="header_address_block",
        metadata={
            "num_lines": len(collected_tokens),
            "matched_features": matched_location_features,
        },
    )
