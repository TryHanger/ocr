"""Total amount field extraction rule for SROIE receipts."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from src.core.schemas import OCRToken
from src.kie.rules.candidate import FieldCandidate


def extract_total_candidate(
    tokens: List[OCRToken],
    config: Optional[Dict[str, Any]] = None,
) -> Optional[FieldCandidate]:
    """Extract candidate for 'total' field from OCR tokens.

    Heuristic principles:
    1. Looks for amount patterns (e.g. 45.00, 123.45).
    2. Requires association with total keywords (TOTAL, GRAND TOTAL, AMOUNT DUE, JUMLAH).
    3. Strictly filters / penalizes negative keywords (SUBTOTAL, CHANGE, CASH, GST, TAX, ROUNDING).
    4. fallback_largest_amount: false by default. If enabled via config, falls back to the largest
       detected amount with a heavy penalty applied to confidence.
    """
    if not tokens:
        return None

    cfg = config or {}
    total_cfg = cfg.get("fields", {}).get("total", {})
    fallback_enabled = cfg.get("engine", {}).get(
        "fallback_largest_amount",
        cfg.get("fallback_largest_amount", False),
    )
    fallback_penalty = cfg.get("engine", {}).get(
        "fallback_penalty",
        cfg.get("fallback_penalty", 0.5),
    )

    total_keywords = [
        k.upper() for k in total_cfg.get(
            "total_keywords",
            ["TOTAL", "GRAND TOTAL", "AMOUNT", "AMOUNT DUE", "BALANCE DUE", "NET TOTAL", "JUMLAH"],
        )
    ]
    negative_keywords = [
        k.upper() for k in total_cfg.get(
            "negative_keywords",
            ["SUBTOTAL", "SUB TOTAL", "CHANGE", "CASH", "ROUNDING", "GST", "TAX", "DISCOUNT", "ITEMS", "QTY"],
        )
    ]

    amount_regex = re.compile(r"(?:RM|MYR|\$)?\s*([0-9]{1,6}\.[0-9]{2})\b", re.IGNORECASE)

    keyword_candidates: List[FieldCandidate] = []
    all_numeric_amounts: List[tuple[float, str, int]] = []  # (val, text, token_idx)

    for idx, token in enumerate(tokens):
        text = token.text.strip()
        text_upper = text.upper()

        # Check negative keywords in the same token
        is_negative = any(neg in text_upper for neg in negative_keywords)

        # Check if current token or preceding token has total keywords
        has_total_keyword = any(kw in text_upper for kw in total_keywords)
        prev_has_total_kw = (
            any(kw in tokens[idx - 1].text.upper() for kw in total_keywords)
            and not any(neg in tokens[idx - 1].text.upper() for neg in negative_keywords)
            if idx > 0
            else False
        )

        # Find amounts in current token
        for match in amount_regex.finditer(text):
            amount_str = match.group(1)
            try:
                amount_val = float(amount_str)
            except ValueError:
                continue

            all_numeric_amounts.append((amount_val, amount_str, idx))

            if is_negative:
                continue

            if has_total_keyword or prev_has_total_kw:
                # Proximity to keyword gives strong score
                # Grand total or total keyword gives higher confidence
                is_grand = "GRAND" in text_upper or (idx > 0 and "GRAND" in tokens[idx - 1].text.upper())
                base_score = 0.95 if is_grand else 0.85
                token_indices = [idx - 1, idx] if prev_has_total_kw else [idx]

                keyword_candidates.append(
                    FieldCandidate(
                        field_name="total",
                        text=amount_str,
                        confidence=round(base_score, 4),
                        token_indices=token_indices,
                        rule_name="keyword_proximity",
                        metadata={"amount_value": amount_val, "is_grand_total": is_grand},
                    )
                )

    if keyword_candidates:
        # Prefer latest occurrence in document if multiple totals (e.g. summary at bottom)
        # or grand total
        keyword_candidates.sort(
            key=lambda c: (c.confidence, c.metadata.get("amount_value", 0.0)),
            reverse=True,
        )
        return keyword_candidates[0]

    # If no keyword-associated total found and fallback is enabled:
    if fallback_enabled and all_numeric_amounts:
        # Find maximum numeric amount
        all_numeric_amounts.sort(key=lambda x: x[0], reverse=True)
        max_val, max_str, token_idx = all_numeric_amounts[0]
        penalized_conf = max(0.0, min(1.0, 1.0 - fallback_penalty))

        return FieldCandidate(
            field_name="total",
            text=max_str,
            confidence=round(penalized_conf, 4),
            token_indices=[token_idx],
            rule_name="fallback_largest_amount",
            metadata={"amount_value": max_val, "fallback_used": True},
        )

    return None
