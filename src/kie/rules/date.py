"""Date field extraction rule for SROIE receipts."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from src.core.schemas import OCRToken
from src.kie.rules.candidate import FieldCandidate

# Month abbreviation lookup
MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _is_valid_date(d: int, m: int, y: int, min_year: int = 1990, max_year: int = 2035) -> bool:
    """Validate plausible day, month, year ranges."""
    if y < 100:
        y += 2000
    if not (min_year <= y <= max_year):
        return False
    if not (1 <= m <= 12):
        return False
    if not (1 <= d <= 31):
        return False
    return True


def extract_date_candidate(
    tokens: List[OCRToken],
    config: Optional[Dict[str, Any]] = None,
) -> Optional[FieldCandidate]:
    """Extract candidate for 'date' field from OCR tokens using regex and heuristics.

    Heuristic principles:
    1. Looks for standardized date patterns: DD/MM/YYYY, YYYY/MM/DD, DD-MMM-YYYY.
    2. Validates calendar reasonableness (1990-2035).
    3. Rewards proximity or presence of DATE/TARIKH keyword.
    4. Extracts the precise date substring rather than full line noise.
    """
    if not tokens:
        return None

    cfg = config or {}
    date_cfg = cfg.get("fields", {}).get("date", {})
    min_year = date_cfg.get("min_year", 1990)
    max_year = date_cfg.get("max_year", 2035)
    date_keywords = [k.upper() for k in date_cfg.get("date_keywords", ["DATE", "TARIKH", "TIME"])]

    # Regex definitions
    pattern_dmy = re.compile(r"(?<!\d)(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{2,4})(?!\d)")
    pattern_ymd = re.compile(r"(?<!\d)(\d{4})[/\-\.](\d{1,2})[/\-\.](\d{1,2})(?!\d)")
    pattern_named_month = re.compile(
        r"(?<!\d)(\d{1,2})[\s\-\.](Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s\-\.](\d{2,4})(?!\d)",
        re.IGNORECASE,
    )

    candidates: List[FieldCandidate] = []

    for idx, token in enumerate(tokens):
        text = token.text.strip()
        text_upper = text.upper()
        has_keyword = any(kw in text_upper for kw in date_keywords)

        # Check neighbor tokens for date keyword
        prev_has_keyword = (
            any(kw in tokens[idx - 1].text.upper() for kw in date_keywords)
            if idx > 0
            else False
        )

        keyword_bonus = 0.25 if (has_keyword or prev_has_keyword) else 0.0

        # Try named month first (e.g. 25 Dec 2018)
        for match in pattern_named_month.finditer(text):
            day = int(match.group(1))
            month_str = match.group(2).lower()[:3]
            month = MONTH_MAP.get(month_str, 0)
            year = int(match.group(3))
            if _is_valid_date(day, month, year, min_year, max_year):
                score = min(1.0, 0.75 + keyword_bonus)
                candidates.append(
                    FieldCandidate(
                        field_name="date",
                        text=match.group(0).strip(),
                        confidence=round(score, 4),
                        token_indices=[idx],
                        rule_name="regex_named_month",
                        metadata={"day": day, "month": month, "year": year},
                    )
                )

        # Try YMD pattern (e.g. 2018-12-25)
        for match in pattern_ymd.finditer(text):
            year = int(match.group(1))
            month = int(match.group(2))
            day = int(match.group(3))
            if _is_valid_date(day, month, year, min_year, max_year):
                score = min(1.0, 0.70 + keyword_bonus)
                candidates.append(
                    FieldCandidate(
                        field_name="date",
                        text=match.group(0).strip(),
                        confidence=round(score, 4),
                        token_indices=[idx],
                        rule_name="regex_ymd",
                        metadata={"day": day, "month": month, "year": year},
                    )
                )

        # Try DMY pattern (e.g. 25/12/2018 or 25-12-2018)
        for match in pattern_dmy.finditer(text):
            # If it already matched YMD or named month, avoid duplicates
            matched_str = match.group(0).strip()
            # Distinguish: could be DD/MM/YYYY or MM/DD/YYYY
            p1 = int(match.group(1))
            p2 = int(match.group(2))
            year = int(match.group(3))

            # Validate at least one valid interpretation
            valid_dmy = _is_valid_date(p1, p2, year, min_year, max_year)
            valid_mdy = _is_valid_date(p2, p1, year, min_year, max_year)

            if valid_dmy or valid_mdy:
                score = min(1.0, 0.70 + keyword_bonus)
                candidates.append(
                    FieldCandidate(
                        field_name="date",
                        text=matched_str,
                        confidence=round(score, 4),
                        token_indices=[idx],
                        rule_name="regex_dmy",
                        metadata={"p1": p1, "p2": p2, "year": year},
                    )
                )

    if not candidates:
        return None

    # Sort descending by confidence; stable sort preserves order
    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return candidates[0]
