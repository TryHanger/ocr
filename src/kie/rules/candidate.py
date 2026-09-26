"""Candidate representations for rule-based Key Information Extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class FieldCandidate:
    """Represents an extracted candidate for a specific KIE field."""

    field_name: str
    text: str
    confidence: float
    token_indices: List[int] = field(default_factory=list)
    rule_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
