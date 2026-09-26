"""Key Information Extraction (KIE) module."""

from src.core.contracts import BaseKIEEngine
from src.kie.factory import get_kie_engine
from src.kie.mock import MockKIEEngine
from src.kie.rule_based import RuleBasedKIEEngine

__all__ = [
    "BaseKIEEngine",
    "RuleBasedKIEEngine",
    "MockKIEEngine",
    "get_kie_engine",
]
