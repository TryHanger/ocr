"""Factory function for creating KIE engines."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

from src.core.contracts import BaseKIEEngine
from src.kie.mock import MockKIEEngine
from src.kie.rule_based import RuleBasedKIEEngine


def get_kie_engine(
    engine_name: str = "rule_based",
    config: Optional[Union[Dict[str, Any], str]] = None,
    **kwargs: Any,
) -> BaseKIEEngine:
    """Instantiate and return a KIE engine by name.

    Args:
        engine_name: One of 'rule_based', 'mock'.
        config: Path to configuration file or dictionary.
        **kwargs: Additional parameters passed to engine constructor.

    Returns:
        Instance conforming to BaseKIEEngine contract.

    Raises:
        ValueError: If engine_name is unsupported.
    """
    normalized_name = engine_name.lower().strip()
    if normalized_name in ("rule_based", "rules", "sroie_rule_based"):
        return RuleBasedKIEEngine(config=config, **kwargs)
    elif normalized_name in ("mock", "dummy"):
        return MockKIEEngine(**kwargs)
    else:
        raise ValueError(
            f"Unsupported KIE engine '{engine_name}'. Supported engines: ['rule_based', 'mock']"
        )
