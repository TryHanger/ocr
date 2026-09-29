"""Centralized deterministic constants for MVP-10: Document AI Improvement Loop.

Ensures that all thresholds, bucket definitions, and sample size constraints,
are explicitly named and defined in one single location without scattered magic numbers.
"""

from typing import List, Tuple

# Minimum evaluated population required before generating research signals
MIN_SIGNAL_SAMPLE_SIZE: int = 5

# Minimum operator corrections required to constitute a notable pattern
MIN_CORRECTION_COUNT: int = 2

# Minimum fraction of corrections concentrated in a single confidence interval (25%)
MIN_BUCKET_SHARE: float = 0.25

# Minimum fraction of all field corrections attributed to a single field (20%)
MIN_FIELD_SHARE: float = 0.20

# Canonical confidence intervals: (bucket_label, lower_bound_inclusive, upper_bound_exclusive)
CONFIDENCE_BUCKETS: List[Tuple[str, float, float]] = [
    ("0.00–0.50", 0.00, 0.50),
    ("0.50–0.70", 0.50, 0.70),
    ("0.70–0.85", 0.70, 0.85),
    ("0.85–0.95", 0.85, 0.95),
    ("0.95–1.00", 0.95, 1.001),
]
