"""Experiment condition representations and matrix expansion logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# Canonical mapping from degradation type name to short D-code
DEGRADATION_CODE_MAP = {
    "none": "D0",
    "control": "D0",
    "gaussian_blur": "D1",
    "motion_blur": "D2",
    "gaussian_noise": "D3",
    "jpeg_compression": "D4",
    "downsampling": "D5",
    "rotation": "D6",
    "perspective": "D7",
    "shadow": "D8",
}

# Reverse lookup for canonical types
CODE_TO_DEGRADATION_MAP = {v: k for k, v in DEGRADATION_CODE_MAP.items() if k not in ("control",)}


@dataclass(frozen=True)
class ExperimentCondition:
    """Represents a single evaluated experimental condition (cell in the matrix)."""

    condition_id: str
    degradation_type: str
    severity: int
    preprocessing_id: str
    baseline_type: str  # "B0", "B1", "B2"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity == 0:
            if self.degradation_type != "none" or self.baseline_type != "B0":
                raise ValueError(
                    f"Severity 0 is strictly reserved for control condition 'D0_S0_P0' (B0). "
                    f"Got degradation='{self.degradation_type}', baseline='{self.baseline_type}'"
                )
        elif self.severity in (1, 2, 3, 4):
            if self.degradation_type == "none":
                raise ValueError("Degradation type cannot be 'none' when severity > 0")
        else:
            raise ValueError(f"Severity must be in [0, 1, 2, 3, 4], got {self.severity}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "degradation_type": self.degradation_type,
            "severity": self.severity,
            "preprocessing_id": self.preprocessing_id,
            "baseline_type": self.baseline_type,
            "metadata": self.metadata,
        }


def format_preprocessing_code(prep_id: str) -> str:
    """Format preprocessing identifier for condition ID."""
    p_norm = prep_id.strip()
    if p_norm.lower() in ("none", "p0", "raw", ""):
        return "P0"
    if p_norm.lower().startswith("p_"):
        return f"P_{p_norm[2:]}"
    if p_norm.startswith("P_"):
        return p_norm
    return f"P_{p_norm}"


def build_condition_id(degradation_type: str, severity: int, preprocessing_id: str) -> str:
    """Generate deterministic, human-readable condition ID (e.g. D0_S0_P0, D1_S1_P_clahe)."""
    d_code = DEGRADATION_CODE_MAP.get(degradation_type.lower().strip(), degradation_type.upper())
    s_code = f"S{severity}"
    p_code = format_preprocessing_code(preprocessing_id)
    return f"{d_code}_{s_code}_{p_code}"


def build_conditions_matrix(
    config: Dict[str, Any],
    smoke_mode: bool = False,
) -> List[ExperimentCondition]:
    """Construct deterministic experiment conditions matrix from configuration.

    Rules:
    1. Exactly one control condition: D0_S0_P0 (baseline_type="B0").
    2. B1 (degradation-only): D1..D8 × severity {1..4} × P0 (baseline_type="B1").
    3. B2 (degradation + preprocessing): D1..D8 × severity {1..4} × P* (baseline_type="B2").
    4. NEVER generate D1..D8 × severity 0 (severity 0 is reserved for D0_S0_P0).
    5. In smoke_mode, returns the explicit minimal CPU smoke matrix:
       - D0_S0_P0
       - D1_S1_P0
       - D1_S4_P0
       - D1_S1_P_clahe
       - D1_S4_P_clahe
    """
    conditions: List[ExperimentCondition] = []

    if smoke_mode:
        # Minimal CPU smoke matrix per TASK-010-R1 Section 3
        smoke_defs = [
            ("none", 0, "none", "B0"),
            ("gaussian_blur", 1, "none", "B1"),
            ("gaussian_blur", 4, "none", "B1"),
            ("gaussian_blur", 1, "p_clahe", "B2"),
            ("gaussian_blur", 4, "p_clahe", "B2"),
        ]
        for deg, sev, prep, b_type in smoke_defs:
            cid = build_condition_id(deg, sev, prep)
            conditions.append(
                ExperimentCondition(
                    condition_id=cid,
                    degradation_type=deg,
                    severity=sev,
                    preprocessing_id=prep,
                    baseline_type=b_type,
                    metadata={"smoke_condition": True},
                )
            )
        return conditions

    # Full Matrix Construction
    matrix_cfg = config.get("matrix", config)

    # 0. Explicit Conditions Override (e.g. for custom smoke configurations)
    explicit_conditions = matrix_cfg.get("explicit_conditions")
    if explicit_conditions:
        for item in explicit_conditions:
            deg = item.get("degradation", item.get("degradation_type", "none"))
            sev = int(item.get("severity", 0))
            prep = item.get("preprocessing", item.get("preprocessing_id", "none"))
            b_type = item.get("baseline_type")
            if not b_type:
                if sev == 0:
                    b_type = "B0"
                elif prep.lower() in ("none", "p0", "raw", ""):
                    b_type = "B1"
                else:
                    b_type = "B2"
            cid = build_condition_id(deg, sev, prep)
            conditions.append(
                ExperimentCondition(
                    condition_id=cid,
                    degradation_type=deg,
                    severity=sev,
                    preprocessing_id=prep,
                    baseline_type=b_type,
                    metadata=item.get("metadata", {}),
                )
            )
        return conditions

    # 1. Control Condition (B0): Exactly one D0_S0_P0
    include_control = matrix_cfg.get("include_control", True)
    if include_control:
        conditions.append(
            ExperimentCondition(
                condition_id="D0_S0_P0",
                degradation_type="none",
                severity=0,
                preprocessing_id="none",
                baseline_type="B0",
                metadata={"is_control": True},
            )
        )

    # Extract configured degradations, severities, and preprocessing pipelines
    degradations: List[str] = matrix_cfg.get(
        "degradations",
        ["gaussian_blur", "motion_blur", "gaussian_noise", "jpeg_compression", "downsampling", "rotation", "perspective", "shadow"],
    )
    # Severity 0 must NOT be in the degradation severities list
    configured_severities: Sequence[int] = matrix_cfg.get("severities", [1, 2, 3, 4])
    severities = [s for s in configured_severities if s in (1, 2, 3, 4)]

    # Preprocessing pipelines for B2
    preprocessing_pipelines: Dict[str, Any] = matrix_cfg.get("preprocessing_pipelines", {})

    # 2. B1: Degradation-only conditions (D1..D8 × severity {1..4} × P0)
    include_b1 = matrix_cfg.get("include_b1", True)
    if include_b1:
        for deg in degradations:
            if deg.lower() in ("none", "control"):
                continue
            for sev in severities:
                cid = build_condition_id(deg, sev, "none")
                conditions.append(
                    ExperimentCondition(
                        condition_id=cid,
                        degradation_type=deg,
                        severity=sev,
                        preprocessing_id="none",
                        baseline_type="B1",
                        metadata={"is_degraded": True},
                    )
                )

    # 3. B2: Degradation + Preprocessing conditions (D1..D8 × severity {1..4} × P*)
    for prep_id in sorted(preprocessing_pipelines.keys()):
        if prep_id.lower() in ("none", "p0", "raw"):
            continue
        for deg in degradations:
            if deg.lower() in ("none", "control"):
                continue
            for sev in severities:
                cid = build_condition_id(deg, sev, prep_id)
                conditions.append(
                    ExperimentCondition(
                        condition_id=cid,
                        degradation_type=deg,
                        severity=sev,
                        preprocessing_id=prep_id,
                        baseline_type="B2",
                        metadata={"is_degraded": True, "is_preprocessed": True},
                    )
                )

    return conditions
