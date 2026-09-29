"""Research Mode Schemas."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class BaselineMetricDTO(BaseModel):
    cer_normalized: float
    char_ned_normalized: float
    macro_f1_normalized: float
    total_evaluated: int


class DegradationImpactItem(BaseModel):
    degradation: str
    severity: int
    cer_mean: float
    ned_mean: float
    macro_f1_mean: float


class PreprocessingRecoveryItem(BaseModel):
    condition_id: str
    degradation: str
    severity: int
    preprocessing: str
    delta_cer_recovery: float
    delta_f1_recovery: float


class ResearchSummaryResponse(BaseModel):
    b0_clean_baseline: BaselineMetricDTO
    b1_degradations: List[DegradationImpactItem]
    b2_top_recoveries: List[PreprocessingRecoveryItem]


class ResearchTrackSummary(BaseModel):
    id: str
    name: str
    description: str
    experiment_count: int
    key_metrics: Dict[str, Any]


class ResearchOverviewResponse(BaseModel):
    tracks: List[ResearchTrackSummary]
    total_experiments: int
    canonical_dataset: str
    execution_provider: str


class ResearchExperimentItem(BaseModel):
    id: str
    track: str
    name: str
    description: str
    dataset: str
    status: str
    metrics_summary: Dict[str, Any]
    source: str


class ResearchExperimentsResponse(BaseModel):
    items: List[ResearchExperimentItem]
    total: int


class ResearchConditionMetric(BaseModel):
    condition_id: str
    name: str
    severity: Optional[int] = None
    cer: Optional[float] = None
    wer: Optional[float] = None
    ned: Optional[float] = None
    f1: Optional[float] = None
    delta_cer: Optional[float] = None
    delta_f1: Optional[float] = None
    preprocessing: Optional[str] = None


class ResearchExperimentDetail(BaseModel):
    id: str
    name: str
    track: str
    description: str
    status: str
    dataset: str
    sample_size: int
    source: str
    metrics_summary: Dict[str, Any]
    conditions: List[ResearchConditionMetric] = []
    ocr_metrics: Optional[Dict[str, Any]] = None
    kie_metrics: Optional[Dict[str, Any]] = None
    findings: List[str] = []
    metadata: Dict[str, Any] = {}


class DegradationCondition(BaseModel):
    condition_id: str
    severity: str
    parameters: Optional[Dict[str, Any]] = None
    parameter: Optional[str] = None
    cer: float
    wer: float
    ned: float
    macro_f1: float
    entity_hmean: float
    delta_cer: float
    delta_wer: Optional[float] = 0.0
    delta_ned: float
    delta_macro_f1: float
    delta_entity_hmean: float


class DegradationExplorerItem(BaseModel):
    id: str
    code: str
    name: str
    description: str
    dataset: str
    sample_size: int
    conditions: List[DegradationCondition]
    findings: List[str]
    source: str
    calibration_source: Optional[str] = None


class DegradationExplorerResponse(BaseModel):
    items: List[DegradationExplorerItem]
    total: int
    control_condition: DegradationCondition
    canonical_dataset: str


class PreprocessingMetrics(BaseModel):
    cer: float
    wer: Optional[float] = None
    ned: float
    macro_f1: float
    entity_hmean: float
    doc_em: Optional[float] = None


class PreprocessingPolicyMetric(BaseModel):
    condition_id: str
    policy_id: str
    policy_name: str
    policy_key: str
    metrics: PreprocessingMetrics
    delta_cer_vs_degraded: float
    delta_wer_vs_degraded: Optional[float] = None
    delta_ned_vs_degraded: float
    delta_macro_f1_vs_degraded: float
    delta_entity_hmean_vs_degraded: float
    delta_cer_vs_clean: float
    delta_macro_f1_vs_clean: float
    relative_recovery_rate: Optional[float] = None


class PreprocessingConditionGroup(BaseModel):
    degradation_code: str
    degradation_name: str
    degradation_key: str
    severity: str
    ref_b1_condition_id: str
    clean_control_metrics: PreprocessingMetrics
    degraded_metrics: PreprocessingMetrics
    policies: List[PreprocessingPolicyMetric]
    findings: List[str]


class PreprocessingExplorerResponse(BaseModel):
    groups: List[PreprocessingConditionGroup]
    total_groups: int
    degradations: List[Dict[str, str]]
    severities: List[str]
    policies: List[Dict[str, str]]
    canonical_dataset: str
    total_conditions_evaluated: int
    source: str


class ResearchDegradationRef(BaseModel):
    code: str
    name: str


class TrackEvidenceRef(BaseModel):
    experiment_id: Optional[str] = None
    available: bool = False
    details: Optional[str] = None


class ResearchEvidenceReference(BaseModel):
    """Reference linking an observed production quality characteristic to controlled research evidence."""

    production_signal: str
    observed_value: Optional[float] = None
    unit: Optional[str] = None

    research_degradation_code: str
    research_degradation_name: str

    b1_experiment_id: Optional[str] = None
    b1_available: bool = False

    b2_degradation_code: Optional[str] = None
    b2_available: bool = False
    b2_evaluated_policies_count: int = 4

    # Nested structures for flexible serialization
    research_degradation: Optional[ResearchDegradationRef] = None
    b1: Optional[TrackEvidenceRef] = None
    b2: Optional[TrackEvidenceRef] = None


