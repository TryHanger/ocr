export type DocumentStatus =
  | "uploaded"
  | "processing"
  | "quality_analyzed"
  | "preprocessed"
  | "ocr_completed"
  | "kie_completed"
  | "validated"
  | "decision_made"
  | "completed_automatic"
  | "completed_manual"
  | "manual_review"
  | "error";

export type AutomationDecision = "automatic" | "manual_review" | "rejected";

export interface NormalizedToken {
  text: string;
  confidence: number;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2] in 0..1
  page_number: number;
}

export interface DocumentPage {
  id: string;
  page_number: number;
  width: number;
  height: number;
  image_url: string;
  full_text?: string;
  tokens: NormalizedToken[];
}

export interface NormalizedField {
  field_name: string;
  value: string;
  original_value?: string;
  corrected_value?: string;
  is_corrected?: boolean;
  confidence: number;
  bbox?: [number, number, number, number]; // [x1, y1, x2, y2] in 0..1
  source_token_indices: number[];
  validation_status: string;
  rule_applied?: string;
}

export interface ProcessingStep {
  id: string;
  step_order: number;
  operation: string;
  parameters?: Record<string, any>;
  duration_ms: number;
  status: string;
  created_at: string;
}

export interface QualityResult {
  document_id: string;
  blur_score: number;
  contrast_score: number;
  noise_score: number;
  rotation_angle: number;
  resolution_dpi: number;
  quality_score: number;
  profile_summary: string;
  metrics: Record<string, any>;
  recommendations: Record<string, boolean>;
}

export interface AuditEvent {
  id: string;
  actor: string;
  action: string;
  field_name?: string;
  old_value?: string;
  new_value?: string;
  created_at: string;
  metadata: Record<string, any>;
}

export interface ReviewDetails {
  reasons: string[];
  validation_errors: Array<{
    field?: string;
    message: string;
    code?: string;
  }>;
  evaluated_at?: string;
}

export interface DocumentItem {
  id: string;
  filename: string;
  document_type: string;
  status: DocumentStatus;
  decision?: AutomationDecision;
  confidence?: number;
  review_reason?: string;
  review_details?: ReviewDetails;
  processing_started_at?: string;
  processing_finished_at?: string;
  review_started_at?: string;
  review_finished_at?: string;
  review_duration_ms?: number;
  review_wait_duration_ms?: number;
  created_at: string;
  updated_at: string;
  processing_duration_ms?: number;
}

export interface DocumentDetail extends DocumentItem {
  file_size_bytes: number;
  mime_type: string;
  file_url: string;
  pages: DocumentPage[];
  fields: Record<string, NormalizedField>;
  quality?: QualityResult;
  audit_events: AuditEvent[];
  steps?: ProcessingStep[];
  research_evidence?: ResearchEvidenceReference[];
  decision_explanation?: DecisionExplanation;
  field_corrections?: DocumentFieldCorrection[];
}

export interface DocumentListResponse {
  total: number;
  items: DocumentItem[];
}

export interface OverviewKPIs {
  total_documents: number;
  processed_count: number;
  automatic_count: number;
  completed_manual_count: number;
  manual_review_count: number;
  error_count: number;
  automation_rate: number;
  manual_review_rate: number;
  failure_rate: number;
  corrections_total: number;
  average_confidence: number;
  average_processing_time_ms: number;
  average_review_time_ms: number;
  queue_pending_count: number;
}

export interface BreakdownItem {
  category: string;
  count: number;
  percentage: number;
}

export type TimePeriod = "today" | "7d" | "30d" | "90d" | "all" | "custom";

export interface AnalyticsFilterParams {
  period?: TimePeriod | string;
  from?: string;
  to?: string;
}

export interface ReviewReasonMetric {
  reason: string;
  count: number;
  share: number;
}

export interface ReviewReasonsResponse {
  total_reviews: number;
  reasons: ReviewReasonMetric[];
}

export interface FieldCorrectionMetric {
  field: string;
  field_name?: string;
  total_occurrences: number;
  corrected_count: number;
  correction_rate: number;
}

export interface FieldCorrectionsResponse {
  total_fields: number;
  items: FieldCorrectionMetric[];
}

export interface ConfidenceBucket {
  bucket: string;
  count: number;
  percentage: number;
  manual_review_count: number;
  correction_count: number;
  review_rate?: number;
}

export interface ConfidenceAnalyticsResponse {
  total_documents: number;
  average_confidence: number;
  distribution: ConfidenceBucket[];
}

export interface DocumentTypeAnalytics {
  document_type: string;
  total_documents: number;
  processed_documents: number;
  automatic_count: number;
  manual_review_count: number;
  error_count: number;
  automation_rate: number;
  manual_review_rate: number;
  failure_rate: number;
  average_confidence: number;
  average_processing_time_ms: number;
  average_review_time_ms: number;
}

export interface DocumentTypesResponse {
  items: DocumentTypeAnalytics[];
}

export interface AutomationTrendItem {
  date: string;
  total_intake: number;
  processed_count: number;
  automatic_count: number;
  manual_review_count: number;
  error_count: number;
  automation_rate: number;
}

export interface AutomationTrendsResponse {
  items: AutomationTrendItem[];
}

export interface QueueAnalyticsResponse {
  manual_review: number;
  processing: number;
  errors: number;
}

export interface AnalyticsOverview {
  kpis: OverviewKPIs;
  review_reasons: BreakdownItem[];
  document_type_distribution: BreakdownItem[];
  status_distribution: BreakdownItem[];
}

export interface ResearchSummary {
  b0_clean_baseline: {
    cer_normalized: number;
    char_ned_normalized: number;
    macro_f1_normalized: number;
    total_evaluated: number;
  };
  b1_degradations: Array<{
    degradation: string;
    severity: number;
    cer_mean: number;
    ned_mean: number;
    macro_f1_mean: number;
  }>;
  b2_top_recoveries: Array<{
    condition_id: string;
    degradation: string;
    severity: number;
    preprocessing: string;
    delta_cer_recovery: number;
    delta_f1_recovery: number;
  }>;
}

export interface ResearchTrackSummary {
  id: string;
  name: string;
  description: string;
  experiment_count: number;
  key_metrics: Record<string, any>;
}

export interface ResearchOverviewResponse {
  tracks: ResearchTrackSummary[];
  total_experiments: number;
  canonical_dataset: string;
  execution_provider: string;
}

export interface ResearchExperimentItem {
  id: string;
  track: string;
  name: string;
  description: string;
  dataset: string;
  status: string;
  metrics_summary: Record<string, any>;
  source: string;
}

export interface ResearchExperimentsResponse {
  items: ResearchExperimentItem[];
  total: number;
}

export interface ResearchConditionMetric {
  condition_id: string;
  name: string;
  severity?: number;
  cer?: number;
  wer?: number;
  ned?: number;
  f1?: number;
  delta_cer?: number;
  delta_f1?: number;
  preprocessing?: string;
}

export interface ResearchExperimentDetail {
  id: string;
  name: string;
  track: string;
  description: string;
  status: string;
  dataset: string;
  sample_size: number;
  source: string;
  metrics_summary: Record<string, any>;
  conditions: ResearchConditionMetric[];
  ocr_metrics?: Record<string, any>;
  kie_metrics?: Record<string, any>;
  findings: string[];
  metadata: Record<string, any>;
}

export interface DegradationCondition {
  condition_id: string;
  severity: string;
  parameters?: Record<string, any> | null;
  parameter?: string | null;
  cer: number;
  wer: number;
  ned: number;
  macro_f1: number;
  entity_hmean: number;
  delta_cer: number;
  delta_wer?: number | null;
  delta_ned: number;
  delta_macro_f1: number;
  delta_entity_hmean: number;
}

export interface DegradationExplorerItem {
  id: string;
  code: string;
  name: string;
  description: string;
  dataset: string;
  sample_size: number;
  conditions: DegradationCondition[];
  findings: string[];
  source: string;
  calibration_source?: string | null;
}

export interface DegradationExplorerResponse {
  items: DegradationExplorerItem[];
  total: number;
  control_condition: DegradationCondition;
  canonical_dataset: string;
}

export interface PreprocessingMetrics {
  cer: number;
  wer?: number | null;
  ned: number;
  macro_f1: number;
  entity_hmean: number;
  doc_em?: number | null;
}

export interface PreprocessingPolicyMetric {
  condition_id: string;
  policy_id: string;
  policy_name: string;
  policy_key: string;
  metrics: PreprocessingMetrics;
  delta_cer_vs_degraded: number;
  delta_wer_vs_degraded?: number | null;
  delta_ned_vs_degraded: number;
  delta_macro_f1_vs_degraded: number;
  delta_entity_hmean_vs_degraded: number;
  delta_cer_vs_clean: number;
  delta_macro_f1_vs_clean: number;
  relative_recovery_rate?: number | null;
}

export interface PreprocessingConditionGroup {
  degradation_code: string;
  degradation_name: string;
  degradation_key: string;
  severity: string;
  ref_b1_condition_id: string;
  clean_control_metrics: PreprocessingMetrics;
  degraded_metrics: PreprocessingMetrics;
  policies: PreprocessingPolicyMetric[];
  findings: string[];
}

export interface PreprocessingExplorerResponse {
  groups: PreprocessingConditionGroup[];
  total_groups: number;
  degradations: Array<{ code: string; id: string; name: string; key: string }>;
  severities: string[];
  policies: Array<{ id: string; name: string; key: string }>;
  canonical_dataset: string;
  total_conditions_evaluated: number;
  source: string;
}

export interface ResearchDegradationRef {
  code: string;
  name: string;
}

export interface TrackEvidenceRef {
  experiment_id?: string | null;
  available: boolean;
  details?: string | null;
}

export interface ResearchEvidenceReference {
  production_signal: string;
  observed_value?: number | null;
  unit?: string | null;
  research_degradation_code: string;
  research_degradation_name: string;
  b1_experiment_id?: string | null;
  b1_available: boolean;
  b2_degradation_code?: string | null;
  b2_available: boolean;
  b2_evaluated_policies_count: number;
  research_degradation?: ResearchDegradationRef;
  b1?: TrackEvidenceRef;
  b2?: TrackEvidenceRef;
}

export interface DecisionReason {
  code: string;
  category: "confidence" | "validation" | "quality" | "pipeline" | "none" | string;
  message: string;
  blocking: boolean;
  field?: string | null;
  observed_value?: number | string | null;
  threshold?: number | string | null;
}

export interface DecisionExplanation {
  decision: "automatic" | "manual_review" | "rejected" | "pending" | string;
  primary_reason?: string | null;
  reasons: DecisionReason[];
  positive_criteria?: string[];
  automatic_eligible: boolean;
  summary: string;
}

export interface DecisionReasonItem {
  code: string;
  category: string;
  count: number;
  percentage: number;
  description: string;
}

export interface DecisionReasonCategoryMetric {
  category: string;
  count: number;
  percentage: number;
}

export interface DecisionReasonFieldMetric {
  field: string;
  count: number;
  percentage: number;
}

export interface DecisionReasonsAnalyticsResponse {
  total_decisions: number;
  automatic_count: number;
  manual_review_count: number;
  automation_rate: number;
  manual_review_rate: number;
  by_reason: DecisionReasonItem[];
  by_category: DecisionReasonCategoryMetric[];
  by_field: DecisionReasonFieldMetric[];
}

export interface DocumentFieldCorrection {
  field: string;
  previous_value?: string | null;
  final_value?: string | null;
  corrected_at: string;
  actor: string;
}

export interface CorrectionSummary {
  total_correction_events: number;
  documents_with_corrections: number;
  completed_manual_review_documents: number;
  correction_rate: number;
  no_change_review_rate: number;
  avg_corrections_per_corrected_document: number;
}

export interface FeedbackFunnel {
  processed: number;
  automatic: number;
  manual_review: number;
  completed_manual_review: number;
  with_corrections: number;
  without_corrections: number;
  automatic_rate: number;
  manual_review_rate: number;
  correction_rate: number;
  no_change_review_rate: number;
}

export interface CorrectionFieldMetric {
  field: string;
  correction_count: number;
  affected_documents: number;
  share: number;
}

export interface CorrectionDocumentTypeMetric {
  document_type: string;
  reviewed_documents: number;
  documents_with_corrections: number;
  correction_events: number;
  correction_rate: number;
}

export interface CorrectionReasonMetric {
  reason: string;
  reviewed_documents: number;
  documents_with_corrections: number;
  correction_events: number;
  correction_rate: number;
}

export interface CorrectionReasonFieldMetric {
  review_reason: string;
  field: string;
  correction_count: number;
  affected_documents: number;
}

export interface ProductionFeedbackResponse {
  summary: CorrectionSummary;
  funnel: FeedbackFunnel;
  by_field: CorrectionFieldMetric[];
  by_document_type: CorrectionDocumentTypeMetric[];
  by_reason: CorrectionReasonMetric[];
  by_reason_field: CorrectionReasonFieldMetric[];
}

export interface ControlCenterPeriod {
  period: string;
  date_from?: string | null;
  date_to?: string | null;
}

export interface ControlCenterSummary {
  total_documents: number;
  processed_count: number;
  automatic_count: number;
  manual_review_count: number;
  completed_manual_count: number;
  error_count: number;
  automation_rate: number;
  manual_review_rate: number;
  correction_rate: number;
  no_change_review_rate: number;
  average_confidence: number;
  average_processing_time_ms: number;
  average_review_time_ms: number;
}

export interface ControlCenterQuality {
  average_quality_score?: number | null;
  average_document_confidence?: number | null;
  average_ocr_confidence?: number | null;
  average_kie_confidence?: number | null;
  validation_pass_rate?: number | null;
  total_documents_validated: number;
  passed_validation_count: number;
  failed_validation_count: number;
}

export interface ControlCenterResponse {
  period: ControlCenterPeriod;
  summary: ControlCenterSummary;
  queue: QueueAnalyticsResponse;
  funnel: FeedbackFunnel;
  quality: ControlCenterQuality;
  review_reasons: DecisionReasonItem[];
  feedback: ProductionFeedbackResponse;
  trends: AutomationTrendItem[];
  recent_documents: DocumentItem[];
}

// ==========================================
// MVP-9: Operational Action & Resolution Center
// ==========================================

export type IssueCategory =
  | "quality"
  | "confidence"
  | "validation"
  | "manual_review"
  | "processing"
  | "correction";

export type IssueSeverity = "critical" | "high" | "medium" | "low" | "info";

export type ResolutionStatus = "open" | "in_review" | "resolved";

export type ActionType =
  | "open_document"
  | "open_review"
  | "open_field"
  | "open_quality_evidence"
  | "open_validation"
  | "open_research_evidence"
  | "retry_processing";

export interface OperationalAction {
  id: string;
  type: ActionType;
  label: string;
  description: string;
  enabled: boolean;
  target: string;
  params: Record<string, any>;
}

export interface IssueEvidence {
  production?: string | null;
  historical?: string | null;
  research?: string | null;
}

export interface OperationalIssue {
  id: string;
  document_id: string;
  category: IssueCategory;
  severity: IssueSeverity;
  code: string;
  title: string;
  description: string;
  observed_value?: number | string | null;
  threshold?: number | string | null;
  field?: string | null;
  source: string;
  status: ResolutionStatus;
  evidence: IssueEvidence;
  available_actions: OperationalAction[];
  created_at?: string | null;
}

export interface DocumentOperationsResponse {
  document_id: string;
  overall_status: ResolutionStatus;
  total_issues: number;
  blocking_issues: number;
  issues: OperationalIssue[];
  summary: string;
}

export interface OperationsBacklogSummary {
  total_open: number;
  total_in_review: number;
  by_category: Record<string, number>;
  by_reason: Record<string, number>;
}

// MVP-10: Document AI Improvement Loop
export type ResearchQuestionStatus = "OPEN" | "IN_PROGRESS" | "EXPERIMENT_AVAILABLE" | "CLOSED";

export type ResearchSignalType =
  | "CORRECTION_CONCENTRATION"
  | "CONFIDENCE_CORRECTION_PATTERN"
  | "REVIEW_REASON_CORRECTION_PATTERN"
  | "QUALITY_CORRECTION_PATTERN"
  | "FIELD_VALIDATION_PATTERN";

export interface FieldCorrectionMetric {
  field: string;
  correction_count: number;
  evaluated_count: number;
  correction_rate: number;
  share_of_all_corrections: number;
  affected_documents: number;
}

export interface ConfidenceCorrectionBucket {
  bucket: string;
  evaluated_fields: number;
  corrected_fields: number;
  correction_rate: number;
  share_of_corrections: number;
}

export interface ResearchSignal {
  id: string;
  signal_type: ResearchSignalType;
  title: string;
  description: string;
  population: Record<string, any>;
  field?: string | null;
  document_type?: string | null;
  evidence: Record<string, any>;
  period: string;
  related_evidence_refs: Array<Record<string, any>>;
}

export interface ResearchQuestion {
  id: string;
  title: string;
  description: string;
  source_signal_id?: string | null;
  field?: string | null;
  document_type?: string | null;
  status: ResearchQuestionStatus;
  related_evidence_refs: Array<Record<string, any>>;
  created_by: string;
  created_at: string;
  updated_at: string;
}

export interface ResearchQuestionListResponse {
  items: ResearchQuestion[];
  total: number;
  limit: number;
  offset: number;
}

export interface ResearchQuestionDetail extends ResearchQuestion {
  enriched_evidence?: Record<string, any> | null;
}

export interface ImprovementOverviewSummary {
  documents_processed: number;
  documents_reviewed: number;
  documents_corrected: number;
  total_field_corrections: number;
  correction_rate: number;
  no_change_review_rate: number;
}

export interface ImprovementOverview {
  period: string;
  date_from?: string | null;
  date_to?: string | null;
  summary: ImprovementOverviewSummary;
  by_field: FieldCorrectionMetric[];
  by_confidence: ConfidenceCorrectionBucket[];
  by_reason: Array<{
    reason: string;
    reviewed_documents: number;
    documents_with_corrections: number;
    correction_events: number;
    correction_rate: number;
  }>;
  by_document_type: Array<{
    document_type: string;
    reviewed_documents: number;
    documents_with_corrections: number;
    correction_events: number;
    correction_rate: number;
  }>;
  active_signals_count: number;
  open_questions_count: number;
}

export interface CreateResearchQuestionRequest {
  title: string;
  description: string;
  source_signal_id?: string | null;
  field?: string | null;
  document_type?: string | null;
  related_evidence_refs?: Array<Record<string, any>> | null;
  created_by?: string;
}

export interface UpdateResearchQuestionRequest {
  title?: string;
  description?: string;
  status?: ResearchQuestionStatus;
  related_evidence_refs?: Array<Record<string, any>> | null;
}



