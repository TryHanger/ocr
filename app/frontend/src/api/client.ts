import axios from "axios";
import {
  AnalyticsFilterParams,
  AnalyticsOverview,
  AutomationTrendsResponse,
  ConfidenceAnalyticsResponse,
  ControlCenterResponse,
  DecisionReasonsAnalyticsResponse,
  DegradationExplorerResponse,
  DocumentDetail,
  DocumentItem,
  DocumentListResponse,
  DocumentTypesResponse,
  FieldCorrectionsResponse,
  NormalizedField,
  PreprocessingExplorerResponse,
  ProductionFeedbackResponse,
  QueueAnalyticsResponse,
  ResearchExperimentDetail,
  ResearchExperimentsResponse,
  ResearchOverviewResponse,
  ResearchSummary,
  ReviewReasonsResponse,
  DocumentOperationsResponse,
  OperationalAction,
  OperationsBacklogSummary,
  CreateResearchQuestionRequest,
  ImprovementOverview,
  ResearchQuestion,
  ResearchQuestionDetail,
  ResearchQuestionListResponse,
  ResearchSignal,
  UpdateResearchQuestionRequest,
} from "../types";

const api = axios.create({
  baseURL: "/api/v1",
  timeout: 30000,
});

export const apiDocuments = {
  list: async (params?: {
    status?: string;
    decision?: string;
    document_type?: string;
    search?: string;
    limit?: number;
    offset?: number;
  }): Promise<DocumentListResponse> => {
    const res = await api.get<DocumentListResponse>("/documents", { params });
    return res.data;
  },

  get: async (id: string): Promise<DocumentDetail> => {
    const res = await api.get<DocumentDetail>(`/documents/${id}`);
    return res.data;
  },

  upload: async (file: File, documentType: string = "receipt"): Promise<DocumentItem> => {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("document_type", documentType);
    const res = await api.post<DocumentItem>("/documents", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return res.data;
  },

  retry: async (id: string): Promise<{ message: string; job_id: string }> => {
    const res = await api.post<{ message: string; job_id: string }>(`/documents/${id}/retry`);
    return res.data;
  },
};

export const apiReview = {
  correctField: async (
    documentId: string,
    fieldName: string,
    correctedValue: string,
    reviewerName: string = "operator",
    notes?: string
  ): Promise<NormalizedField> => {
    const res = await api.patch<NormalizedField>(`/documents/${documentId}/fields/${fieldName}`, {
      field_name: fieldName,
      corrected_value: correctedValue,
      reviewer_name: reviewerName,
      notes,
    });
    return res.data;
  },

  approve: async (
    documentId: string,
    reviewerName: string = "operator",
    notes?: string
  ): Promise<{ status: string; decision: string; message: string }> => {
    const res = await api.post(`/documents/${documentId}/approve`, {
      reviewer_name: reviewerName,
      notes,
    });
    return res.data;
  },

  completeReview: async (
    documentId: string,
    reviewerName: string = "operator",
    notes?: string
  ): Promise<{ status: string; decision: string; message: string }> => {
    const res = await api.post(`/documents/${documentId}/review/complete`, {
      reviewer_name: reviewerName,
      notes,
    });
    return res.data;
  },

  reject: async (
    documentId: string,
    reason: string,
    reviewerName: string = "operator",
    notes?: string
  ): Promise<{ status: string; decision: string; message: string }> => {
    const res = await api.post(`/documents/${documentId}/reject`, {
      reason,
      reviewer_name: reviewerName,
      notes,
    });
    return res.data;
  },
};

export const apiAnalytics = {
  getOverview: async (params?: AnalyticsFilterParams): Promise<AnalyticsOverview> => {
    const res = await api.get<AnalyticsOverview>("/analytics/overview", { params });
    return res.data;
  },

  getReviewReasons: async (params?: AnalyticsFilterParams): Promise<ReviewReasonsResponse> => {
    const res = await api.get<ReviewReasonsResponse>("/analytics/review-reasons", { params });
    return res.data;
  },

  getCorrections: async (params?: AnalyticsFilterParams): Promise<FieldCorrectionsResponse> => {
    const res = await api.get<FieldCorrectionsResponse>("/analytics/corrections", { params });
    return res.data;
  },

  getConfidence: async (params?: AnalyticsFilterParams): Promise<ConfidenceAnalyticsResponse> => {
    const res = await api.get<ConfidenceAnalyticsResponse>("/analytics/confidence", { params });
    return res.data;
  },

  getDocumentTypes: async (params?: AnalyticsFilterParams): Promise<DocumentTypesResponse> => {
    const res = await api.get<DocumentTypesResponse>("/analytics/document-types", { params });
    return res.data;
  },

  getTrends: async (params?: AnalyticsFilterParams): Promise<AutomationTrendsResponse> => {
    const res = await api.get<AutomationTrendsResponse>("/analytics/trends", { params });
    return res.data;
  },

  getQueue: async (): Promise<QueueAnalyticsResponse> => {
    const res = await api.get<QueueAnalyticsResponse>("/analytics/queue");
    return res.data;
  },

  getDecisionReasons: async (params?: AnalyticsFilterParams): Promise<DecisionReasonsAnalyticsResponse> => {
    const res = await api.get<DecisionReasonsAnalyticsResponse>("/analytics/decision-reasons", { params });
    return res.data;
  },

  getFeedback: async (params?: AnalyticsFilterParams): Promise<ProductionFeedbackResponse> => {
    const res = await api.get<ProductionFeedbackResponse>("/analytics/feedback", { params });
    return res.data;
  },

  getControlCenter: async (params?: AnalyticsFilterParams): Promise<ControlCenterResponse> => {
    const res = await api.get<ControlCenterResponse>("/analytics/control-center", { params });
    return res.data;
  },
};

export const apiResearch = {
  getSummary: async (): Promise<ResearchSummary> => {
    const res = await api.get<ResearchSummary>("/research/summary");
    return res.data;
  },

  getOverview: async (): Promise<ResearchOverviewResponse> => {
    const res = await api.get<ResearchOverviewResponse>("/research/overview");
    return res.data;
  },

  getExperiments: async (track?: string): Promise<ResearchExperimentsResponse> => {
    const params = track ? { track } : undefined;
    const res = await api.get<ResearchExperimentsResponse>("/research/experiments", { params });
    return res.data;
  },

  getExperiment: async (id: string): Promise<ResearchExperimentDetail> => {
    const res = await api.get<ResearchExperimentDetail>(`/research/experiments/${id}`);
    return res.data;
  },

  getDegradations: async (degradation?: string): Promise<DegradationExplorerResponse> => {
    const params = degradation ? { degradation } : undefined;
    const res = await api.get<DegradationExplorerResponse>("/research/degradations", { params });
    return res.data;
  },

  getPreprocessing: async (params?: {
    degradation?: string;
    severity?: string;
    policy?: string;
  }): Promise<PreprocessingExplorerResponse> => {
    const res = await api.get<PreprocessingExplorerResponse>("/research/preprocessing", { params });
    return res.data;
  },
};

export const apiOperations = {
  getDocumentIssues: async (documentId: string): Promise<DocumentOperationsResponse> => {
    const res = await api.get<DocumentOperationsResponse>(`/operations/documents/${documentId}/issues`);
    return res.data;
  },

  getDocumentActions: async (documentId: string): Promise<OperationalAction[]> => {
    const res = await api.get<OperationalAction[]>(`/operations/documents/${documentId}/actions`);
    return res.data;
  },

  getBacklog: async (): Promise<OperationsBacklogSummary> => {
    const res = await api.get<OperationsBacklogSummary>("/operations/backlog");
    return res.data;
  },
};

export const apiImprovement = {
  getOverview: async (params?: { period?: string; date_from?: string; date_to?: string }): Promise<ImprovementOverview> => {
    const res = await api.get<ImprovementOverview>("/improvement/overview", { params });
    return res.data;
  },

  getSignals: async (params?: {
    period?: string;
    date_from?: string;
    date_to?: string;
    field?: string;
  }): Promise<ResearchSignal[]> => {
    const res = await api.get<ResearchSignal[]>("/improvement/signals", { params });
    return res.data;
  },

  listQuestions: async (params?: {
    status?: string;
    field?: string;
    limit?: number;
    offset?: number;
  }): Promise<ResearchQuestionListResponse> => {
    const res = await api.get<ResearchQuestionListResponse>("/improvement/questions", { params });
    return res.data;
  },

  createQuestion: async (data: CreateResearchQuestionRequest): Promise<ResearchQuestion> => {
    const res = await api.post<ResearchQuestion>("/improvement/questions", data);
    return res.data;
  },

  getQuestionDetail: async (id: string): Promise<ResearchQuestionDetail> => {
    const res = await api.get<ResearchQuestionDetail>(`/improvement/questions/${id}`);
    return res.data;
  },

  updateQuestion: async (id: string, data: UpdateResearchQuestionRequest): Promise<ResearchQuestion> => {
    const res = await api.patch<ResearchQuestion>(`/improvement/questions/${id}`, data);
    return res.data;
  },
};

export const apiSystem = {
  getVersion: async (): Promise<{ version: string; app_name: string; environment: string }> => {
    const res = await axios.get("/api/version");
    return res.data;
  },
  getHealth: async (): Promise<{ status: string }> => {
    const res = await axios.get("/health");
    return res.data;
  },
};
