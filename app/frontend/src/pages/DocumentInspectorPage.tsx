import React, { useEffect, useState } from "react";
import { DocumentDetail, DocumentOperationsResponse, OperationalAction } from "../types";
import { apiDocuments, apiOperations, apiReview } from "../api/client";
import { DocumentViewer } from "../components/DocumentViewer";
import { StatusBadge } from "../components/StatusBadge";
import { ConfidenceBadge } from "../components/ConfidenceBadge";
import { OperationalActionCenter } from "../components/OperationalActionCenter";
import {
  ArrowLeft,
  CheckCircle,
  XCircle,
  RotateCw,
  Edit2,
  Check,
  CheckCircle2,
  Activity,
  ShieldCheck,
  FileText,
  History,
  FlaskConical,
  ExternalLink,
  Info,
} from "lucide-react";

interface DocumentInspectorPageProps {
  documentId: string;
  onBack: () => void;
  onNavigateToResearch?: (track: "B1" | "B2", degradationCode: string) => void;
}

export const DocumentInspectorPage: React.FC<DocumentInspectorPageProps> = ({
  documentId,
  onBack,
  onNavigateToResearch,
}) => {
  const [doc, setDoc] = useState<DocumentDetail | null>(null);
  const [selectedField, setSelectedField] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"fields" | "quality" | "ocr" | "audit">("fields");
  const [editingField, setEditingField] = useState<string | null>(null);
  const [editValue, setEditValue] = useState("");
  const [isLoading, setIsLoading] = useState(true);

  // MVP-9 Operational Action & Resolution State
  const [operations, setOperations] = useState<DocumentOperationsResponse | null>(null);
  const [isLoadingOps, setIsLoadingOps] = useState(false);
  const [opsError, setOpsError] = useState<string | null>(null);

  const loadOperations = async () => {
    setIsLoadingOps(true);
    setOpsError(null);
    try {
      const opsData = await apiOperations.getDocumentIssues(documentId);
      setOperations(opsData);
    } catch (err: any) {
      console.error("Failed to load operational issues", err);
      setOpsError(err?.message || "Failed to load operational guidance");
    } finally {
      setIsLoadingOps(false);
    }
  };

  const loadDocument = async () => {
    try {
      const data = await apiDocuments.get(documentId);
      setDoc(data);
    } catch (err) {
      console.error("Failed to load document details", err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadDocument();
    loadOperations();
    // Poll if document is still in processing state
    if (doc?.status === "processing" || doc?.status === "uploaded") {
      const interval = setInterval(() => {
        loadDocument();
        loadOperations();
      }, 1500);
      return () => clearInterval(interval);
    }
  }, [documentId, doc?.status]);

  const handleOperationalAction = (action: OperationalAction) => {
    switch (action.type) {
      case "open_field":
        if (action.params?.field_name) {
          setSelectedField(action.params.field_name);
          setActiveTab("fields");
        }
        break;
      case "open_quality_evidence":
        setActiveTab("quality");
        break;
      case "open_validation":
        setActiveTab("fields");
        break;
      case "open_research_evidence":
        if (action.params?.condition_code && onNavigateToResearch) {
          const track = (action.params.track as "B1" | "B2") || "B2";
          onNavigateToResearch(track, action.params.condition_code);
        }
        break;
      case "open_review":
        const reviewBtn = document.getElementById("complete-review-btn");
        if (reviewBtn) {
          reviewBtn.scrollIntoView({ behavior: "smooth", block: "center" });
          reviewBtn.focus();
        }
        break;
      case "retry_processing":
        handleRetry();
        break;
      case "open_document":
        setSelectedField(null);
        break;
    }
  };

  const handleSaveField = async (fieldName: string) => {
    try {
      await apiReview.correctField(documentId, fieldName, editValue, "operator");
      setEditingField(null);
      await loadDocument();
      await loadOperations();
    } catch (err) {
      console.error("Failed to correct field", err);
    }
  };

  const [validationErrorBanner, setValidationErrorBanner] = useState<string | null>(null);
  const [isCompleting, setIsCompleting] = useState(false);

  const handleCompleteReview = async () => {
    setIsCompleting(true);
    setValidationErrorBanner(null);
    try {
      await apiReview.completeReview(documentId, "operator");
      await loadDocument();
      await loadOperations();
    } catch (err: any) {
      const detail = err.response?.data?.detail || "Validation check failed before review completion";
      setValidationErrorBanner(detail);
      await loadDocument();
      await loadOperations();
    } finally {
      setIsCompleting(false);
    }
  };

  const handleReject = async () => {
    const reason = prompt("Enter rejection reason:", "Document unreadable / invalid");
    if (reason) {
      try {
        await apiReview.reject(documentId, reason, "operator");
        await loadDocument();
        await loadOperations();
      } catch (err) {
        console.error("Failed to reject document", err);
      }
    }
  };

  const handleRetry = async () => {
    try {
      await apiDocuments.retry(documentId);
      await loadDocument();
      await loadOperations();
    } catch (err) {
      console.error("Failed to retry job", err);
    }
  };

  if (isLoading || !doc) {
    return (
      <div className="flex items-center justify-center p-24 text-slate-400">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-500"></div>
      </div>
    );
  }

  const primaryPage = doc.pages[0];

  return (
    <div className="space-y-4 max-w-7xl mx-auto h-full flex flex-col">
      {/* Top Header Bar */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 bg-slate-900/80 p-4 rounded-2xl border border-slate-800 shrink-0">
        <div className="flex items-center gap-3">
          <button
            onClick={onBack}
            className="p-2 text-slate-400 hover:text-slate-100 hover:bg-slate-800 rounded-xl transition-colors"
            title="Back to queue"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-base font-semibold text-slate-100 truncate max-w-md">
                {doc.filename}
              </h2>
              <StatusBadge status={doc.status} size="sm" />
              <ConfidenceBadge confidence={doc.confidence} size="sm" />
            </div>
            <p className="text-xs text-slate-400 font-mono mt-0.5">
              ID: {doc.id} · Type: {doc.document_type} · Processing: {doc.processing_duration_ms || 0} ms
              {doc.review_duration_ms ? ` · Review: ${(doc.review_duration_ms / 1000).toFixed(1)}s` : ""}
            </p>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-2">
          {doc.status === "manual_review" && (
            <>
              <button
                id="complete-review-btn"
                onClick={handleCompleteReview}
                disabled={isCompleting}
                className="flex items-center gap-1.5 px-3.5 py-1.5 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white rounded-xl text-xs font-semibold shadow-md shadow-emerald-600/20 transition-all active:scale-95"
              >
                <CheckCircle className="w-4 h-4" />
                {isCompleting ? "Validating..." : "Complete Review"}
              </button>
              <button
                onClick={handleReject}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-rose-600/20 text-rose-300 hover:bg-rose-600/30 border border-rose-500/30 rounded-xl text-xs font-semibold transition-all active:scale-95"
              >
                <XCircle className="w-4 h-4" />
                Reject
              </button>
            </>
          )}

          <button
            onClick={handleRetry}
            className="p-2 text-slate-400 hover:text-indigo-400 hover:bg-slate-800 rounded-xl transition-colors"
            title="Rerun pipeline"
          >
            <RotateCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Validation Error Banner */}
      {validationErrorBanner && (
        <div className="p-3 bg-rose-950/50 border border-rose-500/50 rounded-xl text-xs text-rose-200">
          <span className="font-semibold text-rose-300">Validation Blocked: </span>
          {validationErrorBanner}
        </div>
      )}

      {/* Automation Decision & Explainability Card */}
      {(() => {
        const exp = doc.decision_explanation;
        const decision = exp?.decision || doc.decision || "pending";
        const isAuto = decision === "automatic";
        const isManual = decision === "manual_review";
        const primaryReason = exp?.primary_reason || doc.review_reason;

        const getCategoryBadgeClass = (category: string) => {
          switch (category) {
            case "confidence":
              return "bg-indigo-950/80 text-indigo-300 border-indigo-800/60";
            case "validation":
              return "bg-amber-950/80 text-amber-300 border-amber-800/60";
            case "quality":
              return "bg-orange-950/80 text-orange-300 border-orange-800/60";
            case "pipeline":
              return "bg-rose-950/80 text-rose-300 border-rose-800/60";
            default:
              return "bg-slate-800 text-slate-300 border-slate-700";
          }
        };

        return (
          <div
            className={`p-4 rounded-2xl border transition-all text-xs ${
              isAuto
                ? "bg-slate-900/90 border-emerald-500/30 shadow-lg shadow-emerald-950/10"
                : isManual
                ? "bg-slate-900/90 border-amber-500/30 shadow-lg shadow-amber-950/10"
                : "bg-slate-900/90 border-slate-800"
            }`}
          >
            {/* Header row: Decision badge, Primary reason, latencies */}
            <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-800/80">
              <div className="flex items-center gap-2.5 flex-wrap">
                <span className="font-semibold text-slate-200 tracking-wide uppercase text-[11px]">
                  Automation Decision & Explainability
                </span>

                <span
                  className={`px-2.5 py-0.5 rounded-full font-bold uppercase tracking-wider text-[11px] border ${
                    isAuto
                      ? "bg-emerald-950 text-emerald-400 border-emerald-700/60"
                      : isManual
                      ? "bg-amber-950 text-amber-400 border-amber-700/60"
                      : "bg-slate-800 text-slate-300 border-slate-700"
                  }`}
                >
                  {decision.replace(/_/g, " ")}
                </span>

                {primaryReason && primaryReason !== "none" && (
                  <span className="px-2.5 py-0.5 rounded-full font-medium text-[11px] bg-amber-950/60 text-amber-300 border border-amber-800/40">
                    Primary: {primaryReason.replace(/_/g, " ")}
                  </span>
                )}

                {doc.status === "completed_manual" && (
                  <span className="px-2 py-0.5 rounded-full text-[10px] bg-slate-800 text-slate-400 border border-slate-700">
                    Historical Decision (Approved by Operator)
                  </span>
                )}
              </div>

              <div className="flex items-center gap-3 text-slate-400 text-[11px] font-mono">
                {doc.processing_duration_ms && <span>Proc: {doc.processing_duration_ms}ms</span>}
                {doc.review_duration_ms ? <span>Review: {doc.review_duration_ms}ms</span> : null}
                {doc.review_wait_duration_ms ? <span>Wait: {doc.review_wait_duration_ms}ms</span> : null}
              </div>
            </div>

            {/* Explanation Summary */}
            <div className="mt-3">
              <p className="text-slate-300 text-xs leading-relaxed">
                {exp?.summary ||
                  (isAuto
                    ? "Document met all criteria for automatic straight-through processing."
                    : isManual
                    ? "Document required manual review based on operational policy."
                    : "Decision evaluation pending.")}
              </p>
            </div>

            {/* If Automatic: Positive criteria checklist */}
            {isAuto && exp?.positive_criteria && exp.positive_criteria.length > 0 && (
              <div className="mt-3 grid grid-cols-1 sm:grid-cols-3 gap-2">
                {exp.positive_criteria.map((crit, idx) => (
                  <div
                    key={idx}
                    className="flex items-center gap-2 px-3 py-2 rounded-xl bg-emerald-950/30 border border-emerald-800/40 text-xs text-emerald-200"
                  >
                    <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                    <span className="leading-tight">{crit}</span>
                  </div>
                ))}
              </div>
            )}

            {/* If Manual Review: Detailed Blocking Conditions / Reasons */}
            {isManual && exp?.reasons && exp.reasons.length > 0 && (
              <div className="mt-3 space-y-2">
                <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                  Blocking Conditions ({exp.reasons.length})
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                  {exp.reasons.map((reason, idx) => (
                    <div
                      key={idx}
                      className="p-3 rounded-xl bg-slate-950/80 border border-slate-800 flex flex-col justify-between gap-2"
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex items-center gap-1.5 flex-wrap">
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wider border ${getCategoryBadgeClass(
                              reason.category
                            )}`}
                          >
                            {reason.category}
                          </span>
                          <span className="font-mono text-slate-300 font-semibold text-[11px]">
                            {reason.code}
                          </span>
                          {reason.field && (
                            <span className="px-1.5 py-0.5 rounded bg-slate-800 text-indigo-300 font-mono text-[10px]">
                              field: {reason.field}
                            </span>
                          )}
                        </div>
                        {reason.blocking && (
                          <span className="px-1.5 py-0.5 rounded bg-rose-950/80 text-rose-300 text-[10px] border border-rose-800/50 font-medium">
                            blocking
                          </span>
                        )}
                      </div>

                      <p className="text-slate-300 text-xs leading-normal">{reason.message}</p>

                      {((reason.observed_value !== null && reason.observed_value !== undefined) ||
                        (reason.threshold !== null && reason.threshold !== undefined)) && (
                        <div className="flex items-center gap-2 pt-1 border-t border-slate-800/80 text-[11px] font-mono text-slate-400">
                          {reason.observed_value !== null && reason.observed_value !== undefined && (
                            <span>
                              Observed: <strong className="text-white">{reason.observed_value}</strong>
                            </span>
                          )}
                          {reason.threshold !== null && reason.threshold !== undefined && (
                            <>
                              <span className="text-slate-600">|</span>
                              <span>
                                Threshold: <strong className="text-amber-300">{reason.threshold}</strong>
                              </span>
                            </>
                          )}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        );
      })()}

      {/* Production Feedback Card (Only for documents routed to / through manual review) */}
      {(doc.status === "manual_review" || doc.status === "completed_manual" || doc.decision === "manual_review") && (
        <div className="p-4 rounded-2xl bg-slate-900/90 border border-slate-800 text-xs shadow-lg space-y-3">
          <div className="flex items-center justify-between pb-2 border-b border-slate-800/80">
            <div className="flex items-center gap-2">
              <History className="w-4 h-4 text-emerald-400" />
              <span className="font-semibold text-slate-200 tracking-wide uppercase text-[11px]">
                Production Feedback (Human Corrections)
              </span>
            </div>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
              {doc.field_corrections && doc.field_corrections.length > 0
                ? `${doc.field_corrections.length} correction event(s)`
                : "No corrections"}
            </span>
          </div>

          {doc.field_corrections && doc.field_corrections.length > 0 ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 pt-1">
              {doc.field_corrections.map((corr, idx) => (
                <div
                  key={idx}
                  className="p-3 rounded-xl bg-slate-950/80 border border-slate-800 flex flex-col justify-between gap-1.5 font-mono"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-emerald-400 text-xs">{corr.field}</span>
                    <span className="text-[10px] text-slate-500">{corr.actor}</span>
                  </div>
                  <div className="text-[11px] space-y-0.5 text-slate-300">
                    <div className="flex items-baseline gap-2">
                      <span className="text-slate-500 text-[10px] uppercase w-12 shrink-0">Before:</span>
                      <span className="text-rose-300/80 line-through truncate">
                        {corr.previous_value !== null && corr.previous_value !== undefined
                          ? corr.previous_value
                          : "(empty)"}
                      </span>
                    </div>
                    <div className="flex items-baseline gap-2">
                      <span className="text-slate-500 text-[10px] uppercase w-12 shrink-0">After:</span>
                      <span className="text-emerald-300 font-semibold truncate">
                        {corr.final_value !== null && corr.final_value !== undefined
                          ? corr.final_value
                          : "(empty)"}
                      </span>
                    </div>
                  </div>
                  <div className="text-[10px] text-slate-500 pt-1 border-t border-slate-800/60 flex items-center justify-between">
                    <span>Corrected:</span>
                    <span>{new Date(corr.corrected_at).toLocaleString()}</span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="py-2 px-3 rounded-xl bg-slate-950/50 border border-slate-800/60 text-slate-400 text-xs flex items-center justify-between">
              <span>No field corrections recorded.</span>
              <span className="text-[11px] text-slate-500 italic">
                (Factual observation: no operator changes were applied during manual review.)
              </span>
            </div>
          )}
        </div>
      )}

      {/* Operational Action & Resolution Center (MVP-9) */}
      <OperationalActionCenter
        operations={operations}
        isLoading={isLoadingOps}
        error={opsError}
        onActionClick={handleOperationalAction}
        onRefresh={loadOperations}
      />

      {/* Main Split Layout: Viewer vs Tabs */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 flex-1 min-h-[600px]">
        {/* Left Column: Document Image & BBox Viewer (7 Cols) */}
        <div className="lg:col-span-7 h-[650px]">
          {primaryPage ? (
            <DocumentViewer
              imageUrl={primaryPage.image_url}
              width={primaryPage.width}
              height={primaryPage.height}
              tokens={primaryPage.tokens}
              fields={doc.fields}
              selectedField={selectedField}
              onSelectField={(name) => {
                setSelectedField(name);
                setActiveTab("fields");
              }}
            />
          ) : (
            <div className="h-full flex items-center justify-center bg-slate-950 border border-slate-800 rounded-xl text-slate-500">
              No page rendered
            </div>
          )}
        </div>

        {/* Right Column: Tabbed Inspector Panel (5 Cols) */}
        <div className="lg:col-span-5 flex flex-col bg-slate-900/80 border border-slate-800 rounded-2xl overflow-hidden shadow-xl h-[650px]">
          {/* Tabs Navigation */}
          <div className="flex border-b border-slate-800 bg-slate-950 text-xs font-medium">
            {[
              { id: "fields", label: "Fields (KIE)", icon: ShieldCheck },
              { id: "quality", label: "Quality", icon: Activity },
              { id: "ocr", label: "OCR Text", icon: FileText },
              { id: "audit", label: "Audit Log", icon: History },
            ].map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`flex-1 py-3 flex items-center justify-center gap-1.5 border-b-2 transition-colors ${
                    isActive
                      ? "border-indigo-500 text-indigo-400 bg-slate-900/60"
                      : "border-transparent text-slate-400 hover:text-slate-200"
                  }`}
                >
                  <Icon className="w-3.5 h-3.5" />
                  {tab.label}
                </button>
              );
            })}
          </div>

          {/* Tab Content Body */}
          <div className="flex-1 p-5 overflow-y-auto space-y-4">
            {/* 1. Extracted Fields Tab */}
            {activeTab === "fields" && (
              <div className="space-y-3">
                {Object.keys(doc.fields).length === 0 ? (
                  <p className="text-xs text-slate-500 text-center py-12">
                    No structured fields extracted yet
                  </p>
                ) : (
                  Object.entries(doc.fields).map(([name, field]) => {
                    const isSelected = selectedField === name;
                    const isEditing = editingField === name;

                    return (
                      <div
                        key={name}
                        onClick={() => setSelectedField(name)}
                        className={`p-3.5 rounded-xl border transition-all cursor-pointer ${
                          isSelected
                            ? "bg-indigo-600/10 border-indigo-500/40 shadow-md"
                            : "bg-slate-950/60 border-slate-800 hover:border-slate-700"
                        }`}
                      >
                        <div className="flex items-center justify-between mb-1.5">
                          <div className="flex items-center gap-1.5">
                            <span className="text-xs font-semibold text-slate-300 uppercase tracking-wide">
                              {name}
                            </span>
                            {field.is_corrected && (
                              <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-amber-500/20 text-amber-300 border border-amber-500/30">
                                Edited
                              </span>
                            )}
                          </div>
                          <div className="flex items-center gap-2">
                            <ConfidenceBadge confidence={field.confidence * 100} size="sm" />
                            {!isEditing && (
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setEditingField(name);
                                  setEditValue(field.value);
                                }}
                                className="p-1 text-slate-400 hover:text-indigo-400 rounded transition-colors"
                                title="Edit field value"
                              >
                                <Edit2 className="w-3.5 h-3.5" />
                              </button>
                            )}
                          </div>
                        </div>

                        {isEditing ? (
                          <div
                            className="flex items-center gap-2 mt-2"
                            onClick={(e) => e.stopPropagation()}
                          >
                            <input
                              type="text"
                              value={editValue}
                              onChange={(e) => setEditValue(e.target.value)}
                              className="flex-1 bg-slate-900 border border-indigo-500 rounded-lg px-2.5 py-1 text-xs text-slate-100 focus:outline-none"
                              autoFocus
                            />
                            <button
                              onClick={() => handleSaveField(name)}
                              className="p-1 bg-indigo-600 text-white rounded-lg hover:bg-indigo-500"
                              title="Save"
                            >
                              <Check className="w-3.5 h-3.5" />
                            </button>
                            <button
                              onClick={() => setEditingField(null)}
                              className="p-1 text-slate-400 hover:text-slate-200"
                              title="Cancel"
                            >
                              <XCircle className="w-3.5 h-3.5" />
                            </button>
                          </div>
                        ) : (
                          <div>
                            <p className="text-sm font-medium text-slate-100 break-words">
                              {field.value}
                            </p>
                            {field.is_corrected && field.original_value && (
                              <p className="text-[11px] text-slate-500 mt-1 italic">
                                Original OCR: "{field.original_value}"
                              </p>
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })
                )}
              </div>
            )}

            {/* 2. Quality Profile Tab */}
            {activeTab === "quality" && (
              <div className="space-y-4">
                {doc.quality ? (
                  <>
                    <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex items-center justify-between">
                      <div>
                        <p className="text-xs text-slate-400">Quality Classification</p>
                        <p className="text-sm font-semibold text-slate-100 mt-0.5">
                          {doc.quality.profile_summary}
                        </p>
                      </div>
                      <div className="text-right">
                        <p className="text-xs text-slate-400">Score</p>
                        <p className="text-lg font-bold font-mono text-emerald-400">
                          {Math.round(doc.quality.quality_score * 100)}%
                        </p>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-3 text-xs">
                      <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800">
                        <span className="text-slate-500">Laplacian Sharpness</span>
                        <p className="font-mono text-slate-200 mt-1">{doc.quality.blur_score}</p>
                      </div>
                      <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800">
                        <span className="text-slate-500">Contrast (RMS)</span>
                        <p className="font-mono text-slate-200 mt-1">
                          {doc.quality.contrast_score}
                        </p>
                      </div>
                      <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800">
                        <span className="text-slate-500">Noise Estimate</span>
                        <p className="font-mono text-slate-200 mt-1">{doc.quality.noise_score}</p>
                      </div>
                      <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800">
                        <span className="text-slate-500">Skew Angle</span>
                        <p className="font-mono text-slate-200 mt-1">
                          {doc.quality.rotation_angle}°
                        </p>
                      </div>
                    </div>

                    {/* Research Evidence Bridge Section */}
                    <div className="mt-4 pt-4 border-t border-slate-800 space-y-3">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <FlaskConical className="w-4 h-4 text-indigo-400" />
                          <h4 className="text-xs font-semibold text-slate-200 uppercase tracking-wider">
                            Research Evidence Bridge
                          </h4>
                        </div>
                        <span className="text-[10px] text-slate-400 bg-slate-800/80 px-2 py-0.5 rounded-full border border-slate-700">
                          Controlled Research
                        </span>
                      </div>

                      {doc.research_evidence && doc.research_evidence.length > 0 ? (
                        <div className="space-y-3">
                          {doc.research_evidence.map((ref) => (
                            <div
                              key={ref.production_signal}
                              className="p-3.5 rounded-xl bg-slate-950/80 border border-slate-800/90 space-y-3"
                            >
                              {/* Signal Header & Observed Value */}
                              <div className="flex items-center justify-between">
                                <div>
                                  <span className="text-[11px] font-semibold text-slate-300 capitalize">
                                    {ref.production_signal} characteristic detected
                                  </span>
                                  {ref.observed_value !== null && ref.observed_value !== undefined && (
                                    <p className="text-xs font-mono text-indigo-300 mt-0.5">
                                      Observed value: <strong className="text-white">{ref.observed_value}</strong>{" "}
                                      <span className="text-slate-400 text-[11px]">{ref.unit || ""}</span>
                                    </p>
                                  )}
                                </div>
                                <div className="text-right">
                                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-indigo-950/60 border border-indigo-800/50 text-indigo-300 font-semibold">
                                    {ref.research_degradation_code}
                                  </span>
                                  <p className="text-[10px] text-slate-400 mt-0.5">{ref.research_degradation_name}</p>
                                </div>
                              </div>

                              {/* Research Coverage Details */}
                              <div className="grid grid-cols-2 gap-2 text-[11px] pt-1">
                                <div className="p-2.5 rounded-lg bg-slate-900/60 border border-slate-800/70 flex flex-col justify-between">
                                  <div>
                                    <div className="flex items-center gap-1.5 text-slate-400 mb-1">
                                      <span className="w-1.5 h-1.5 rounded-full bg-rose-400" />
                                      <span className="font-semibold text-slate-300">Track B1</span>
                                    </div>
                                    <p className="text-[10px] text-slate-400 leading-snug">
                                      Controlled synthetic curves across <strong>S1–S4</strong>
                                    </p>
                                  </div>
                                  {ref.b1_available && onNavigateToResearch && (
                                    <button
                                      onClick={() => onNavigateToResearch("B1", ref.research_degradation_code)}
                                      className="mt-2 w-full flex items-center justify-center gap-1 py-1 px-2 rounded-lg bg-rose-500/10 hover:bg-rose-500/20 text-rose-300 border border-rose-500/20 text-[11px] font-medium transition-colors"
                                    >
                                      <ExternalLink className="w-3 h-3" />
                                      Open B1 Evidence
                                    </button>
                                  )}
                                </div>

                                <div className="p-2.5 rounded-lg bg-slate-900/60 border border-slate-800/70 flex flex-col justify-between">
                                  <div>
                                    <div className="flex items-center gap-1.5 text-slate-400 mb-1">
                                      <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                                      <span className="font-semibold text-slate-300">Track B2</span>
                                    </div>
                                    <p className="text-[10px] text-slate-400 leading-snug">
                                      <strong>{ref.b2_evaluated_policies_count || 4} policies</strong> evaluated under {ref.research_degradation_code}
                                    </p>
                                  </div>
                                  {ref.b2_available && onNavigateToResearch && (
                                    <button
                                      onClick={() => onNavigateToResearch("B2", ref.research_degradation_code)}
                                      className="mt-2 w-full flex items-center justify-center gap-1 py-1 px-2 rounded-lg bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-300 border border-cyan-500/20 text-[11px] font-medium transition-colors"
                                    >
                                      <ExternalLink className="w-3 h-3" />
                                      Open B2 Evidence
                                    </button>
                                  )}
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="p-3 rounded-lg bg-slate-950/40 border border-slate-800/80 text-xs text-slate-400">
                          No controlled research degradation characteristics match the observed quality profile of this document.
                        </div>
                      )}

                      {/* Mandatory Research Disclaimer */}
                      <div className="p-2.5 rounded-xl bg-slate-950/50 border border-slate-800/60 text-[11px] text-slate-400 flex items-start gap-2">
                        <Info className="w-3.5 h-3.5 text-indigo-400 mt-0.5 shrink-0" />
                        <p className="leading-relaxed">
                          Research evidence describes controlled experimental conditions across synthetic S1–S4 benchmarks. It is not an individual-document prediction or preprocessing recommendation.
                        </p>
                      </div>
                    </div>

                    {/* Preprocessing Pipeline History */}
                    <div className="mt-4 pt-4 border-t border-slate-800">
                      <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">
                        Preprocessing Pipeline Provenance
                      </h4>
                      {doc.steps && doc.steps.length > 0 ? (
                        <div className="space-y-2">
                          {doc.steps.map((st) => (
                            <div
                              key={st.id || st.step_order}
                              className="p-2.5 rounded-lg bg-slate-950/70 border border-slate-800 flex items-center justify-between text-xs"
                            >
                              <div>
                                <div className="flex items-center gap-2">
                                  <span className="w-4 h-4 rounded-full bg-indigo-500/20 text-indigo-400 font-mono text-[10px] flex items-center justify-center font-bold">
                                    {st.step_order}
                                  </span>
                                  <span className="font-medium text-slate-200 uppercase tracking-wide">
                                    {st.operation}
                                  </span>
                                  <span
                                    className={`text-[10px] px-1.5 py-0.5 rounded border ${
                                      st.status === "completed"
                                        ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                                        : "bg-amber-500/10 text-amber-400 border-amber-500/20"
                                    }`}
                                  >
                                    {st.status}
                                  </span>
                                </div>
                                {st.parameters && Object.keys(st.parameters).length > 0 && (
                                  <div className="text-[11px] text-slate-400 font-mono mt-1 pl-6">
                                    {JSON.stringify(st.parameters)}
                                  </div>
                                )}
                              </div>
                              <span className="text-slate-500 font-mono text-[11px]">
                                {st.duration_ms.toFixed(1)} ms
                              </span>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="p-3 rounded-lg bg-slate-950/40 border border-slate-800/80 text-xs text-slate-400">
                          Clean document pass-through: no aggressive preprocessing operations required.
                        </div>
                      )}
                    </div>
                  </>
                ) : (
                  <p className="text-xs text-slate-500 text-center py-12">
                    Quality analysis pending
                  </p>
                )}
              </div>
            )}

            {/* 3. OCR Raw Text Tab */}
            {activeTab === "ocr" && (
              <div className="space-y-4">
                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                  <h4 className="text-xs font-semibold text-slate-400 mb-2 uppercase">
                    Full Text Transcription
                  </h4>
                  <pre className="text-xs font-mono text-slate-300 whitespace-pre-wrap leading-relaxed max-h-96 overflow-y-auto">
                    {primaryPage?.full_text || "No OCR text extracted"}
                  </pre>
                </div>
              </div>
            )}

            {/* 4. Audit Log Tab */}
            {activeTab === "audit" && (
              <div className="space-y-3">
                {doc.audit_events.map((ev) => (
                  <div
                    key={ev.id}
                    className="p-3 rounded-xl bg-slate-950 border border-slate-800/80 text-xs space-y-1"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-indigo-400 capitalize">
                        {ev.action.replace(/_/g, " ")}
                      </span>
                      <span className="text-slate-500 font-mono text-[10px]">
                        {new Date(ev.created_at).toLocaleTimeString()}
                      </span>
                    </div>
                    <p className="text-slate-400">Actor: {ev.actor}</p>
                    {ev.old_value && (
                      <p className="text-amber-400">
                        Changed: {ev.old_value} → {ev.new_value}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
