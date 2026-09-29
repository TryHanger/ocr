import React, { useEffect, useState, useRef } from "react";
import { DocumentDetail, DocumentItem, NormalizedField } from "../types";
import { apiDocuments, apiReview } from "../api/client";
import { DocumentViewer } from "../components/DocumentViewer";
import { ConfidenceBadge } from "../components/ConfidenceBadge";
import {
  CheckSquare,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Save,
  ExternalLink,
  ShieldAlert,
  Clock,
} from "lucide-react";

interface ManualReviewPageProps {
  onInspect: (id: string) => void;
}

const REQUIRED_RECEIPT_FIELDS = ["company", "date", "total"];

export const ManualReviewPage: React.FC<ManualReviewPageProps> = ({ onInspect }) => {
  const [reviewDocs, setReviewDocs] = useState<DocumentItem[]>([]);
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);
  const [docDetail, setDocDetail] = useState<DocumentDetail | null>(null);
  const [fieldValues, setFieldValues] = useState<Record<string, string>>({});
  const [selectedField, setSelectedField] = useState<string | null>(null);
  const [savingField, setSavingField] = useState<string | null>(null);
  const [savedFields, setSavedFields] = useState<Record<string, boolean>>({});
  const [validationErrorBanner, setValidationErrorBanner] = useState<string | null>(null);
  const [isCompleting, setIsCompleting] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  const inputRefs = useRef<Record<string, HTMLInputElement | null>>({});

  const loadReviewQueue = async () => {
    try {
      const res = await apiDocuments.list({ status: "manual_review", limit: 50 });
      setReviewDocs(res.items);
      if (res.items.length > 0 && !selectedDocId) {
        setSelectedDocId(res.items[0].id);
      } else if (res.items.length === 0) {
        setSelectedDocId(null);
        setDocDetail(null);
      }
    } catch (err) {
      console.error("Failed to load review queue", err);
    } finally {
      setIsLoading(false);
    }
  };

  const loadSelectedDoc = async (id: string) => {
    try {
      setValidationErrorBanner(null);
      setSavedFields({});
      const data = await apiDocuments.get(id);
      setDocDetail(data);
      const initialVals: Record<string, string> = {};
      Object.entries(data.fields).forEach(([k, v]) => {
        initialVals[k] = v.corrected_value !== undefined && v.corrected_value !== null ? v.corrected_value : v.value;
      });
      // Pre-seed any missing standard fields with empty strings
      REQUIRED_RECEIPT_FIELDS.forEach((rf) => {
        if (initialVals[rf] === undefined) {
          initialVals[rf] = "";
        }
      });
      setFieldValues(initialVals);
    } catch (err) {
      console.error("Failed to load document detail for review", err);
    }
  };

  useEffect(() => {
    loadReviewQueue();
  }, []);

  useEffect(() => {
    if (selectedDocId) {
      loadSelectedDoc(selectedDocId);
    }
  }, [selectedDocId]);

  const handleSelectFieldFromViewer = (fieldName: string) => {
    setSelectedField(fieldName);
    if (inputRefs.current[fieldName]) {
      inputRefs.current[fieldName]?.focus();
      inputRefs.current[fieldName]?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  };

  const handleSaveFieldCorrection = async (fieldName: string) => {
    if (!selectedDocId) return;
    const val = fieldValues[fieldName] ?? "";
    setSavingField(fieldName);
    try {
      const updatedField = await apiReview.correctField(selectedDocId, fieldName, val, "operator");
      if (docDetail) {
        setDocDetail({
          ...docDetail,
          fields: {
            ...docDetail.fields,
            [fieldName]: updatedField,
          },
        });
      }
      setSavedFields((prev) => ({ ...prev, [fieldName]: true }));
      setTimeout(() => {
        setSavedFields((prev) => ({ ...prev, [fieldName]: false }));
      }, 2500);
      setValidationErrorBanner(null);
    } catch (err: any) {
      const msg = err.response?.data?.detail || err.message || "Failed to save field correction";
      setValidationErrorBanner(`Error saving '${fieldName}': ${msg}`);
    } finally {
      setSavingField(null);
    }
  };

  const handleCompleteReview = async () => {
    if (!selectedDocId) return;
    setIsCompleting(true);
    setValidationErrorBanner(null);
    try {
      await apiReview.completeReview(selectedDocId, "operator");
      const nextQueue = reviewDocs.filter((d) => d.id !== selectedDocId);
      setReviewDocs(nextQueue);
      if (nextQueue.length > 0) {
        setSelectedDocId(nextQueue[0].id);
      } else {
        setSelectedDocId(null);
        setDocDetail(null);
      }
    } catch (err: any) {
      const detail = err.response?.data?.detail || "Validation check failed before review completion";
      setValidationErrorBanner(detail);
      if (selectedDocId) {
        loadSelectedDoc(selectedDocId);
      }
    } finally {
      setIsCompleting(false);
    }
  };

  const handleReject = async () => {
    if (!selectedDocId) return;
    const reason = prompt("Enter rejection reason:", "Document unreadable or invalid");
    if (reason) {
      try {
        await apiReview.reject(selectedDocId, reason, "operator");
        const nextQueue = reviewDocs.filter((d) => d.id !== selectedDocId);
        setReviewDocs(nextQueue);
        if (nextQueue.length > 0) {
          setSelectedDocId(nextQueue[0].id);
        } else {
          setSelectedDocId(null);
          setDocDetail(null);
        }
      } catch (err: any) {
        setValidationErrorBanner(err.response?.data?.detail || "Failed to reject document");
      }
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-24 text-slate-400">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-amber-500"></div>
      </div>
    );
  }

  if (reviewDocs.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center p-24 text-slate-500 border border-dashed border-slate-800 rounded-2xl max-w-xl mx-auto text-center">
        <CheckCircle className="w-12 h-12 text-emerald-500 mb-3 stroke-1" />
        <h3 className="text-base font-semibold text-slate-200">Review Queue is Clear!</h3>
        <p className="text-xs text-slate-400 mt-1">
          All ingested documents have satisfied automated confidence thresholds or have already been reviewed.
        </p>
      </div>
    );
  }

  const primaryPage = docDetail?.pages[0];

  const allFieldNames = Array.from(
    new Set([...Object.keys(docDetail?.fields || {}), ...REQUIRED_RECEIPT_FIELDS])
  );

  return (
    <div className="space-y-4 max-w-7xl mx-auto h-full flex flex-col">
      {/* Top Header Bar */}
      <div className="flex items-center justify-between bg-slate-900/80 p-4 rounded-2xl border border-slate-800 shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-amber-500/10 text-amber-400 flex items-center justify-center">
            <CheckSquare className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-sm font-semibold text-slate-100">Manual Review Station</h2>
              <span className="px-2 py-0.5 rounded-full text-[11px] font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/20">
                {reviewDocs.length} pending
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Inspect OCR detections, correct questionable fields, and sign off via the validation gate.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {selectedDocId && (
            <button
              onClick={() => onInspect(selectedDocId)}
              className="flex items-center gap-1.5 px-3 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-xl text-xs font-semibold transition-all"
              title="Open full inspector"
            >
              <ExternalLink className="w-3.5 h-3.5" />
              Inspector
            </button>
          )}

          <button
            onClick={handleReject}
            className="flex items-center gap-1.5 px-3 py-2 bg-rose-600/20 text-rose-300 hover:bg-rose-600/30 border border-rose-500/30 rounded-xl text-xs font-semibold active:scale-95 transition-all"
          >
            <XCircle className="w-4 h-4" />
            Reject
          </button>

          <button
            onClick={handleCompleteReview}
            disabled={isCompleting}
            className="flex items-center gap-1.5 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white rounded-xl text-xs font-semibold shadow-lg shadow-emerald-600/20 active:scale-95 transition-all"
          >
            <CheckCircle className="w-4 h-4" />
            {isCompleting ? "Validating & Completing..." : "Complete Review"}
          </button>
        </div>
      </div>

      {/* Validation Error Banner (When gate rejects completion) */}
      {validationErrorBanner && (
        <div className="p-3.5 bg-rose-950/50 border border-rose-500/50 rounded-xl flex items-start gap-3 text-xs text-rose-200 animate-fadeIn">
          <ShieldAlert className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
          <div className="flex-1">
            <span className="font-semibold text-rose-300">Validation Gate Blocked Completion:</span>
            <p className="mt-0.5 text-rose-200/90">{validationErrorBanner}</p>
            <p className="mt-1 text-[11px] text-rose-400">
              Please correct the invalid or missing field values below, click <strong>Save</strong> on each, then retry <strong>Complete Review</strong>.
            </p>
          </div>
        </div>
      )}

      {/* Active Document Info Banner */}
      {docDetail && (
        <div className="flex items-center justify-between bg-slate-950 px-4 py-2.5 rounded-xl border border-slate-800 text-xs">
          <div className="flex items-center gap-3">
            <span className="font-mono text-slate-300 font-semibold">{docDetail.filename}</span>
            <span className="text-slate-500">·</span>
            <span className="text-amber-400 flex items-center gap-1">
              <AlertTriangle className="w-3.5 h-3.5" />
              Reason: <strong>{docDetail.review_reason?.replace(/_/g, " ") || "Verification required"}</strong>
            </span>
            {docDetail.review_details?.reasons && docDetail.review_details.reasons.length > 0 && (
              <span className="text-slate-400 text-[11px]">
                ({docDetail.review_details.reasons.join(", ")})
              </span>
            )}
          </div>
          <div className="flex items-center gap-3">
            <ConfidenceBadge confidence={docDetail.confidence} size="sm" />
            {docDetail.review_started_at && (
              <span className="text-[11px] text-slate-500 flex items-center gap-1">
                <Clock className="w-3 h-3" />
                Wait: {docDetail.review_wait_duration_ms ? `${(docDetail.review_wait_duration_ms / 1000).toFixed(1)}s` : "active"}
              </span>
            )}
          </div>
        </div>
      )}

      {/* Main 3-Column Review Workspace */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 flex-1 min-h-[600px]">
        {/* Left Queue List (3 Cols) */}
        <div className="lg:col-span-3 bg-slate-900/80 border border-slate-800 rounded-2xl overflow-hidden p-3 space-y-2 h-[680px] overflow-y-auto">
          <p className="text-xs font-semibold text-slate-400 px-2 py-1 uppercase tracking-wider">
            Review Queue ({reviewDocs.length})
          </p>
          {reviewDocs.map((doc) => {
            const isSelected = selectedDocId === doc.id;
            return (
              <button
                key={doc.id}
                onClick={() => setSelectedDocId(doc.id)}
                className={`w-full text-left p-3 rounded-xl border transition-all ${
                  isSelected
                    ? "bg-amber-500/10 border-amber-500/40 shadow-md ring-1 ring-amber-500/30"
                    : "bg-slate-950/60 border-slate-800/80 hover:border-slate-700"
                }`}
              >
                <p className="text-xs font-medium text-slate-200 truncate">{doc.filename}</p>
                <div className="flex items-center justify-between mt-2">
                  <span className="text-[11px] text-amber-400 font-medium capitalize">
                    {doc.review_reason?.replace(/_/g, " ") || "Review required"}
                  </span>
                  <ConfidenceBadge confidence={doc.confidence} size="sm" />
                </div>
              </button>
            );
          })}
        </div>

        {/* Center Viewer with Bounding Box Overlay (5 Cols) */}
        <div className="lg:col-span-5 h-[680px]">
          {primaryPage ? (
            <DocumentViewer
              imageUrl={primaryPage.image_url}
              width={primaryPage.width}
              height={primaryPage.height}
              tokens={primaryPage.tokens}
              fields={docDetail?.fields}
              selectedField={selectedField}
              onSelectField={handleSelectFieldFromViewer}
            />
          ) : (
            <div className="h-full flex items-center justify-center bg-slate-950 border border-slate-800 rounded-xl text-slate-500 text-xs">
              Select a document to review
            </div>
          )}
        </div>

        {/* Right Fields Form (4 Cols) */}
        <div className="lg:col-span-4 bg-slate-900/80 border border-slate-800 rounded-2xl p-4 space-y-3.5 h-[680px] overflow-y-auto shadow-xl">
          <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
            <div>
              <h3 className="text-xs font-semibold text-slate-200 uppercase tracking-wider">
                Field Corrections
              </h3>
              <p className="text-[11px] text-slate-400 mt-0.5">
                Click a field to highlight on scan; click bbox on scan to jump here.
              </p>
            </div>
          </div>

          {docDetail &&
            allFieldNames.map((name) => {
              const field: NormalizedField | undefined = docDetail.fields[name];
              const isMissing = !field || !field.value;
              const isLowConf = field ? field.confidence < 0.85 : true;
              const isSelected = selectedField === name;
              const isSaved = !!savedFields[name];
              const isSaving = savingField === name;
              const isDirty = field ? (fieldValues[name] ?? "") !== (field.corrected_value || field.value) : !!(fieldValues[name] ?? "");

              return (
                <div
                  key={name}
                  onClick={() => setSelectedField(name)}
                  className={`p-3 rounded-xl border transition-all ${
                    isSelected
                      ? "bg-indigo-600/10 border-indigo-500/50 shadow-md ring-1 ring-indigo-500/30"
                      : isMissing
                      ? "bg-rose-950/20 border-rose-500/30 hover:border-rose-500/50"
                      : isLowConf
                      ? "bg-amber-500/5 border-amber-500/30 hover:border-amber-500/50"
                      : "bg-slate-950 border-slate-800 hover:border-slate-700"
                  }`}
                >
                  <div className="flex items-center justify-between mb-1.5">
                    <label className="text-xs font-semibold text-slate-300 uppercase flex items-center gap-1.5">
                      {isMissing ? (
                        <span className="text-rose-400 flex items-center gap-1 font-mono text-[11px]">
                          <AlertTriangle className="w-3.5 h-3.5 text-rose-400" />
                          {name} (Missing)
                        </span>
                      ) : (
                        <>
                          {isLowConf && <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />}
                          {name}
                        </>
                      )}
                    </label>

                    <div className="flex items-center gap-1.5">
                      {field?.validation_status === "INVALID" && (
                        <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-rose-500/20 text-rose-300 border border-rose-500/30">
                          INVALID
                        </span>
                      )}
                      {field?.validation_status === "VALID" && (
                        <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                          VALID
                        </span>
                      )}
                      {field && <ConfidenceBadge confidence={field.confidence * 100} size="sm" />}
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <input
                      ref={(el) => {
                        inputRefs.current[name] = el;
                      }}
                      type="text"
                      value={fieldValues[name] ?? ""}
                      placeholder={`Enter ${name}...`}
                      onFocus={() => setSelectedField(name)}
                      onChange={(e) =>
                        setFieldValues({ ...fieldValues, [name]: e.target.value })
                      }
                      className="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-100 focus:outline-none focus:border-indigo-500 font-mono"
                    />

                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleSaveFieldCorrection(name);
                      }}
                      disabled={isSaving}
                      className={`px-2.5 py-1.5 rounded-lg text-xs font-medium flex items-center gap-1 transition-all ${
                        isSaved
                          ? "bg-emerald-600 text-white"
                          : isDirty
                          ? "bg-indigo-600 hover:bg-indigo-500 text-white shadow"
                          : "bg-slate-800 hover:bg-slate-700 text-slate-300"
                      }`}
                      title="Save field correction"
                    >
                      {isSaved ? (
                        <>
                          <CheckCircle className="w-3.5 h-3.5" />
                          Saved
                        </>
                      ) : (
                        <>
                          <Save className="w-3.5 h-3.5" />
                          {isSaving ? "Saving..." : "Save"}
                        </>
                      )}
                    </button>
                  </div>

                  {field?.original_value && field.original_value !== (fieldValues[name] ?? "") && (
                    <p className="text-[10px] text-slate-500 mt-1 italic">
                      Original OCR: "{field.original_value}"
                    </p>
                  )}
                </div>
              );
            })}
        </div>
      </div>
    </div>
  );
};
