import React, { useEffect, useState } from "react";
import {
  RotateCcw,
  Sparkles,
  HelpCircle,
  FlaskConical,
  BarChart2,
  CheckCircle2,
  AlertTriangle,
  Clock,
  Layers,
  ExternalLink,
  Plus,
  X,
  ChevronLeft,
  ChevronRight,
  TrendingUp,
  FileCheck,
} from "lucide-react";
import { apiImprovement } from "../api/client";
import {
  CreateResearchQuestionRequest,
  ImprovementOverview,
  ResearchQuestion,
  ResearchQuestionStatus,
  ResearchSignal,
} from "../types";

interface Props {
  onNavigateToResearch?: (track: "B1" | "B2", degradationCode: string) => void;
}

export const ImprovementCenterPage: React.FC<Props> = ({ onNavigateToResearch }) => {
  const [period, setPeriod] = useState<string>("all");
  const [activeTab, setActiveTab] = useState<"distributions" | "signals" | "questions">("distributions");
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const [overview, setOverview] = useState<ImprovementOverview | null>(null);
  const [signals, setSignals] = useState<ResearchSignal[]>([]);
  const [questions, setQuestions] = useState<ResearchQuestion[]>([]);
  const [questionsTotal, setQuestionsTotal] = useState<number>(0);
  const [questionStatusFilter, setQuestionStatusFilter] = useState<string>("");
  const [questionPage, setQuestionPage] = useState<number>(0);
  const pageSize = 10;

  // Question Creation Modal State
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [newTitle, setNewTitle] = useState("");
  const [newDescription, setNewDescription] = useState("");
  const [newField, setNewField] = useState("");
  const [newDocType, setNewDocType] = useState("");
  const [newSignalId, setNewSignalId] = useState<string | undefined>(undefined);
  const [newRelatedRefs, setNewRelatedRefs] = useState<any[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Question Detail Modal State
  const [selectedQuestion, setSelectedQuestion] = useState<ResearchQuestion | null>(null);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [ovData, sigData, qData] = await Promise.all([
        apiImprovement.getOverview({ period }),
        apiImprovement.getSignals({ period }),
        apiImprovement.listQuestions({
          status: questionStatusFilter || undefined,
          limit: pageSize,
          offset: questionPage * pageSize,
        }),
      ]);
      setOverview(ovData);
      setSignals(sigData);
      setQuestions(qData.items);
      setQuestionsTotal(qData.total);
    } catch (err: any) {
      setError(err?.response?.data?.detail || "Failed to load improvement loop metrics.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [period, questionStatusFilter, questionPage]);

  const handleOpenCreateFromSignal = (sig: ResearchSignal) => {
    setNewTitle(`Investigate observed pattern on ${sig.field || sig.signal_type}`);
    setNewDescription(sig.description);
    setNewField(sig.field || "");
    setNewDocType(sig.document_type || "receipt");
    setNewSignalId(sig.id);
    setNewRelatedRefs(sig.related_evidence_refs || []);
    setIsCreateModalOpen(true);
  };

  const handleOpenBlankCreate = () => {
    setNewTitle("");
    setNewDescription("");
    setNewField("");
    setNewDocType("receipt");
    setNewSignalId(undefined);
    setNewRelatedRefs([]);
    setIsCreateModalOpen(true);
  };

  const handleCreateQuestionSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newTitle.trim() || !newDescription.trim()) return;

    setIsSubmitting(true);
    try {
      const payload: CreateResearchQuestionRequest = {
        title: newTitle.trim(),
        description: newDescription.trim(),
        field: newField.trim() || null,
        document_type: newDocType.trim() || null,
        source_signal_id: newSignalId || null,
        related_evidence_refs: newRelatedRefs.length > 0 ? newRelatedRefs : null,
        created_by: "researcher",
      };
      await apiImprovement.createQuestion(payload);
      setIsCreateModalOpen(false);
      fetchData();
    } catch (err: any) {
      alert(err?.response?.data?.detail || "Failed to create research question.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleStatusChange = async (questionId: string, newStatus: ResearchQuestionStatus) => {
    try {
      await apiImprovement.updateQuestion(questionId, { status: newStatus });
      fetchData();
    } catch (err: any) {
      alert(err?.response?.data?.detail || "Failed to update question status.");
    }
  };

  const getStatusBadge = (st: string) => {
    switch (st) {
      case "OPEN":
        return "bg-sky-500/10 text-sky-400 border-sky-500/30";
      case "IN_PROGRESS":
        return "bg-amber-500/10 text-amber-400 border-amber-500/30";
      case "EXPERIMENT_AVAILABLE":
        return "bg-indigo-500/10 text-indigo-400 border-indigo-500/30";
      case "CLOSED":
        return "bg-slate-500/10 text-slate-400 border-slate-500/30";
      default:
        return "bg-slate-700/20 text-slate-400 border-slate-700/40";
    }
  };

  return (
    <div className="flex-1 overflow-y-auto bg-slate-950 p-6 space-y-6 text-slate-100">
      {/* Top Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-lg bg-indigo-600/20 text-indigo-400 border border-indigo-500/30">
              <RotateCcw className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-white">
                Document AI Improvement Loop
              </h1>
              <p className="text-xs text-slate-400">
                MVP-10: Closed feedback bridge linking operator corrections to descriptive research signals, hypothesis tracking, and experimental evidence.
              </p>
            </div>
          </div>
        </div>

        {/* Period Selector & Controls */}
        <div className="flex items-center gap-3">
          <div className="flex bg-slate-900 border border-slate-800 rounded-lg p-0.5 text-xs font-medium">
            {[
              { id: "all", label: "All Time" },
              { id: "today", label: "Today" },
              { id: "7d", label: "7 Days" },
              { id: "30d", label: "30 Days" },
              { id: "90d", label: "90 Days" },
            ].map((p) => (
              <button
                key={p.id}
                onClick={() => setPeriod(p.id)}
                className={`px-3 py-1.5 rounded-md transition-colors ${
                  period === p.id
                    ? "bg-indigo-600 text-white shadow-sm"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {p.label}
              </button>
            ))}
          </div>

          <button
            onClick={fetchData}
            disabled={loading}
            className="p-2 rounded-lg bg-slate-900 border border-slate-800 text-slate-300 hover:text-white hover:bg-slate-800 transition-colors"
            title="Refresh metrics"
          >
            <RotateCcw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-lg bg-red-950/40 border border-red-800/60 text-red-300 text-xs flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* KPI Overview Cards */}
      {overview && (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800/80">
            <span className="text-[11px] font-medium text-slate-400 block uppercase tracking-wider">
              Processed
            </span>
            <div className="text-xl font-bold text-slate-100 mt-1">
              {overview.summary.documents_processed}
            </div>
            <span className="text-[10px] text-slate-500 mt-0.5 block">documents</span>
          </div>

          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800/80">
            <span className="text-[11px] font-medium text-slate-400 block uppercase tracking-wider">
              Reviewed
            </span>
            <div className="text-xl font-bold text-sky-400 mt-1">
              {overview.summary.documents_reviewed}
            </div>
            <span className="text-[10px] text-slate-500 mt-0.5 block">operator reviewed</span>
          </div>

          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800/80">
            <span className="text-[11px] font-medium text-slate-400 block uppercase tracking-wider">
              Corrected
            </span>
            <div className="text-xl font-bold text-amber-400 mt-1">
              {overview.summary.documents_corrected}
            </div>
            <span className="text-[10px] text-slate-500 mt-0.5 block">
              {overview.summary.correction_rate}% correction rate
            </span>
          </div>

          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800/80">
            <span className="text-[11px] font-medium text-slate-400 block uppercase tracking-wider">
              Field Corrections
            </span>
            <div className="text-xl font-bold text-indigo-400 mt-1">
              {overview.summary.total_field_corrections}
            </div>
            <span className="text-[10px] text-slate-500 mt-0.5 block">total edit events</span>
          </div>

          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800/80">
            <span className="text-[11px] font-medium text-slate-400 block uppercase tracking-wider">
              No-Change Rate
            </span>
            <div className="text-xl font-bold text-emerald-400 mt-1">
              {overview.summary.no_change_review_rate}%
            </div>
            <span className="text-[10px] text-slate-500 mt-0.5 block">clean confirmations</span>
          </div>

          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800/80">
            <span className="text-[11px] font-medium text-slate-400 block uppercase tracking-wider">
              Signals & Questions
            </span>
            <div className="text-xl font-bold text-purple-400 mt-1 flex items-baseline gap-1">
              <span>{overview.active_signals_count}</span>
              <span className="text-xs text-slate-500 font-normal">/ {overview.open_questions_count}</span>
            </div>
            <span className="text-[10px] text-slate-500 mt-0.5 block">signals / open questions</span>
          </div>
        </div>
      )}

      {/* Main Tab Navigation */}
      <div className="border-b border-slate-800 flex items-center justify-between">
        <div className="flex gap-2">
          <button
            onClick={() => setActiveTab("distributions")}
            className={`pb-3 px-3 text-xs font-semibold border-b-2 flex items-center gap-2 transition-colors ${
              activeTab === "distributions"
                ? "border-indigo-500 text-indigo-400"
                : "border-transparent text-slate-400 hover:text-slate-200"
            }`}
          >
            <BarChart2 className="w-4 h-4" />
            <span>Feedback Distributions</span>
          </button>

          <button
            onClick={() => setActiveTab("signals")}
            className={`pb-3 px-3 text-xs font-semibold border-b-2 flex items-center gap-2 transition-colors ${
              activeTab === "signals"
                ? "border-indigo-500 text-indigo-400"
                : "border-transparent text-slate-400 hover:text-slate-200"
            }`}
          >
            <Sparkles className="w-4 h-4" />
            <span>Research Signals</span>
            {signals.length > 0 && (
              <span className="px-1.5 py-0.2 text-[10px] rounded-full bg-indigo-500/20 text-indigo-300 font-mono">
                {signals.length}
              </span>
            )}
          </button>

          <button
            onClick={() => setActiveTab("questions")}
            className={`pb-3 px-3 text-xs font-semibold border-b-2 flex items-center gap-2 transition-colors ${
              activeTab === "questions"
                ? "border-indigo-500 text-indigo-400"
                : "border-transparent text-slate-400 hover:text-slate-200"
            }`}
          >
            <HelpCircle className="w-4 h-4" />
            <span>Research Questions</span>
            {questionsTotal > 0 && (
              <span className="px-1.5 py-0.2 text-[10px] rounded-full bg-slate-800 text-slate-300 font-mono">
                {questionsTotal}
              </span>
            )}
          </button>
        </div>

        {activeTab === "questions" && (
          <button
            onClick={handleOpenBlankCreate}
            className="mb-2 px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium flex items-center gap-1.5 shadow-sm transition-colors"
          >
            <Plus className="w-3.5 h-3.5" />
            <span>New Research Question</span>
          </button>
        )}
      </div>

      {/* Tab 1: Feedback Distributions */}
      {activeTab === "distributions" && overview && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Top Corrected Fields with Dual Metrics */}
            <div className="p-5 rounded-xl bg-slate-900/40 border border-slate-800/80 space-y-4">
              <div>
                <h3 className="text-sm font-semibold text-white">Corrections by Field</h3>
                <p className="text-xs text-slate-400">
                  Dual metric: field correction rate vs total share of corrections.
                </p>
              </div>

              {overview.by_field.length === 0 ? (
                <div className="py-8 text-center text-xs text-slate-500">
                  No field corrections recorded in this period.
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-xs text-left">
                    <thead>
                      <tr className="border-b border-slate-800 text-slate-400 uppercase tracking-wider text-[10px]">
                        <th className="pb-2 font-medium">Field</th>
                        <th className="pb-2 font-medium text-right">Corrections</th>
                        <th className="pb-2 font-medium text-right">Evaluated</th>
                        <th className="pb-2 font-medium text-right">Field Error Rate</th>
                        <th className="pb-2 font-medium text-right">Share of Total</th>
                        <th className="pb-2 font-medium text-right">Docs</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60 font-mono">
                      {overview.by_field.map((fm) => (
                        <tr key={fm.field} className="hover:bg-slate-800/20">
                          <td className="py-2.5 font-semibold text-slate-200">{fm.field}</td>
                          <td className="py-2.5 text-right text-indigo-400">{fm.correction_count}</td>
                          <td className="py-2.5 text-right text-slate-400">{fm.evaluated_count}</td>
                          <td className="py-2.5 text-right">
                            <span className={fm.correction_rate > 30 ? "px-1.5 py-0.5 rounded text-[11px] bg-amber-500/10 text-amber-300" : "text-slate-300"}>
                              {fm.correction_rate}%
                            </span>
                          </td>
                          <td className="py-2.5 text-right text-slate-400">{fm.share_of_all_corrections}%</td>
                          <td className="py-2.5 text-right text-slate-500">{fm.affected_documents}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            {/* Confidence Buckets with Strict Denominator */}
            <div className="p-5 rounded-xl bg-slate-900/40 border border-slate-800/80 space-y-4">
              <div>
                <h3 className="text-sm font-semibold text-white">Corrections by Initial Confidence Bucket</h3>
                <p className="text-xs text-slate-400">
                  Denominator strictly reflects fields evaluated with numerical confidence.
                </p>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-xs text-left">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400 uppercase tracking-wider text-[10px]">
                      <th className="pb-2 font-medium">Confidence Interval</th>
                      <th className="pb-2 font-medium text-right">Evaluated Fields</th>
                      <th className="pb-2 font-medium text-right">Corrected</th>
                      <th className="pb-2 font-medium text-right">Bucket Correction Rate</th>
                      <th className="pb-2 font-medium text-right">Share of Corrections</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-mono">
                    {overview.by_confidence.map((cb) => (
                      <tr key={cb.bucket} className="hover:bg-slate-800/20">
                        <td className="py-2.5 font-medium text-slate-200">{cb.bucket}</td>
                        <td className="py-2.5 text-right text-slate-400">{cb.evaluated_fields}</td>
                        <td className="py-2.5 text-right text-indigo-400">{cb.corrected_fields}</td>
                        <td className="py-2.5 text-right text-amber-300">{cb.correction_rate}%</td>
                        <td className="py-2.5 text-right text-slate-400">{cb.share_of_corrections}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Review Reasons */}
            <div className="p-5 rounded-xl bg-slate-900/40 border border-slate-800/80 space-y-4">
              <div>
                <h3 className="text-sm font-semibold text-white">Corrections by Triggering Review Reason</h3>
                <p className="text-xs text-slate-400">
                  Evaluates which reasons led to actual operator corrections vs clean confirmation.
                </p>
              </div>

              {overview.by_reason.length === 0 ? (
                <div className="py-8 text-center text-xs text-slate-500">
                  No review routing reasons recorded.
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-xs text-left">
                    <thead>
                      <tr className="border-b border-slate-800 text-slate-400 uppercase tracking-wider text-[10px]">
                        <th className="pb-2 font-medium">Reason</th>
                        <th className="pb-2 font-medium text-right">Reviewed</th>
                        <th className="pb-2 font-medium text-right">Corrected</th>
                        <th className="pb-2 font-medium text-right">Correction Events</th>
                        <th className="pb-2 font-medium text-right">Correction Rate</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60 font-mono">
                      {overview.by_reason.map((rm) => (
                        <tr key={rm.reason} className="hover:bg-slate-800/20">
                          <td className="py-2.5 font-medium text-slate-200">{rm.reason}</td>
                          <td className="py-2.5 text-right text-slate-400">{rm.reviewed_documents}</td>
                          <td className="py-2.5 text-right text-slate-300">{rm.documents_with_corrections}</td>
                          <td className="py-2.5 text-right text-indigo-400">{rm.correction_events}</td>
                          <td className="py-2.5 text-right text-amber-300">{rm.correction_rate}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            {/* Document Types */}
            <div className="p-5 rounded-xl bg-slate-900/40 border border-slate-800/80 space-y-4">
              <div>
                <h3 className="text-sm font-semibold text-white">Corrections by Document Type</h3>
                <p className="text-xs text-slate-400">
                  Segmentation across receipt, invoice, and other schemas.
                </p>
              </div>

              {overview.by_document_type.length === 0 ? (
                <div className="py-8 text-center text-xs text-slate-500">
                  No document types recorded.
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-xs text-left">
                    <thead>
                      <tr className="border-b border-slate-800 text-slate-400 uppercase tracking-wider text-[10px]">
                        <th className="pb-2 font-medium">Type</th>
                        <th className="pb-2 font-medium text-right">Reviewed</th>
                        <th className="pb-2 font-medium text-right">Corrected Docs</th>
                        <th className="pb-2 font-medium text-right">Correction Events</th>
                        <th className="pb-2 font-medium text-right">Correction Rate</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60 font-mono">
                      {overview.by_document_type.map((dt) => (
                        <tr key={dt.document_type} className="hover:bg-slate-800/20">
                          <td className="py-2.5 font-medium text-slate-200 uppercase">{dt.document_type}</td>
                          <td className="py-2.5 text-right text-slate-400">{dt.reviewed_documents}</td>
                          <td className="py-2.5 text-right text-slate-300">{dt.documents_with_corrections}</td>
                          <td className="py-2.5 text-right text-indigo-400">{dt.correction_events}</td>
                          <td className="py-2.5 text-right text-amber-300">{dt.correction_rate}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Tab 2: Research Signals */}
      {activeTab === "signals" && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-white">Active Research Signals</h3>
              <p className="text-xs text-slate-400">
                Deterministic, strictly observational patterns derived from production facts (Observation ≠ Recommendation).
              </p>
            </div>
            <span className="text-xs text-slate-500 font-mono">
              Sample threshold: ≥5 reviewed, ≥2 corrections
            </span>
          </div>

          {signals.length === 0 ? (
            <div className="py-16 text-center rounded-xl bg-slate-900/30 border border-slate-800/80 space-y-2">
              <Sparkles className="w-8 h-8 text-slate-600 mx-auto" />
              <h4 className="text-sm font-medium text-slate-300">No qualifying research signals detected</h4>
              <p className="text-xs text-slate-500 max-w-md mx-auto">
                Signals are derived deterministically when factual correction patterns exceed minimum sample thresholds.
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {signals.map((sig) => (
                <div
                  key={sig.id}
                  className="p-5 rounded-xl bg-slate-900/50 border border-slate-800 flex flex-col justify-between space-y-4 hover:border-slate-700 transition-colors"
                >
                  <div className="space-y-2.5">
                    <div className="flex items-center justify-between gap-2">
                      <span className="px-2 py-0.5 rounded text-[10px] font-mono font-semibold uppercase bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                        {sig.signal_type.replace(/_/g, " ")}
                      </span>
                      <span className="text-[10px] text-slate-500 font-mono">
                        {sig.id}
                      </span>
                    </div>

                    <h4 className="text-sm font-semibold text-slate-100">{sig.title}</h4>
                    <p className="text-xs text-slate-300 leading-relaxed">{sig.description}</p>
                  </div>

                  {/* Dual Evidence Visualization Panels */}
                  <div className="space-y-2.5 pt-2 border-t border-slate-800/80">
                    {/* 1. Production Evidence Panel */}
                    <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/60 space-y-1.5">
                      <div className="flex items-center gap-1.5 text-[11px] font-semibold text-emerald-400">
                        <CheckCircle2 className="w-3.5 h-3.5" />
                        <span>Observed Production Evidence</span>
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-[11px] font-mono text-slate-400">
                        {Object.entries(sig.evidence).map(([k, v]) => (
                          <div key={k} className="flex justify-between border-b border-slate-800/40 py-0.5">
                            <span className="text-slate-500">{k}:</span>
                            <span className="text-slate-200">{String(v)}</span>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* 2. Related Research Evidence Panel */}
                    {sig.related_evidence_refs && sig.related_evidence_refs.length > 0 && (
                      <div className="p-3 rounded-lg bg-indigo-950/20 border border-indigo-900/30 space-y-1.5">
                        <div className="flex items-center gap-1.5 text-[11px] font-semibold text-indigo-300">
                          <FlaskConical className="w-3.5 h-3.5" />
                          <span>Related Experimental Evidence (Research Track)</span>
                        </div>
                        <p className="text-[11px] text-slate-400">
                          Informational connection to frozen benchmarks. Not a prescriptive production recommendation.
                        </p>
                        <div className="space-y-1 pt-1">
                          {sig.related_evidence_refs.map((ref, idx) => (
                            <div key={idx} className="flex items-center justify-between text-[11px] font-mono text-indigo-200">
                              <span>{ref.name || ref.experiment_id || ref.degradation_code}</span>
                              {ref.degradation_code && onNavigateToResearch && (
                                <button
                                  onClick={() => onNavigateToResearch("B1", ref.degradation_code)}
                                  className="text-xs text-indigo-400 hover:text-indigo-300 underline flex items-center gap-1"
                                >
                                  <span>View Explorer</span>
                                  <ExternalLink className="w-3 h-3" />
                                </button>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="pt-2">
                    <button
                      onClick={() => handleOpenCreateFromSignal(sig)}
                      className="w-full py-2 px-3 rounded-lg bg-indigo-600/15 hover:bg-indigo-600/25 border border-indigo-500/30 text-indigo-300 text-xs font-medium flex items-center justify-center gap-1.5 transition-colors"
                    >
                      <Plus className="w-3.5 h-3.5" />
                      <span>Formulate Research Question</span>
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Tab 3: Research Questions */}
      {activeTab === "questions" && (
        <div className="space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div className="flex bg-slate-900 border border-slate-800 rounded-lg p-0.5 text-xs font-medium">
              {[
                { id: "", label: "All Questions" },
                { id: "OPEN", label: "Open" },
                { id: "IN_PROGRESS", label: "In Progress" },
                { id: "EXPERIMENT_AVAILABLE", label: "Experiment Available" },
                { id: "CLOSED", label: "Closed" },
              ].map((st) => (
                <button
                  key={st.id}
                  onClick={() => {
                    setQuestionStatusFilter(st.id);
                    setQuestionPage(0);
                  }}
                  className={`px-3 py-1.5 rounded-md transition-colors ${
                    questionStatusFilter === st.id
                      ? "bg-indigo-600 text-white shadow-sm"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  {st.label}
                </button>
              ))}
            </div>

            <span className="text-xs text-slate-500">
              Showing {questions.length} of {questionsTotal} persistent questions
            </span>
          </div>

          {questions.length === 0 ? (
            <div className="py-16 text-center rounded-xl bg-slate-900/30 border border-slate-800/80 space-y-3">
              <HelpCircle className="w-8 h-8 text-slate-600 mx-auto" />
              <h4 className="text-sm font-medium text-slate-300">No research questions found</h4>
              <p className="text-xs text-slate-500 max-w-sm mx-auto">
                Research questions track investigations formulated by researchers from observed signals.
              </p>
              <button
                onClick={handleOpenBlankCreate}
                className="px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium inline-flex items-center gap-1.5"
              >
                <Plus className="w-3.5 h-3.5" />
                <span>Create Research Question</span>
              </button>
            </div>
          ) : (
            <div className="space-y-3">
              {questions.map((q) => (
                <div
                  key={q.id}
                  className="p-4 rounded-xl bg-slate-900/40 border border-slate-800 flex flex-col md:flex-row md:items-center justify-between gap-4 hover:border-slate-700 transition-colors"
                >
                  <div className="space-y-1.5 flex-1">
                    <div className="flex items-center gap-2">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-semibold border ${getStatusBadge(q.status)}`}>
                        {q.status}
                      </span>
                      {q.field && (
                        <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-slate-800 text-slate-300">
                          field: {q.field}
                        </span>
                      )}
                      {q.source_signal_id && (
                        <span className="text-[10px] text-slate-500 font-mono">
                          from: {q.source_signal_id}
                        </span>
                      )}
                    </div>

                    <h4 className="text-sm font-semibold text-slate-100">{q.title}</h4>
                    <p className="text-xs text-slate-400 line-clamp-2">{q.description}</p>

                    <div className="flex items-center gap-3 text-[11px] text-slate-500 pt-1 font-mono">
                      <span>Author: {q.created_by}</span>
                      <span>Created: {new Date(q.created_at).toLocaleDateString()}</span>
                    </div>
                  </div>

                  {/* Actions & Lifecycle Transition Dropdown */}
                  <div className="flex items-center gap-3 shrink-0">
                    <div className="flex flex-col items-end gap-1">
                      <span className="text-[10px] text-slate-500 uppercase tracking-wider">
                        Transition Status
                      </span>
                      <select
                        value={q.status}
                        onChange={(e) => handleStatusChange(q.id, e.target.value as ResearchQuestionStatus)}
                        className="bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
                      >
                        <option value="OPEN">OPEN</option>
                        <option value="IN_PROGRESS">IN_PROGRESS</option>
                        <option value="EXPERIMENT_AVAILABLE">EXPERIMENT_AVAILABLE</option>
                        <option value="CLOSED">CLOSED</option>
                      </select>
                    </div>

                    {q.related_evidence_refs && q.related_evidence_refs.length > 0 && (
                      <button
                        onClick={() => setSelectedQuestion(q)}
                        className="p-2 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-300 hover:text-white transition-colors"
                        title="View Related Experimental Evidence"
                      >
                        <FlaskConical className="w-4 h-4 text-indigo-400" />
                      </button>
                    )}
                  </div>
                </div>
              ))}

              {/* Pagination Controls */}
              {questionsTotal > pageSize && (
                <div className="flex items-center justify-between pt-2">
                  <button
                    onClick={() => setQuestionPage((p) => Math.max(0, p - 1))}
                    disabled={questionPage === 0}
                    className="px-3 py-1.5 rounded bg-slate-900 border border-slate-800 text-xs text-slate-400 hover:text-white disabled:opacity-50 flex items-center gap-1"
                  >
                    <ChevronLeft className="w-3.5 h-3.5" />
                    <span>Previous</span>
                  </button>
                  <span className="text-xs text-slate-500 font-mono">
                    Page {questionPage + 1} of {Math.ceil(questionsTotal / pageSize)}
                  </span>
                  <button
                    onClick={() => setQuestionPage((p) => p + 1)}
                    disabled={(questionPage + 1) * pageSize >= questionsTotal}
                    className="px-3 py-1.5 rounded bg-slate-900 border border-slate-800 text-xs text-slate-400 hover:text-white disabled:opacity-50 flex items-center gap-1"
                  >
                    <span>Next</span>
                    <ChevronRight className="w-3.5 h-3.5" />
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Modal: Create Research Question */}
      {isCreateModalOpen && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-xl max-w-lg w-full p-6 space-y-4 shadow-xl">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2">
                <HelpCircle className="w-5 h-5 text-indigo-400" />
                <h3 className="text-sm font-semibold text-white">Create Research Question</h3>
              </div>
              <button
                onClick={() => setIsCreateModalOpen(false)}
                className="text-slate-400 hover:text-slate-200"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <form onSubmit={handleCreateQuestionSubmit} className="space-y-3.5 text-xs">
              <div className="space-y-1">
                <label className="text-slate-300 font-medium">Question Title</label>
                <input
                  type="text"
                  value={newTitle}
                  onChange={(e) => setNewTitle(e.target.value)}
                  placeholder="e.g. Investigate extraction errors on total_amount"
                  required
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-slate-100 focus:outline-none focus:border-indigo-500"
                />
              </div>

              <div className="space-y-1">
                <label className="text-slate-300 font-medium">Descriptive Investigation Context</label>
                <textarea
                  value={newDescription}
                  onChange={(e) => setNewDescription(e.target.value)}
                  rows={4}
                  placeholder="Describe the factual pattern observed in production data..."
                  required
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-slate-100 focus:outline-none focus:border-indigo-500 leading-relaxed"
                />
                <span className="text-[10px] text-slate-500 block">
                  Remember: State factual observations without assuming causality.
                </span>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <label className="text-slate-300 font-medium">Target Field (Optional)</label>
                  <input
                    type="text"
                    value={newField}
                    onChange={(e) => setNewField(e.target.value)}
                    placeholder="e.g. total"
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-slate-100 focus:outline-none focus:border-indigo-500 font-mono"
                  />
                </div>

                <div className="space-y-1">
                  <label className="text-slate-300 font-medium">Document Type</label>
                  <input
                    type="text"
                    value={newDocType}
                    onChange={(e) => setNewDocType(e.target.value)}
                    placeholder="e.g. receipt"
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-slate-100 focus:outline-none focus:border-indigo-500 font-mono"
                  />
                </div>
              </div>

              {newSignalId && (
                <div className="p-2.5 rounded-lg bg-indigo-950/20 border border-indigo-900/30 text-[11px] text-indigo-300 font-mono">
                  Originating Signal ID: {newSignalId}
                </div>
              )}

              <div className="flex justify-end gap-2 pt-3 border-t border-slate-800">
                <button
                  type="button"
                  onClick={() => setIsCreateModalOpen(false)}
                  className="px-3 py-1.5 rounded-lg bg-slate-800 text-slate-300 hover:text-white"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isSubmitting}
                  className="px-4 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-medium disabled:opacity-50"
                >
                  {isSubmitting ? "Creating..." : "Save Question"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Modal: View Related Evidence */}
      {selectedQuestion && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-xl max-w-lg w-full p-6 space-y-4 shadow-xl">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2">
                <FlaskConical className="w-5 h-5 text-indigo-400" />
                <h3 className="text-sm font-semibold text-white">Related Research Evidence</h3>
              </div>
              <button
                onClick={() => setSelectedQuestion(null)}
                className="text-slate-400 hover:text-slate-200"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="space-y-3 text-xs">
              <div>
                <span className="text-slate-500 uppercase tracking-wider text-[10px] block">Question</span>
                <p className="font-semibold text-slate-200">{selectedQuestion.title}</p>
              </div>

              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800/80 space-y-2">
                <span className="text-[11px] font-semibold text-indigo-300 block">
                  Linked B0/B1/B2 Benchmarks (Informational)
                </span>
                <p className="text-[11px] text-slate-400 leading-relaxed">
                  These controlled experiments explore related dimensions in research datasets. They do not dictate automated production behavior.
                </p>
                <div className="space-y-1.5 pt-1">
                  {selectedQuestion.related_evidence_refs.map((ref: any, idx: number) => (
                    <div key={idx} className="p-2 rounded bg-slate-900 border border-slate-800 flex items-center justify-between">
                      <div>
                        <span className="font-semibold text-slate-200 block">
                          {ref.name || ref.experiment_id || ref.degradation_code}
                        </span>
                        <span className="text-[10px] text-slate-500 font-mono">Track: {ref.track || "B1/B2"}</span>
                      </div>
                      {ref.degradation_code && onNavigateToResearch && (
                        <button
                          onClick={() => {
                            setSelectedQuestion(null);
                            onNavigateToResearch("B1", ref.degradation_code);
                          }}
                          className="text-xs text-indigo-400 hover:text-indigo-300 underline flex items-center gap-1"
                        >
                          <span>Open Explorer</span>
                          <ExternalLink className="w-3 h-3" />
                        </button>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="flex justify-end pt-2">
              <button
                onClick={() => setSelectedQuestion(null)}
                className="px-4 py-1.5 rounded-lg bg-slate-800 text-slate-300 hover:text-white text-xs"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
