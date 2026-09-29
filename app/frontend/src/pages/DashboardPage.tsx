import React, { useEffect, useState } from "react";
import {
  ControlCenterResponse,
  ImprovementOverview,
  OperationsBacklogSummary,
  TimePeriod,
} from "../types";
import { apiAnalytics, apiDocuments, apiImprovement, apiOperations } from "../api/client";
import { DocumentTable } from "../components/DocumentTable";
import {
  TrendingUp,
  FileCheck,
  AlertTriangle,
  Clock,
  Gauge,
  Layers,
  ArrowRight,
  RefreshCw,
  FlaskConical,
  BarChart3,
  ShieldCheck,
  Edit3,
  Activity,
} from "lucide-react";

interface DashboardPageProps {
  onInspect: (id: string) => void;
  onReview: (id: string) => void;
  onViewAllDocuments: () => void;
  onNavigateToReview?: () => void;
  onNavigateToAnalytics?: () => void;
  onNavigateToResearch?: () => void;
  onNavigateToImprovement?: () => void;
}

export const DashboardPage: React.FC<DashboardPageProps> = ({
  onInspect,
  onReview,
  onViewAllDocuments,
  onNavigateToReview,
  onNavigateToAnalytics,
  onNavigateToResearch,
  onNavigateToImprovement,
}) => {
  const [period, setPeriod] = useState<TimePeriod>("30d");
  const [customFrom, setCustomFrom] = useState("");
  const [customTo, setCustomTo] = useState("");
  const [data, setData] = useState<ControlCenterResponse | null>(null);
  const [backlog, setBacklog] = useState<OperationsBacklogSummary | null>(null);
  const [improvement, setImprovement] = useState<ImprovementOverview | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadData = async (showRefreshIndicator = false) => {
    if (showRefreshIndicator) {
      setIsRefreshing(true);
    }
    setError(null);

    try {
      const params: any = { period };
      if (period === "custom") {
        if (customFrom) {
          const fromDate = new Date(customFrom);
          params.from = fromDate.toISOString().replace(/\.\d{3}Z$/, "Z");
        }
        if (customTo) {
          const toDate = new Date(customTo);
          params.to = toDate.toISOString().replace(/\.\d{3}Z$/, "Z");
        }
      }
      const [response, backlogData, improvementData] = await Promise.all([
        apiAnalytics.getControlCenter(params),
        apiOperations.getBacklog().catch(() => null),
        apiImprovement.getOverview(params).catch(() => null),
      ]);
      setData(response);
      setBacklog(backlogData);
      setImprovement(improvementData);
    } catch (err: any) {
      console.error("Failed to load Control Center data", err);
      setError(err?.message || "Failed to load Control Center metrics. Please verify backend connection.");
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  };

  useEffect(() => {
    if (period !== "custom" || (customFrom && customTo)) {
      loadData();
    }
    const interval = setInterval(() => {
      if (period !== "custom" || (customFrom && customTo)) {
        loadData(false);
      }
    }, 6000);
    return () => clearInterval(interval);
  }, [period, customFrom, customTo]);

  const summary = data?.summary;
  const queue = data?.queue;
  const funnel = data?.funnel;
  const quality = data?.quality;
  const feedback = data?.feedback;

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-12">
      {/* 1. Header & Period Filter */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <div className="flex items-center gap-2.5">
            <h1 className="text-2xl font-bold text-slate-100 tracking-tight">Document AI Control Center</h1>
            <span className="px-2 py-0.5 text-[10px] font-semibold tracking-wider uppercase rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
              Live Operations
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Centralized operational throughput, real-time backlog, decision explainability, and quality telemetry
          </p>
        </div>

        <div className="flex flex-col sm:flex-row items-end sm:items-center gap-3 w-full sm:w-auto">
          {period === "custom" && (
            <div className="flex items-center gap-2 text-xs">
              <input
                type="date"
                value={customFrom}
                onChange={(e) => setCustomFrom(e.target.value)}
                className="bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1 text-slate-200"
              />
              <span className="text-slate-500">to</span>
              <input
                type="date"
                value={customTo}
                onChange={(e) => setCustomTo(e.target.value)}
                className="bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1 text-slate-200"
              />
            </div>
          )}

          <div className="flex items-center gap-1.5 bg-slate-900/90 p-1 rounded-xl border border-slate-800 shadow-sm">
            {(["today", "7d", "30d", "90d", "all", "custom"] as TimePeriod[]).map((p) => (
              <button
                key={p}
                onClick={() => setPeriod(p)}
                className={`px-3 py-1 rounded-lg text-xs font-semibold uppercase tracking-wider transition-colors ${
                  period === p
                    ? "bg-indigo-600 text-white shadow-sm"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {p === "all" ? "All Time" : p}
              </button>
            ))}

            <button
              onClick={() => loadData(true)}
              disabled={isRefreshing}
              className="p-1.5 text-slate-400 hover:text-slate-200 hover:bg-slate-800 rounded-lg transition-colors ml-1"
              title="Refresh Control Center"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isRefreshing ? "animate-spin text-indigo-400" : ""}`} />
            </button>
          </div>
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 flex items-center justify-between text-rose-300 text-xs">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
            <span>{error}</span>
          </div>
          <button
            onClick={() => loadData(true)}
            className="px-3 py-1 bg-rose-500/20 hover:bg-rose-500/30 text-rose-200 rounded-lg font-medium transition-colors"
          >
            Retry
          </button>
        </div>
      )}

      {/* 2. Real-Time Operational Queue Backlog */}
      <div className="p-5 rounded-2xl bg-gradient-to-r from-slate-900/90 via-slate-900/70 to-indigo-950/30 border border-indigo-500/20 shadow-xl relative overflow-hidden">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 border-b border-slate-800/80 pb-3">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-indigo-400 animate-pulse" />
            <span className="text-xs font-bold uppercase tracking-wider text-indigo-300">
              Current Operational Backlog (Live State)
            </span>
          </div>
          <span className="text-[11px] text-slate-400 italic">
            Real-time queue state independent of historical reporting period
          </span>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
          {/* Manual Review Queue */}
          <div
            onClick={onNavigateToReview}
            className="p-3.5 rounded-xl bg-slate-950/60 border border-amber-500/20 hover:border-amber-500/40 transition-colors cursor-pointer group"
          >
            <div className="flex items-center justify-between">
              <span className="text-xs text-slate-400 group-hover:text-amber-300 transition-colors">
                Waiting for Review
              </span>
              <span className="text-[10px] text-amber-400/80 font-mono">LIVE</span>
            </div>
            <div className="mt-2 flex items-baseline justify-between">
              <span className="text-2xl font-bold font-mono text-amber-300">
                {queue ? queue.manual_review : "—"}
              </span>
              {onNavigateToReview && (
                <span className="text-[11px] text-amber-400 group-hover:underline flex items-center gap-0.5">
                  Queue <ArrowRight className="w-3 h-3" />
                </span>
              )}
            </div>
            <p className="text-[10px] text-slate-500 mt-1">Pending human operator action</p>
          </div>

          {/* In Processing */}
          <div className="p-3.5 rounded-xl bg-slate-950/60 border border-indigo-500/20">
            <div className="flex items-center justify-between">
              <span className="text-xs text-slate-400">In Processing</span>
              <span className="text-[10px] text-indigo-400/80 font-mono">LIVE</span>
            </div>
            <div className="mt-2">
              <span className="text-2xl font-bold font-mono text-indigo-300">
                {queue ? queue.processing : "—"}
              </span>
            </div>
            <p className="text-[10px] text-slate-500 mt-1">Active pipeline workers</p>
          </div>

          {/* Processing Errors */}
          <div className="p-3.5 rounded-xl bg-slate-950/60 border border-rose-500/20">
            <div className="flex items-center justify-between">
              <span className="text-xs text-slate-400">Processing Errors</span>
              <span className="text-[10px] text-rose-400/80 font-mono">LIVE</span>
            </div>
            <div className="mt-2">
              <span className="text-2xl font-bold font-mono text-rose-300">
                {queue ? queue.errors : "—"}
              </span>
            </div>
            <p className="text-[10px] text-slate-500 mt-1">Pipeline execution errors</p>
          </div>

          {/* Total Active Backlog */}
          <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800">
            <div className="flex items-center justify-between">
              <span className="text-xs text-slate-400">Total Active Backlog</span>
              <span className="text-[10px] text-slate-400 font-mono">LIVE</span>
            </div>
            <div className="mt-2">
              <span className="text-2xl font-bold font-mono text-slate-100">
                {queue ? queue.manual_review + queue.processing : "—"}
              </span>
            </div>
            <p className="text-[10px] text-slate-500 mt-1">Current unfinalized workload</p>
          </div>
        </div>

        {/* Live Reason & Category Breakdown from MVP-9 Operational Backlog */}
        {backlog && (backlog.total_open > 0 || backlog.total_in_review > 0) && (
          <div className="mt-4 pt-3 border-t border-slate-800/80 space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold text-slate-300 uppercase tracking-wider text-[11px]">
                Waiting for Review by Root Cause Reason:
              </span>
              <div className="flex items-center gap-3 text-[11px] font-mono">
                <span className="text-amber-400">Open (Unassigned): {backlog.total_open}</span>
                <span className="text-slate-600">|</span>
                <span className="text-blue-400">In Review: {backlog.total_in_review}</span>
              </div>
            </div>

            <div className="flex flex-wrap gap-2 pt-1">
              {Object.entries(backlog.by_reason).map(([reason, count]) => (
                <div
                  key={reason}
                  onClick={onNavigateToReview}
                  className="px-2.5 py-1 rounded-lg bg-slate-950/80 border border-slate-800 hover:border-amber-500/40 cursor-pointer transition-colors flex items-center gap-2 text-xs font-mono"
                  title="Click to view manual review queue"
                >
                  <span className="text-slate-300 font-semibold">{reason}</span>
                  <span className="px-1.5 py-0.2 rounded bg-amber-950/80 text-amber-300 font-bold border border-amber-800/60 text-[10px]">
                    {count}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* 3. Primary Historical KPI Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
        {/* Documents Processed */}
        <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-lg relative flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-slate-400">Processed</span>
              <FileCheck className="w-4 h-4 text-indigo-400" />
            </div>
            <div className="mt-2.5 flex items-baseline gap-1.5">
              <span className="text-2xl font-bold font-mono text-slate-100">
                {summary ? summary.processed_count : "—"}
              </span>
              <span className="text-[11px] text-slate-500">of {summary?.total_documents || 0}</span>
            </div>
          </div>
          <div className="mt-3 pt-2 border-t border-slate-800/80 text-[10px] text-slate-400 space-y-0.5">
            <div>Auto: <span className="text-emerald-400 font-mono">{summary?.automatic_count || 0}</span></div>
            <div>Reviewed: <span className="text-amber-400 font-mono">{summary?.completed_manual_count || 0}</span></div>
          </div>
        </div>

        {/* Automation Rate */}
        <div
          onClick={onNavigateToAnalytics}
          className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 hover:border-emerald-500/30 transition-colors shadow-lg relative flex flex-col justify-between cursor-pointer group"
        >
          <div>
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-slate-400 group-hover:text-emerald-400 transition-colors">
                Automation Rate
              </span>
              <TrendingUp className="w-4 h-4 text-emerald-400" />
            </div>
            <div className="mt-2.5 flex items-baseline gap-1">
              <span className="text-2xl font-bold font-mono text-emerald-400">
                {summary ? `${summary.automation_rate}%` : "—"}
              </span>
            </div>
          </div>
          <div className="mt-3 pt-2 border-t border-slate-800/80 text-[10px] text-slate-500" title="Automatic decisions / processed documents">
            Automatic / processed documents
          </div>
        </div>

        {/* Manual Review Rate */}
        <div
          onClick={onNavigateToReview}
          className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 hover:border-amber-500/30 transition-colors shadow-lg relative flex flex-col justify-between cursor-pointer group"
        >
          <div>
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-slate-400 group-hover:text-amber-400 transition-colors">
                Manual Review Rate
              </span>
              <AlertTriangle className="w-4 h-4 text-amber-400" />
            </div>
            <div className="mt-2.5 flex items-baseline gap-1">
              <span className="text-2xl font-bold font-mono text-amber-400">
                {summary ? `${summary.manual_review_rate}%` : "—"}
              </span>
            </div>
          </div>
          <div className="mt-3 pt-2 border-t border-slate-800/80 text-[10px] text-slate-500" title="Manual review decisions / processed documents">
            Manual reviews / processed documents
          </div>
        </div>

        {/* Correction Rate */}
        <div
          onClick={onNavigateToAnalytics}
          className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 hover:border-purple-500/30 transition-colors shadow-lg relative flex flex-col justify-between cursor-pointer group"
        >
          <div>
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-slate-400 group-hover:text-purple-400 transition-colors">
                Correction Rate
              </span>
              <Edit3 className="w-4 h-4 text-purple-400" />
            </div>
            <div className="mt-2.5 flex items-baseline gap-1">
              <span className="text-2xl font-bold font-mono text-purple-300">
                {summary ? `${summary.correction_rate}%` : "—"}
              </span>
            </div>
          </div>
          <div className="mt-3 pt-2 border-t border-slate-800/80 text-[10px] text-slate-500" title="Corrected documents / completed manual reviews">
            Corrected / completed manual reviews
          </div>
        </div>

        {/* Average Processing Time */}
        <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-lg relative flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-slate-400">Avg Speed</span>
              <Clock className="w-4 h-4 text-indigo-400" />
            </div>
            <div className="mt-2.5 flex items-baseline gap-1.5">
              <span className="text-2xl font-bold font-mono text-slate-100">
                {summary ? `${(summary.average_processing_time_ms / 1000).toFixed(2)}s` : "—"}
              </span>
            </div>
          </div>
          <div className="mt-3 pt-2 border-t border-slate-800/80 text-[10px] text-slate-500">
            {summary?.average_processing_time_ms || 0} ms pipeline latency
          </div>
        </div>

        {/* Average Confidence */}
        <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-lg relative flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-slate-400">Avg Confidence</span>
              <Gauge className="w-4 h-4 text-indigo-400" />
            </div>
            <div className="mt-2.5 flex items-baseline gap-1.5">
              <span className="text-2xl font-bold font-mono text-slate-100">
                {summary ? `${(summary.average_confidence * 100).toFixed(1)}%` : "—"}
              </span>
            </div>
          </div>
          <div className="mt-3 pt-2 border-t border-slate-800/80 text-[10px] text-slate-500">
            Normalized composite score (0..1)
          </div>
        </div>
      </div>

      {/* MVP-10 Document AI Improvement Loop Status Banner */}
      {improvement && (
        <div className="p-4 rounded-2xl bg-indigo-950/20 border border-indigo-500/20 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center text-indigo-400 shrink-0">
              <FlaskConical className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-sm font-semibold text-slate-200">
                  Document AI Improvement Loop
                </span>
                <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                  MVP-10
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-0.5">
                Connecting operator corrections to deterministic research signals and hypothesis questions.
              </p>
            </div>
          </div>

          <div className="flex items-center gap-6">
            <div className="text-right">
              <span className="text-[11px] text-slate-400 block">Active Signals</span>
              <span className="text-lg font-bold font-mono text-indigo-300">
                {improvement.active_signals_count}
              </span>
            </div>
            <div className="h-8 w-px bg-slate-800" />
            <div className="text-right">
              <span className="text-[11px] text-slate-400 block">Open Questions</span>
              <span className="text-lg font-bold font-mono text-amber-300">
                {improvement.open_questions_count}
              </span>
            </div>
            <div className="h-8 w-px bg-slate-800" />
            <div className="text-right">
              <span className="text-[11px] text-slate-400 block">Field Corrections</span>
              <span className="text-lg font-bold font-mono text-emerald-300">
                {improvement.summary.total_field_corrections}
              </span>
            </div>

            {onNavigateToImprovement && (
              <button
                onClick={onNavigateToImprovement}
                className="px-3.5 py-2 rounded-xl bg-indigo-600/30 hover:bg-indigo-600/50 border border-indigo-500/40 text-indigo-200 text-xs font-semibold flex items-center gap-1.5 transition-colors shrink-0 ml-2"
              >
                <span>Improvement Center</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
        </div>
      )}

      {/* 4. Middle Section: Automation Loss Funnel & Review Reasons */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Automation Loss Funnel */}
        <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-lg space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
              <Layers className="w-4 h-4 text-indigo-400" />
              Automation Loss & Resolution Funnel
            </h3>
            <span className="text-[11px] text-slate-400 font-mono">Descriptive Operational Flow</span>
          </div>

          {funnel && funnel.processed > 0 ? (
            <div className="space-y-3 pt-1">
              {/* Processed */}
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-300 font-medium">1. Ingested & Processed</span>
                  <span className="font-mono text-slate-200 font-bold">{funnel.processed} docs (100%)</span>
                </div>
                <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                  <div className="h-full bg-slate-400 rounded-full" style={{ width: "100%" }} />
                </div>
              </div>

              {/* Automatic vs Manual */}
              <div className="grid grid-cols-2 gap-3 pt-1">
                <div className="p-3 rounded-xl bg-slate-950/60 border border-emerald-500/20 space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-emerald-400 font-medium">2. Automatic</span>
                    <span className="font-mono text-emerald-400 font-bold">{funnel.automatic_rate}%</span>
                  </div>
                  <div className="text-sm font-mono font-bold text-slate-200">{funnel.automatic} docs</div>
                  <p className="text-[10px] text-slate-500">Straight-through automated pass</p>
                </div>

                <div className="p-3 rounded-xl bg-slate-950/60 border border-amber-500/20 space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-amber-400 font-medium">3. Manual Review</span>
                    <span className="font-mono text-amber-400 font-bold">{funnel.manual_review_rate}%</span>
                  </div>
                  <div className="text-sm font-mono font-bold text-slate-200">{funnel.manual_review} docs</div>
                  <p className="text-[10px] text-slate-500">Routed for human verification</p>
                </div>
              </div>

              {/* Resolution: Completed Reviews */}
              <div className="pt-2 border-t border-slate-800/80 space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-300 font-medium">4. Completed Manual Reviews</span>
                  <span className="font-mono text-slate-200 font-bold">
                    {funnel.completed_manual_review} resolved
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div className="p-2.5 rounded-lg bg-slate-950/40 border border-purple-500/20">
                    <div className="flex justify-between text-[11px]">
                      <span className="text-purple-300">With Corrections</span>
                      <span className="font-mono text-purple-300 font-bold">{funnel.correction_rate}%</span>
                    </div>
                    <div className="text-sm font-mono font-bold text-slate-200 mt-1">
                      {funnel.with_corrections} docs
                    </div>
                  </div>

                  <div className="p-2.5 rounded-lg bg-slate-950/40 border border-slate-800">
                    <div className="flex justify-between text-[11px]">
                      <span className="text-slate-400">Without Corrections</span>
                      <span className="font-mono text-slate-400 font-bold">{funnel.no_change_review_rate}%</span>
                    </div>
                    <div className="text-sm font-mono font-bold text-slate-200 mt-1">
                      {funnel.without_corrections} docs
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="py-8 text-center text-xs text-slate-500">
              No processed documents in the selected period.
            </div>
          )}
        </div>

        {/* Review Reasons Root Causes */}
        <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-lg space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-amber-400" />
              Dominant Causes of Manual Review
            </h3>
            {onNavigateToAnalytics && (
              <button
                onClick={onNavigateToAnalytics}
                className="text-xs text-indigo-400 hover:text-indigo-300 flex items-center gap-1 font-medium transition-colors"
              >
                Explainability Analytics <ArrowRight className="w-3 h-3" />
              </button>
            )}
          </div>

          {data?.review_reasons && data.review_reasons.length > 0 ? (
            <div className="space-y-3 pt-1">
              {data.review_reasons.map((r) => (
                <div key={r.code} className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <div className="flex items-center gap-2">
                      <span className="text-slate-300 capitalize">{r.description || r.code.replace(/_/g, " ")}</span>
                      <span className="text-[10px] px-1.5 py-0.2 rounded bg-slate-800 text-slate-400 border border-slate-700/60 uppercase">
                        {r.category}
                      </span>
                    </div>
                    <span className="font-mono text-slate-400">
                      {r.count} docs ({r.percentage}%)
                    </span>
                  </div>
                  <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-amber-500 rounded-full"
                      style={{ width: `${r.percentage}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="py-8 text-center text-xs text-slate-500">
              No manual review causes recorded in this period.
            </div>
          )}
        </div>
      </div>

      {/* 5. Quality & Production Feedback Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Quality & Confidence Indicators */}
        <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-lg space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
              Quality & Confidence Telemetry
            </h3>
            {onNavigateToResearch && (
              <button
                onClick={onNavigateToResearch}
                className="text-xs text-indigo-400 hover:text-indigo-300 flex items-center gap-1 font-medium transition-colors"
                title="Explore B0 Clean, B1 Degradation, and B2 Preprocessing experiments"
              >
                Research Mode <FlaskConical className="w-3 h-3" />
              </button>
            )}
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 pt-1">
            {/* Image Quality Score */}
            <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 space-y-1">
              <span className="text-[11px] text-slate-400">Image Quality</span>
              <div className="text-lg font-mono font-bold text-slate-200">
                {quality?.average_quality_score != null ? `${(quality.average_quality_score * 100).toFixed(1)}%` : "—"}
              </div>
              <p className="text-[10px] text-slate-500">Analyzer index (0..1)</p>
            </div>

            {/* Document Confidence */}
            <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 space-y-1">
              <span className="text-[11px] text-slate-400">Doc Confidence</span>
              <div className="text-lg font-mono font-bold text-indigo-300">
                {quality?.average_document_confidence != null ? `${(quality.average_document_confidence * 100).toFixed(1)}%` : "—"}
              </div>
              <p className="text-[10px] text-slate-500">Composite score (0..1)</p>
            </div>

            {/* KIE Field Confidence */}
            <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 space-y-1">
              <span className="text-[11px] text-slate-400">KIE Fields</span>
              <div className="text-lg font-mono font-bold text-slate-200">
                {quality?.average_kie_confidence != null ? `${(quality.average_kie_confidence * 100).toFixed(1)}%` : "—"}
              </div>
              <p className="text-[10px] text-slate-500">Extracted fields avg</p>
            </div>

            {/* OCR Token Confidence */}
            <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 space-y-1">
              <span className="text-[11px] text-slate-400">OCR Tokens</span>
              <div className="text-lg font-mono font-bold text-slate-200">
                {quality?.average_ocr_confidence != null ? `${(quality.average_ocr_confidence * 100).toFixed(1)}%` : "—"}
              </div>
              <p className="text-[10px] text-slate-500">Token confidence avg</p>
            </div>

            {/* Document Validation Pass Rate */}
            <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 space-y-1 sm:col-span-2">
              <span className="text-[11px] text-slate-400">Validation Pass Rate</span>
              <div className="text-lg font-mono font-bold text-emerald-400">
                {quality?.validation_pass_rate != null ? `${(quality.validation_pass_rate * 100).toFixed(1)}%` : "—"}
              </div>
              <p className="text-[10px] text-slate-500">
                {quality?.passed_validation_count || 0} of {quality?.total_documents_validated || 0} docs passed domain validation
              </p>
            </div>
          </div>

          <p className="text-[11px] text-slate-500 italic pt-1 border-t border-slate-800/80">
            Production signals only. Benchmark research metrics (B0/B1 CER, KIE F1) remain strictly isolated in Research Mode.
          </p>
        </div>

        {/* Production Feedback Summary */}
        <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-lg space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
              <Edit3 className="w-4 h-4 text-purple-400" />
              Production Feedback & Review Resolution
            </h3>
            {onNavigateToAnalytics && (
              <button
                onClick={onNavigateToAnalytics}
                className="text-xs text-indigo-400 hover:text-indigo-300 flex items-center gap-1 font-medium transition-colors"
              >
                Feedback Details <ArrowRight className="w-3 h-3" />
              </button>
            )}
          </div>

          <div className="grid grid-cols-2 gap-3 pt-1">
            <div className="p-3 rounded-xl bg-slate-950/60 border border-purple-500/20 space-y-1">
              <span className="text-[11px] text-slate-400">Correction Rate</span>
              <div className="text-xl font-mono font-bold text-purple-300">
                {feedback ? `${feedback.summary.correction_rate}%` : "—"}
              </div>
              <p className="text-[10px] text-slate-500">
                {feedback?.summary.documents_with_corrections || 0} of {feedback?.summary.completed_manual_review_documents || 0} reviews modified
              </p>
            </div>

            <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 space-y-1">
              <span className="text-[11px] text-slate-400">No-Change Reviews</span>
              <div className="text-xl font-mono font-bold text-slate-200">
                {feedback ? `${feedback.summary.no_change_review_rate}%` : "—"}
              </div>
              <p className="text-[10px] text-slate-500">Approved as-is without field changes</p>
            </div>
          </div>

          <div className="space-y-2 pt-2 border-t border-slate-800/80">
            <span className="text-xs font-medium text-slate-300">Frequently Corrected Fields</span>
            {feedback?.by_field && feedback.by_field.length > 0 ? (
              <div className="space-y-1.5">
                {feedback.by_field.slice(0, 4).map((f) => (
                  <div key={f.field} className="flex justify-between items-center text-xs">
                    <span className="font-mono text-slate-300">{f.field}</span>
                    <span className="font-mono text-purple-300">
                      {f.correction_count} edits ({f.share}%)
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-xs text-slate-500 py-2">No field corrections recorded in this period.</p>
            )}
          </div>

          <p className="text-[11px] text-slate-500 italic">
            Reflects operator edits; does not assert OCR or KIE algorithm failure causality.
          </p>
        </div>
      </div>

      {/* 6. Automation Trends */}
      {data?.trends && data.trends.length > 0 && (
        <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-lg space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-indigo-400" />
              Automation & Ingestion Timeline
            </h3>
            <span className="text-[11px] text-slate-400 font-mono">Daily Volume and Automation Share</span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-7 gap-2 pt-1 overflow-x-auto">
            {data.trends.map((t) => (
              <div key={t.date} className="p-2.5 rounded-xl bg-slate-950/60 border border-slate-800/80 space-y-1 text-center">
                <span className="text-[10px] text-slate-400 font-mono">{t.date.slice(5)}</span>
                <div className="text-sm font-mono font-bold text-slate-200">{t.processed_count} docs</div>
                <div className="text-[11px] font-mono text-emerald-400 font-semibold">
                  {(t.automation_rate * 100).toFixed(0)}% auto
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 7. Recent Ingestion Stream (Recent Documents Table) */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-base font-semibold text-slate-200 flex items-center gap-2">
            <Layers className="w-4 h-4 text-indigo-400" />
            Recent Documents Stream
          </h3>
          <button
            onClick={onViewAllDocuments}
            className="text-xs text-indigo-400 hover:text-indigo-300 flex items-center gap-1 font-medium transition-colors"
          >
            View All Documents <ArrowRight className="w-3 h-3" />
          </button>
        </div>

        <DocumentTable
          documents={data?.recent_documents || []}
          onInspect={onInspect}
          onReview={onReview}
          onRetry={async (id) => {
            await apiDocuments.retry(id);
            loadData(false);
          }}
          isLoading={isLoading}
        />
      </div>
    </div>
  );
};
