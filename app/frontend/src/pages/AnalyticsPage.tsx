import React, { useEffect, useState } from "react";
import {
  AnalyticsFilterParams,
  AnalyticsOverview,
  AutomationTrendsResponse,
  ConfidenceAnalyticsResponse,
  DecisionReasonsAnalyticsResponse,
  DocumentTypesResponse,
  FieldCorrectionMetric,
  ProductionFeedbackResponse,
  ReviewReasonsResponse,
  TimePeriod,
} from "../types";
import { apiAnalytics } from "../api/client";
import {
  BarChart3,
  TrendingUp,
  AlertTriangle,
  Clock,
  Layers,
  CheckCircle2,
  Edit3,
  Sliders,
  Calendar,
  FileText,
  Activity,
  ArrowUpRight,
  Filter,
  History,
  GitFork,
  CheckCheck,
} from "lucide-react";

export const AnalyticsPage: React.FC = () => {
  const [period, setPeriod] = useState<TimePeriod>("7d");
  const [customFrom, setCustomFrom] = useState<string>("");
  const [customTo, setCustomTo] = useState<string>("");

  const [data, setData] = useState<AnalyticsOverview | null>(null);
  const [corrections, setCorrections] = useState<FieldCorrectionMetric[]>([]);
  const [confidenceData, setConfidenceData] = useState<ConfidenceAnalyticsResponse | null>(null);
  const [reviewReasons, setReviewReasons] = useState<ReviewReasonsResponse | null>(null);
  const [docTypes, setDocTypes] = useState<DocumentTypesResponse | null>(null);
  const [trends, setTrends] = useState<AutomationTrendsResponse | null>(null);
  const [decisionReasons, setDecisionReasons] = useState<DecisionReasonsAnalyticsResponse | null>(null);
  const [reasonsView, setReasonsView] = useState<"reasons" | "categories" | "fields">("reasons");
  const [feedbackData, setFeedbackData] = useState<ProductionFeedbackResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const loadData = async () => {
    setIsLoading(true);
    try {
      const params: AnalyticsFilterParams = { period };
      if (period === "custom") {
        if (customFrom) params.from = new Date(customFrom).toISOString();
        if (customTo) params.to = new Date(customTo).toISOString();
      }

      const [overview, corrRes, confRes, reasonsRes, typesRes, trendsRes, decReasonsRes, feedbackRes] =
        await Promise.all([
          apiAnalytics.getOverview(params),
          apiAnalytics.getCorrections(params).catch(() => ({ total_fields: 0, items: [] })),
          apiAnalytics.getConfidence(params).catch(() => null),
          apiAnalytics.getReviewReasons(params).catch(() => null),
          apiAnalytics.getDocumentTypes(params).catch(() => null),
          apiAnalytics.getTrends(params).catch(() => null),
          apiAnalytics.getDecisionReasons(params).catch(() => null),
          apiAnalytics.getFeedback(params).catch(() => null),
        ]);

      setData(overview);
      // GAP-01 Canonical resolution: always prefer items, fallback to fields
      const fieldItems: FieldCorrectionMetric[] = (corrRes as any).items || (corrRes as any).fields || [];
      setCorrections(fieldItems);
      setConfidenceData(confRes);
      setReviewReasons(reasonsRes);
      setDocTypes(typesRes);
      setTrends(trendsRes);
      setDecisionReasons(decReasonsRes);
      setFeedbackData(feedbackRes);
    } catch (err) {
      console.error("Failed to load analytics", err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (period !== "custom" || (customFrom && customTo)) {
      loadData();
    }
  }, [period]);

  const handleApplyCustom = () => {
    if (customFrom && customTo) {
      loadData();
    }
  };

  const kpis = data?.kpis;

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* Top Filter Bar */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 p-4 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-lg">
        <div className="flex items-center gap-2">
          <Calendar className="w-5 h-5 text-indigo-400" />
          <span className="text-sm font-semibold text-slate-200">Analytics Time Range</span>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {(["today", "7d", "30d", "90d", "all", "custom"] as TimePeriod[]).map((p) => (
            <button
              key={p}
              onClick={() => setPeriod(p)}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold uppercase tracking-wider transition-colors ${
                period === p
                  ? "bg-indigo-600 text-white shadow-md shadow-indigo-500/20"
                  : "bg-slate-800/80 text-slate-400 hover:text-slate-200 hover:bg-slate-700/80"
              }`}
            >
              {p === "all" ? "All Time" : p}
            </button>
          ))}

          {period === "custom" && (
            <div className="flex items-center gap-2 mt-2 sm:mt-0 bg-slate-950 p-1.5 rounded-lg border border-slate-800">
              <input
                type="date"
                value={customFrom}
                onChange={(e) => setCustomFrom(e.target.value)}
                className="bg-transparent text-xs text-slate-200 focus:outline-none"
              />
              <span className="text-xs text-slate-500">to</span>
              <input
                type="date"
                value={customTo}
                onChange={(e) => setCustomTo(e.target.value)}
                className="bg-transparent text-xs text-slate-200 focus:outline-none"
              />
              <button
                onClick={handleApplyCustom}
                className="px-2.5 py-1 bg-indigo-600 hover:bg-indigo-500 text-white rounded text-xs font-medium"
              >
                Apply
              </button>
            </div>
          )}
        </div>
      </div>

      {isLoading && !data ? (
        <div className="flex items-center justify-center p-24 text-slate-400">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-500"></div>
        </div>
      ) : (
        <>
          {/* Executive KPI Header Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-6 gap-4">
            {/* Automation Rate */}
            <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-lg">
              <span className="text-xs text-slate-400 font-medium">Automation Rate</span>
              <p className="text-2xl font-bold font-mono text-emerald-400 mt-2">
                {kpis ? `${kpis.automation_rate}%` : "0%"}
              </p>
              <p className="text-[11px] text-slate-500 mt-1">
                {kpis?.automatic_count || 0} straight-through
              </p>
            </div>

            {/* Manual Review Rate */}
            <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-lg">
              <span className="text-xs text-slate-400 font-medium">Manual Review Rate</span>
              <p className="text-2xl font-bold font-mono text-amber-400 mt-2">
                {kpis?.manual_review_rate !== undefined ? `${kpis.manual_review_rate}%` : "0%"}
              </p>
              <p className="text-[11px] text-slate-500 mt-1">
                {kpis?.completed_manual_count || 0} done / {kpis?.manual_review_count || 0} queue
              </p>
            </div>

            {/* Failure Rate */}
            <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-lg">
              <span className="text-xs text-slate-400 font-medium">Failure Rate</span>
              <p className="text-2xl font-bold font-mono text-rose-400 mt-2">
                {kpis?.failure_rate !== undefined ? `${kpis.failure_rate}%` : "0%"}
              </p>
              <p className="text-[11px] text-slate-500 mt-1">{kpis?.error_count || 0} failed jobs</p>
            </div>

            {/* Total Ingestion */}
            <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-lg">
              <span className="text-xs text-slate-400 font-medium">Total Ingestion</span>
              <p className="text-2xl font-bold font-mono text-indigo-400 mt-2">
                {kpis?.total_documents || 0}
              </p>
              <p className="text-[11px] text-slate-500 mt-1">{kpis?.processed_count || 0} processed</p>
            </div>

            {/* Total Field Corrections */}
            <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-lg">
              <span className="text-xs text-slate-400 font-medium">Field Corrections</span>
              <p className="text-2xl font-bold font-mono text-purple-400 mt-2">
                {kpis?.corrections_total || 0}
              </p>
              <p className="text-[11px] text-slate-500 mt-1">Human adjustments</p>
            </div>

            {/* Avg Review Duration */}
            <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-lg">
              <span className="text-xs text-slate-400 font-medium">Avg Review Time</span>
              <p className="text-2xl font-bold font-mono text-slate-100 mt-2">
                {kpis?.average_review_time_ms ? `${(kpis.average_review_time_ms / 1000).toFixed(1)}s` : "—"}
              </p>
              <p className="text-[11px] text-slate-500 mt-1">Machine: {kpis?.average_processing_time_ms ? `${(kpis.average_processing_time_ms / 1000).toFixed(1)}s` : "—"}</p>
            </div>
          </div>

          {/* Section: Automation Rate by Document Type */}
          <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-lg space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                <FileText className="w-4 h-4 text-indigo-400" />
                Automation Rate by Document Type
              </h3>
              <span className="text-xs text-slate-400">
                Performance & straight-through readiness across ingested document classes
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400">
                    <th className="py-2.5 px-3 uppercase tracking-wider font-semibold">Document Type</th>
                    <th className="py-2.5 px-3 uppercase tracking-wider font-semibold text-right">Total</th>
                    <th className="py-2.5 px-3 uppercase tracking-wider font-semibold text-right">Processed</th>
                    <th className="py-2.5 px-3 uppercase tracking-wider font-semibold text-right">Automation Rate</th>
                    <th className="py-2.5 px-3 uppercase tracking-wider font-semibold">Automation Distribution</th>
                    <th className="py-2.5 px-3 uppercase tracking-wider font-semibold text-right">Manual Rate</th>
                    <th className="py-2.5 px-3 uppercase tracking-wider font-semibold text-right">Avg Confidence</th>
                    <th className="py-2.5 px-3 uppercase tracking-wider font-semibold text-right">Avg Proc Time</th>
                    <th className="py-2.5 px-3 uppercase tracking-wider font-semibold text-right">Avg Review Time</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {docTypes && docTypes.items.length > 0 ? (
                    docTypes.items.map((dt) => {
                      return (
                        <tr key={dt.document_type} className="hover:bg-slate-800/30 transition-colors">
                          <td className="py-3 px-3 font-mono font-semibold text-slate-200 uppercase">
                            {dt.document_type}
                          </td>
                          <td className="py-3 px-3 font-mono text-slate-300 text-right">{dt.total_documents}</td>
                          <td className="py-3 px-3 font-mono text-slate-300 text-right">{dt.processed_documents}</td>
                          <td className="py-3 px-3 font-mono font-bold text-right">
                            <span
                              className={
                                dt.automation_rate >= 80
                                  ? "text-emerald-400"
                                  : dt.automation_rate >= 50
                                  ? "text-amber-400"
                                  : "text-rose-400"
                              }
                            >
                              {dt.automation_rate}%
                            </span>
                          </td>
                          <td className="py-3 px-3">
                            <div className="w-36 bg-slate-950 h-2 rounded-full overflow-hidden flex">
                              <div
                                className="bg-emerald-500 h-full"
                                style={{ width: `${dt.automation_rate}%` }}
                                title={`Automatic: ${dt.automation_rate}%`}
                              />
                              <div
                                className="bg-amber-500 h-full"
                                style={{ width: `${dt.manual_review_rate}%` }}
                                title={`Manual Review: ${dt.manual_review_rate}%`}
                              />
                              <div
                                className="bg-rose-500 h-full"
                                style={{ width: `${dt.failure_rate}%` }}
                                title={`Failure: ${dt.failure_rate}%`}
                              />
                            </div>
                          </td>
                          <td className="py-3 px-3 font-mono text-amber-400 text-right">
                            {dt.manual_review_rate}%
                          </td>
                          <td className="py-3 px-3 font-mono text-slate-300 text-right">
                            {dt.average_confidence}%
                          </td>
                          <td className="py-3 px-3 font-mono text-slate-400 text-right">
                            {dt.average_processing_time_ms ? `${(dt.average_processing_time_ms / 1000).toFixed(2)}s` : "—"}
                          </td>
                          <td className="py-3 px-3 font-mono text-slate-400 text-right">
                            {dt.average_review_time_ms ? `${(dt.average_review_time_ms / 1000).toFixed(1)}s` : "—"}
                          </td>
                        </tr>
                      );
                    })
                  ) : (
                    <tr>
                      <td colSpan={9} className="py-8 text-center text-slate-500">
                        No document types recorded in selected period
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* Section: Daily Automation Trends */}
          <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-lg space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-emerald-400" />
                Daily Automation Trends & Volume
              </h3>
              <span className="text-xs text-slate-400">
                Tracking day-by-day throughput and straight-through automation rate
              </span>
            </div>

            {trends && trends.items.length > 0 ? (
              <div className="space-y-4 pt-2">
                {/* Visual Bar Timeline */}
                <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-7 gap-3">
                  {trends.items.map((item) => {
                    const hasTraffic = item.processed_count > 0;
                    return (
                      <div
                        key={item.date}
                        className="bg-slate-950/60 p-3 rounded-xl border border-slate-800/80 flex flex-col justify-between"
                      >
                        <div className="flex items-center justify-between text-xs text-slate-400 mb-2">
                          <span className="font-mono">{item.date.slice(5)}</span>
                          <span className="font-semibold text-slate-300">{item.processed_count} docs</span>
                        </div>

                        {/* Bar representation */}
                        <div className="space-y-1.5 my-2">
                          <div className="flex justify-between text-[11px]">
                            <span className="text-slate-500">Auto Rate:</span>
                            <span
                              className={`font-mono font-bold ${
                                item.automation_rate >= 80
                                  ? "text-emerald-400"
                                  : item.automation_rate > 0
                                  ? "text-amber-400"
                                  : "text-slate-500"
                              }`}
                            >
                              {item.automation_rate}%
                            </span>
                          </div>
                          <div className="h-2 w-full bg-slate-900 rounded-full overflow-hidden flex">
                            <div
                              className="h-full bg-emerald-500"
                              style={{ width: `${item.automation_rate}%` }}
                            />
                            <div
                              className="h-full bg-amber-500"
                              style={{
                                width: `${
                                  item.processed_count > 0
                                    ? ((item.manual_review_count / item.processed_count) * 100).toFixed(1)
                                    : 0
                                }%`,
                              }}
                            />
                          </div>
                        </div>

                        <div className="text-[10px] text-slate-500 flex justify-between pt-1 border-t border-slate-900">
                          <span>Auto: {item.automatic_count}</span>
                          <span>Rev: {item.manual_review_count}</span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : (
              <p className="text-xs text-slate-500 text-center py-8">
                No timeline data available for selected period
              </p>
            )}
          </div>

          {/* Grid: Root Causes & Confidence Calibration */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Reasons for Manual Review */}
            <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-lg space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-amber-400" />
                  Root Causes for Manual Review
                </h3>
                {decisionReasons && (
                  <div className="flex bg-slate-950 p-0.5 rounded-lg border border-slate-800 text-[10px]">
                    <button
                      onClick={() => setReasonsView("reasons")}
                      className={`px-2 py-0.5 rounded ${
                        reasonsView === "reasons"
                          ? "bg-amber-600/30 text-amber-300 font-semibold"
                          : "text-slate-400 hover:text-slate-200"
                      }`}
                    >
                      Reasons
                    </button>
                    <button
                      onClick={() => setReasonsView("categories")}
                      className={`px-2 py-0.5 rounded ${
                        reasonsView === "categories"
                          ? "bg-amber-600/30 text-amber-300 font-semibold"
                          : "text-slate-400 hover:text-slate-200"
                      }`}
                    >
                      Categories
                    </button>
                    <button
                      onClick={() => setReasonsView("fields")}
                      className={`px-2 py-0.5 rounded ${
                        reasonsView === "fields"
                          ? "bg-amber-600/30 text-amber-300 font-semibold"
                          : "text-slate-400 hover:text-slate-200"
                      }`}
                    >
                      Fields
                    </button>
                  </div>
                )}
              </div>

              <div className="space-y-4 pt-2">
                {reasonsView === "categories" && decisionReasons?.by_category && decisionReasons.by_category.length > 0 ? (
                  decisionReasons.by_category.map((cat) => (
                    <div key={cat.category} className="space-y-1.5">
                      <div className="flex justify-between text-xs">
                        <span className="text-slate-300 font-medium capitalize font-mono">
                          {cat.category}
                        </span>
                        <span className="font-mono text-slate-400">
                          {cat.count} conditions ({cat.percentage}%)
                        </span>
                      </div>
                      <div className="h-2 w-full bg-slate-950 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-indigo-500 rounded-full"
                          style={{ width: `${cat.percentage}%` }}
                        />
                      </div>
                    </div>
                  ))
                ) : reasonsView === "fields" && decisionReasons?.by_field && decisionReasons.by_field.length > 0 ? (
                  decisionReasons.by_field.map((f) => (
                    <div key={f.field} className="space-y-1.5">
                      <div className="flex justify-between text-xs">
                        <span className="text-slate-300 font-medium font-mono">
                          field: {f.field}
                        </span>
                        <span className="font-mono text-slate-400">
                          {f.count} failures ({f.percentage}%)
                        </span>
                      </div>
                      <div className="h-2 w-full bg-slate-950 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-rose-500 rounded-full"
                          style={{ width: `${f.percentage}%` }}
                        />
                      </div>
                    </div>
                  ))
                ) : reviewReasons && reviewReasons.reasons.length > 0 ? (
                  reviewReasons.reasons.map((r) => (
                    <div key={r.reason} className="space-y-1.5">
                      <div className="flex justify-between text-xs">
                        <span className="text-slate-300 font-medium uppercase font-mono">
                          {r.reason.replace(/_/g, " ")}
                        </span>
                        <span className="font-mono text-slate-400">
                          {r.count} docs ({r.share}%)
                        </span>
                      </div>
                      <div className="h-2 w-full bg-slate-950 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-amber-500 rounded-full"
                          style={{ width: `${r.share}%` }}
                        />
                      </div>
                    </div>
                  ))
                ) : data?.review_reasons && data.review_reasons.length > 0 ? (
                  data.review_reasons.map((r) => (
                    <div key={r.category} className="space-y-1.5">
                      <div className="flex justify-between text-xs">
                        <span className="text-slate-300 capitalize">{r.category.replace(/_/g, " ")}</span>
                        <span className="font-mono text-slate-400">
                          {r.count} docs ({r.percentage}%)
                        </span>
                      </div>
                      <div className="h-2 w-full bg-slate-950 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-amber-500 rounded-full"
                          style={{ width: `${r.percentage}%` }}
                        />
                      </div>
                    </div>
                  ))
                ) : (
                  <p className="text-xs text-slate-500 text-center py-8">
                    No manual reviews recorded in this period
                  </p>
                )}
              </div>
            </div>

            {/* Confidence Distribution & Calibration */}
            <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-lg space-y-4">
              <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                <Sliders className="w-4 h-4 text-indigo-400" />
                Confidence Distribution & HITL Calibration
              </h3>

              <div className="space-y-3 pt-2">
                {confidenceData && confidenceData.distribution.length > 0 ? (
                  confidenceData.distribution.map((bucket) => (
                    <div key={bucket.bucket} className="space-y-1">
                      <div className="flex justify-between text-xs">
                        <span className="font-mono text-slate-300 font-semibold">{bucket.bucket}</span>
                        <span className="text-slate-400 font-mono text-[11px]">
                          {bucket.count} docs ({bucket.percentage}%) · {bucket.manual_review_count} reviews ({bucket.review_rate ?? 0}%)
                        </span>
                      </div>
                      <div className="h-2 w-full bg-slate-950 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full ${
                            bucket.bucket.startsWith("0.9")
                              ? "bg-emerald-500"
                              : bucket.bucket.startsWith("0.8")
                              ? "bg-teal-500"
                              : bucket.bucket.startsWith("0.7")
                              ? "bg-amber-500"
                              : "bg-rose-500"
                          }`}
                          style={{ width: `${Math.max(bucket.percentage, bucket.count > 0 ? 3 : 0)}%` }}
                        />
                      </div>
                    </div>
                  ))
                ) : (
                  <p className="text-xs text-slate-500 text-center py-8">
                    No confidence distribution data available
                  </p>
                )}
              </div>
            </div>
          </div>

          {/* Section: Field Correction Frequency & Vulnerability Table (GAP-01 Canonical) */}
          <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-lg space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                <Edit3 className="w-4 h-4 text-emerald-400" />
                Field-Level Correction Frequency & Vulnerability
              </h3>
              <span className="text-xs text-slate-400">
                Identifies fields frequently misrecognized by OCR/KIE
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400">
                    <th className="py-2.5 px-3 uppercase tracking-wider font-semibold">Field Name</th>
                    <th className="py-2.5 px-3 uppercase tracking-wider font-semibold text-right">Total Extracted</th>
                    <th className="py-2.5 px-3 uppercase tracking-wider font-semibold text-right">Corrected</th>
                    <th className="py-2.5 px-3 uppercase tracking-wider font-semibold text-right">Correction Rate</th>
                    <th className="py-2.5 px-3 uppercase tracking-wider font-semibold">Reliability</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {corrections.length > 0 ? (
                    corrections.map((row) => {
                      const fieldName = row.field || row.field_name || "unknown";
                      const rate = row.correction_rate;
                      const isHighRisk = rate >= 30;
                      return (
                        <tr key={fieldName} className="hover:bg-slate-800/30 transition-colors">
                          <td className="py-2.5 px-3 font-mono font-semibold text-slate-200 uppercase">
                            {fieldName}
                          </td>
                          <td className="py-2.5 px-3 font-mono text-slate-300 text-right">
                            {row.total_occurrences}
                          </td>
                          <td className="py-2.5 px-3 font-mono text-slate-300 text-right">
                            {row.corrected_count}
                          </td>
                          <td className="py-2.5 px-3 font-mono font-semibold text-right">
                            <span
                              className={
                                isHighRisk
                                  ? "text-rose-400"
                                  : rate > 10
                                  ? "text-amber-400"
                                  : "text-emerald-400"
                              }
                            >
                              {rate}%
                            </span>
                          </td>
                          <td className="py-2.5 px-3">
                            <div className="w-32 bg-slate-950 h-1.5 rounded-full overflow-hidden">
                              <div
                                className={`h-full rounded-full ${
                                  isHighRisk
                                    ? "bg-rose-500"
                                    : rate > 10
                                    ? "bg-amber-500"
                                    : "bg-emerald-500"
                                }`}
                                style={{ width: `${Math.min(100, Math.max(5, 100 - rate))}%` }}
                              />
                            </div>
                          </td>
                        </tr>
                      );
                    })
                  ) : (
                    <tr>
                      <td colSpan={5} className="py-8 text-center text-slate-500">
                        No field corrections recorded yet in selected period.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* Section: MVP-7 Production Feedback & Continuous Improvement */}
          <div className="space-y-6 pt-4 border-t border-slate-800/80">
            {/* Section Title */}
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-base font-semibold text-slate-100 flex items-center gap-2">
                  <History className="w-5 h-5 text-emerald-400" />
                  Production Feedback & Continuous Improvement
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Factual operational evidence and human-in-the-loop review outcomes
                </p>
              </div>
              <span className="text-[11px] font-mono px-2.5 py-1 rounded-full bg-slate-900 border border-slate-800 text-slate-400">
                Closed-Loop Feedback
              </span>
            </div>

            {/* 1. Feedback Summary KPIs */}
            {feedbackData && (
              <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1">
                  <span className="text-xs text-slate-400">Reviewed Docs</span>
                  <p className="text-xl font-bold font-mono text-slate-100">
                    {feedbackData.summary.completed_manual_review_documents}
                  </p>
                </div>
                <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1">
                  <span className="text-xs text-slate-400">Field Corrections</span>
                  <p className="text-xl font-bold font-mono text-emerald-400">
                    {feedbackData.summary.total_correction_events}
                  </p>
                </div>
                <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1">
                  <span className="text-xs text-slate-400">Correction Rate</span>
                  <p className="text-xl font-bold font-mono text-amber-400">
                    {feedbackData.summary.correction_rate}%
                  </p>
                </div>
                <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1">
                  <span className="text-xs text-slate-400">No-Change Reviews</span>
                  <p className="text-xl font-bold font-mono text-teal-400">
                    {feedbackData.summary.no_change_review_rate}%
                  </p>
                </div>
                <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1">
                  <span className="text-xs text-slate-400">Avg Corr / Doc</span>
                  <p className="text-xl font-bold font-mono text-indigo-400">
                    {feedbackData.summary.avg_corrections_per_corrected_document}
                  </p>
                </div>
              </div>
            )}

            {/* 2. Automation Loss Funnel */}
            {feedbackData && (
              <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-lg space-y-4">
                <div className="flex items-center justify-between">
                  <h4 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                    <GitFork className="w-4 h-4 text-indigo-400" />
                    Automation Loss & Review Resolution Funnel
                  </h4>
                  <span className="text-xs text-slate-400 font-mono">
                    Processed: {feedbackData.funnel.processed} docs
                  </span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-5 gap-3 pt-2">
                  <div className="p-3.5 rounded-xl bg-slate-950/80 border border-slate-800 space-y-1">
                    <span className="text-[11px] text-slate-400 font-semibold uppercase tracking-wider">
                      1. Processed
                    </span>
                    <p className="text-lg font-bold font-mono text-slate-100">
                      {feedbackData.funnel.processed}
                    </p>
                    <span className="text-[10px] text-slate-500 font-mono">Total intake</span>
                  </div>

                  <div className="p-3.5 rounded-xl bg-slate-950/80 border border-emerald-800/40 space-y-1">
                    <span className="text-[11px] text-emerald-400 font-semibold uppercase tracking-wider">
                      2. Automatic
                    </span>
                    <p className="text-lg font-bold font-mono text-emerald-300">
                      {feedbackData.funnel.automatic}
                    </p>
                    <span className="text-[10px] text-emerald-500 font-mono">
                      {feedbackData.funnel.automatic_rate}% rate
                    </span>
                  </div>

                  <div className="p-3.5 rounded-xl bg-slate-950/80 border border-amber-800/40 space-y-1">
                    <span className="text-[11px] text-amber-400 font-semibold uppercase tracking-wider">
                      3. Manual Review
                    </span>
                    <p className="text-lg font-bold font-mono text-amber-300">
                      {feedbackData.funnel.manual_review}
                    </p>
                    <span className="text-[10px] text-amber-500 font-mono">
                      {feedbackData.funnel.manual_review_rate}% rate
                    </span>
                  </div>

                  <div className="p-3.5 rounded-xl bg-slate-950/80 border border-teal-800/40 space-y-1">
                    <span className="text-[11px] text-teal-400 font-semibold uppercase tracking-wider">
                      4. Completed Review
                    </span>
                    <p className="text-lg font-bold font-mono text-teal-300">
                      {feedbackData.funnel.completed_manual_review}
                    </p>
                    <span className="text-[10px] text-teal-500 font-mono">Operator validated</span>
                  </div>

                  <div className="p-3.5 rounded-xl bg-slate-950/80 border border-rose-800/40 space-y-1">
                    <span className="text-[11px] text-rose-400 font-semibold uppercase tracking-wider">
                      5. With Corrections
                    </span>
                    <p className="text-lg font-bold font-mono text-rose-300">
                      {feedbackData.funnel.with_corrections}
                    </p>
                    <span className="text-[10px] text-rose-500 font-mono">
                      {feedbackData.funnel.correction_rate}% of reviewed
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* 3. Grid: Corrections by Field & Corrections by Document Type */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Corrections by Field */}
              <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-lg space-y-4">
                <h4 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                  <Edit3 className="w-4 h-4 text-emerald-400" />
                  Corrections by Field
                </h4>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead>
                      <tr className="border-b border-slate-800 text-slate-400">
                        <th className="py-2 px-2 uppercase font-semibold">Field</th>
                        <th className="py-2 px-2 uppercase font-semibold text-right">Corrections</th>
                        <th className="py-2 px-2 uppercase font-semibold text-right">Documents</th>
                        <th className="py-2 px-2 uppercase font-semibold text-right">Share</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60">
                      {feedbackData && feedbackData.by_field.length > 0 ? (
                        feedbackData.by_field.map((f) => (
                          <tr key={f.field} className="hover:bg-slate-800/30">
                            <td className="py-2 px-2 font-mono font-semibold text-slate-200 uppercase">
                              {f.field}
                            </td>
                            <td className="py-2 px-2 font-mono text-slate-300 text-right">
                              {f.correction_count}
                            </td>
                            <td className="py-2 px-2 font-mono text-slate-400 text-right">
                              {f.affected_documents}
                            </td>
                            <td className="py-2 px-2 font-mono text-emerald-400 font-semibold text-right">
                              {f.share}%
                            </td>
                          </tr>
                        ))
                      ) : (
                        <tr>
                          <td colSpan={4} className="py-6 text-center text-slate-500">
                            No field corrections in selected period
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Corrections by Document Type */}
              <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-lg space-y-4">
                <h4 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                  <FileText className="w-4 h-4 text-indigo-400" />
                  Corrections by Document Type
                </h4>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead>
                      <tr className="border-b border-slate-800 text-slate-400">
                        <th className="py-2 px-2 uppercase font-semibold">Type</th>
                        <th className="py-2 px-2 uppercase font-semibold text-right">Reviewed</th>
                        <th className="py-2 px-2 uppercase font-semibold text-right">Corrected</th>
                        <th className="py-2 px-2 uppercase font-semibold text-right">Events</th>
                        <th className="py-2 px-2 uppercase font-semibold text-right">Rate</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60">
                      {feedbackData && feedbackData.by_document_type.length > 0 ? (
                        feedbackData.by_document_type.map((dt) => (
                          <tr key={dt.document_type} className="hover:bg-slate-800/30">
                            <td className="py-2 px-2 font-mono font-semibold text-slate-200 uppercase">
                              {dt.document_type}
                            </td>
                            <td className="py-2 px-2 font-mono text-slate-300 text-right">
                              {dt.reviewed_documents}
                            </td>
                            <td className="py-2 px-2 font-mono text-slate-400 text-right">
                              {dt.documents_with_corrections}
                            </td>
                            <td className="py-2 px-2 font-mono text-slate-400 text-right">
                              {dt.correction_events}
                            </td>
                            <td className="py-2 px-2 font-mono text-amber-400 font-semibold text-right">
                              {dt.correction_rate}%
                            </td>
                          </tr>
                        ))
                      ) : (
                        <tr>
                          <td colSpan={5} className="py-6 text-center text-slate-500">
                            No reviewed documents in selected period
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            {/* 4. Grid: Corrections by Review Reason & Reason x Field Matrix */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Corrections by Review Reason */}
              <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-lg space-y-4">
                <h4 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-amber-400" />
                  Corrections by Review Reason
                </h4>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead>
                      <tr className="border-b border-slate-800 text-slate-400">
                        <th className="py-2 px-2 uppercase font-semibold">Reason</th>
                        <th className="py-2 px-2 uppercase font-semibold text-right">Reviewed</th>
                        <th className="py-2 px-2 uppercase font-semibold text-right">Corrected</th>
                        <th className="py-2 px-2 uppercase font-semibold text-right">Events</th>
                        <th className="py-2 px-2 uppercase font-semibold text-right">Rate</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60">
                      {feedbackData && feedbackData.by_reason.length > 0 ? (
                        feedbackData.by_reason.map((r) => (
                          <tr key={r.reason} className="hover:bg-slate-800/30">
                            <td className="py-2 px-2 font-mono font-semibold text-slate-200 uppercase">
                              {r.reason.replace(/_/g, " ")}
                            </td>
                            <td className="py-2 px-2 font-mono text-slate-300 text-right">
                              {r.reviewed_documents}
                            </td>
                            <td className="py-2 px-2 font-mono text-slate-400 text-right">
                              {r.documents_with_corrections}
                            </td>
                            <td className="py-2 px-2 font-mono text-slate-400 text-right">
                              {r.correction_events}
                            </td>
                            <td className="py-2 px-2 font-mono text-amber-400 font-semibold text-right">
                              {r.correction_rate}%
                            </td>
                          </tr>
                        ))
                      ) : (
                        <tr>
                          <td colSpan={5} className="py-6 text-center text-slate-500">
                            No review reasons recorded in selected period
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Review Reason x Field Matrix */}
              <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-lg space-y-4">
                <h4 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                  <Layers className="w-4 h-4 text-indigo-400" />
                  Review Reason × Field Co-occurrence
                </h4>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead>
                      <tr className="border-b border-slate-800 text-slate-400">
                        <th className="py-2 px-2 uppercase font-semibold">Review Reason</th>
                        <th className="py-2 px-2 uppercase font-semibold">Field</th>
                        <th className="py-2 px-2 uppercase font-semibold text-right">Corrections</th>
                        <th className="py-2 px-2 uppercase font-semibold text-right">Documents</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60">
                      {feedbackData && feedbackData.by_reason_field.length > 0 ? (
                        feedbackData.by_reason_field.map((rf, idx) => (
                          <tr key={idx} className="hover:bg-slate-800/30">
                            <td className="py-2 px-2 font-mono font-medium text-slate-300 uppercase">
                              {rf.review_reason.replace(/_/g, " ")}
                            </td>
                            <td className="py-2 px-2 font-mono font-semibold text-emerald-400 uppercase">
                              {rf.field}
                            </td>
                            <td className="py-2 px-2 font-mono text-slate-300 text-right">
                              {rf.correction_count}
                            </td>
                            <td className="py-2 px-2 font-mono text-slate-400 text-right">
                              {rf.affected_documents}
                            </td>
                          </tr>
                        ))
                      ) : (
                        <tr>
                          <td colSpan={4} className="py-6 text-center text-slate-500">
                            No reason-field correlations in selected period
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

