import React, { useEffect, useState } from "react";
import {
  DegradationCondition,
  DegradationExplorerItem,
  DegradationExplorerResponse,
} from "../types";
import { apiResearch } from "../api/client";
import {
  Sliders,
  ShieldCheck,
  Activity,
  AlertTriangle,
  FileText,
  Layers,
  Database,
  CheckCircle2,
  TrendingDown,
} from "lucide-react";

type MetricKey = "cer" | "ned" | "macro_f1" | "entity_hmean";

interface MetricConfig {
  key: MetricKey;
  label: string;
  fullName: string;
  lowerIsBetter: boolean;
  color: string;
  format: (v: number) => string;
}

const METRICS: MetricConfig[] = [
  {
    key: "cer",
    label: "CER",
    fullName: "Normalized Character Error Rate",
    lowerIsBetter: true,
    color: "#f43f5e",
    format: (v) => v.toFixed(4),
  },
  {
    key: "ned",
    label: "NED",
    fullName: "Normalized Edit Distance",
    lowerIsBetter: false,
    color: "#06b6d4",
    format: (v) => v.toFixed(4),
  },
  {
    key: "macro_f1",
    label: "Macro F1",
    fullName: "KIE Field Extraction F1",
    lowerIsBetter: false,
    color: "#8b5cf6",
    format: (v) => v.toFixed(4),
  },
  {
    key: "entity_hmean",
    label: "Entity H-Mean",
    fullName: "SROIE Strict Official Entity H-Mean",
    lowerIsBetter: false,
    color: "#10b981",
    format: (v) => v.toFixed(4),
  },
];

interface DegradationExplorerPageProps {
  initialDegradation?: string;
}

export const DegradationExplorerPage: React.FC<DegradationExplorerPageProps> = ({
  initialDegradation,
}) => {
  const [data, setData] = useState<DegradationExplorerResponse | null>(null);
  const [selectedDegId, setSelectedDegId] = useState<string>("b1_gaussian_blur");
  const [selectedMetric, setSelectedMetric] = useState<MetricKey>("cer");
  const [selectedConditionId, setSelectedConditionId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadDegradations = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await apiResearch.getDegradations();
      setData(res);

      const target = initialDegradation || new URLSearchParams(window.location.search).get("degradation");
      if (target && res.items.length > 0) {
        const match = res.items.find(
          (item) =>
            item.code.toUpperCase() === target.toUpperCase() ||
            item.id.toLowerCase() === target.toLowerCase() ||
            item.name.toLowerCase().includes(target.toLowerCase())
        );
        if (match) {
          setSelectedDegId(match.id);
        } else if (!res.items.some((i) => i.id === selectedDegId)) {
          setSelectedDegId(res.items[0].id);
        }
      } else if (res.items.length > 0 && !res.items.some((i) => i.id === selectedDegId)) {
        setSelectedDegId(res.items[0].id);
      }
    } catch (err: any) {
      console.error("Failed to load degradation explorer data", err);
      setError("Unable to load degradation experiments. Please verify research artifacts are accessible.");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadDegradations();
  }, []);

  useEffect(() => {
    if (data && initialDegradation) {
      const match = data.items.find(
        (item) =>
          item.code.toUpperCase() === initialDegradation.toUpperCase() ||
          item.id.toLowerCase() === initialDegradation.toLowerCase() ||
          item.name.toLowerCase().includes(initialDegradation.toLowerCase())
      );
      if (match) {
        setSelectedDegId(match.id);
      }
    }
  }, [initialDegradation, data]);

  const currentItem: DegradationExplorerItem | undefined =
    data?.items.find((item) => item.id === selectedDegId) || data?.items[0];

  const conditions = currentItem?.conditions || [];
  const selectedCondition: DegradationCondition | undefined =
    conditions.find((c) => c.condition_id === selectedConditionId) || conditions[0];

  const currentMetricConfig = METRICS.find((m) => m.key === selectedMetric) || METRICS[0];

  // SVG Chart Dimensions & Scale calculations
  const chartWidth = 640;
  const chartHeight = 220;
  const padding = { top: 25, right: 35, bottom: 45, left: 55 };
  const innerWidth = chartWidth - padding.left - padding.right;
  const innerHeight = chartHeight - padding.top - padding.bottom;

  // Find min/max for the current metric across conditions (with safe margin)
  const metricValues = conditions.map((c) => c[selectedMetric] as number);
  const minVal = Math.max(0, Math.min(...metricValues) * 0.85);
  const maxVal = Math.min(1.0, Math.max(...metricValues) * 1.15 || 1.0);
  const valRange = maxVal - minVal || 1.0;

  const points = conditions.map((c, idx) => {
    const x = padding.left + (idx / Math.max(1, conditions.length - 1)) * innerWidth;
    const val = c[selectedMetric] as number;
    const y = padding.top + (1 - (val - minVal) / valRange) * innerHeight;
    return { x, y, condition: c, val };
  });

  const pathD =
    points.length > 0
      ? points.reduce(
          (acc, p, idx) => `${acc} ${idx === 0 ? "M" : "L"} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`,
          ""
        )
      : "";

  const areaD =
    points.length > 0
      ? `${pathD} L ${points[points.length - 1].x.toFixed(1)} ${(
          padding.top + innerHeight
        ).toFixed(1)} L ${points[0].x.toFixed(1)} ${(padding.top + innerHeight).toFixed(1)} Z`
      : "";

  const controlVal = data?.control_condition
    ? (data.control_condition[selectedMetric] as number)
    : 0;
  const controlY = padding.top + (1 - (controlVal - minVal) / valRange) * innerHeight;

  if (isLoading && !data) {
    return (
      <div className="flex flex-col items-center justify-center p-24 text-slate-400 space-y-3">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-500"></div>
        <p className="text-xs font-mono text-slate-500">
          Loading B1 degradation conditions and calibration data...
        </p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="max-w-4xl mx-auto p-6 rounded-2xl bg-rose-500/10 border border-rose-500/20 text-rose-300 space-y-3">
        <div className="flex items-center gap-2 font-semibold">
          <AlertTriangle className="w-5 h-5 text-rose-400" />
          <span>Degradation Explorer Unavailable</span>
        </div>
        <p className="text-xs text-rose-200/80">{error}</p>
        <button
          onClick={loadDegradations}
          className="px-3 py-1.5 bg-rose-600/30 hover:bg-rose-600/50 rounded-lg text-xs font-medium text-white transition-colors"
        >
          Retry Connection
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-12">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <div className="flex items-center gap-2 text-indigo-400 text-xs font-semibold uppercase tracking-wider">
            <Sliders className="w-4 h-4" />
            <span>Track B1 Degradation Robustness</span>
          </div>
          <h1 className="text-xl md:text-2xl font-bold text-white mt-1">
            Degradation Explorer
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Interactive evaluation of OCR and KIE degradation curves across monotonic severities S1–S4 compared to Pristine Baseline (Control).
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-right">
            <div className="text-[10px] text-slate-500 font-mono uppercase">Canonical Split</div>
            <div className="text-xs font-semibold text-slate-200">{data.canonical_dataset}</div>
          </div>
        </div>
      </div>

      {/* Research Disclaimer Banner */}
      <div className="p-4 rounded-xl bg-indigo-950/30 border border-indigo-500/30 flex items-start gap-3 text-indigo-200">
        <ShieldCheck className="w-5 h-5 text-indigo-400 shrink-0 mt-0.5" />
        <div className="text-xs leading-relaxed space-y-1">
          <div className="font-semibold text-indigo-300">
            RESEARCH BENCHMARK ARTIFACTS — READ-ONLY MODE
          </div>
          <div className="text-indigo-200/80">
            Degradation impact and sensitivity curves are derived from frozen experimental runs ({data.canonical_dataset}).
            No simulated or synthetic documents are injected into production processing queues.
          </div>
        </div>
      </div>

      {/* Baseline Control Summary Card */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800">
          <div className="text-[10px] text-slate-500 font-mono uppercase">Baseline Control CER</div>
          <div className="text-lg font-bold text-slate-100 mt-0.5">{data.control_condition.cer.toFixed(4)}</div>
          <div className="text-[10px] text-slate-400 mt-0.5">RapidOCR pristine text CER</div>
        </div>
        <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800">
          <div className="text-[10px] text-slate-500 font-mono uppercase">Baseline Control NED</div>
          <div className="text-lg font-bold text-slate-100 mt-0.5">{data.control_condition.ned.toFixed(4)}</div>
          <div className="text-[10px] text-slate-400 mt-0.5">Normalized edit distance</div>
        </div>
        <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800">
          <div className="text-[10px] text-slate-500 font-mono uppercase">Baseline Control Macro F1</div>
          <div className="text-lg font-bold text-slate-100 mt-0.5">{data.control_condition.macro_f1.toFixed(4)}</div>
          <div className="text-[10px] text-slate-400 mt-0.5">KIE token extraction F1</div>
        </div>
        <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800">
          <div className="text-[10px] text-slate-500 font-mono uppercase">Baseline Entity H-Mean</div>
          <div className="text-lg font-bold text-slate-100 mt-0.5">{data.control_condition.entity_hmean.toFixed(4)}</div>
          <div className="text-[10px] text-slate-400 mt-0.5">Official SROIE strict entity metric</div>
        </div>
      </div>

      {/* Degradation Selector (8 Degradation Types) */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
            <Layers className="w-3.5 h-3.5 text-indigo-400" />
            <span>Select Degradation Type ({data.items.length})</span>
          </label>
          <span className="text-[11px] text-slate-500 font-mono">D1 through D8 conditions</span>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2">
          {data.items.map((item) => {
            const isSelected = item.id === selectedDegId;
            return (
              <button
                key={item.id}
                onClick={() => {
                  setSelectedDegId(item.id);
                  setSelectedConditionId(null);
                }}
                className={`p-2.5 rounded-xl text-left transition-all border ${
                  isSelected
                    ? "bg-indigo-600/20 border-indigo-500/50 shadow-lg shadow-indigo-500/10 text-white"
                    : "bg-slate-900/60 border-slate-800/80 hover:bg-slate-800/60 text-slate-300 hover:text-white"
                }`}
              >
                <div className="flex items-center justify-between">
                  <span
                    className={`px-1.5 py-0.5 rounded text-[10px] font-mono font-bold ${
                      isSelected ? "bg-indigo-500 text-white" : "bg-slate-800 text-slate-400"
                    }`}
                  >
                    {item.code}
                  </span>
                </div>
                <div className="text-xs font-semibold mt-1.5 truncate" title={item.name}>
                  {item.name.replace(" Robustness", "")}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {currentItem && (
        <div className="space-y-6">
          {/* Main Visualization Card & Metric Switcher */}
          <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-slate-800/80 pb-4">
              <div>
                <div className="flex items-center gap-2">
                  <span className="px-2 py-0.5 rounded text-xs font-mono font-bold bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                    {currentItem.code}
                  </span>
                  <h2 className="text-base font-bold text-white">{currentItem.name}</h2>
                </div>
                <p className="text-xs text-slate-400 mt-1">{currentItem.description}</p>
              </div>

              {/* Metric Switcher Pills */}
              <div className="flex items-center gap-1.5 bg-slate-950 p-1 rounded-xl border border-slate-800 self-start sm:self-center">
                {METRICS.map((m) => {
                  const isActive = selectedMetric === m.key;
                  return (
                    <button
                      key={m.key}
                      onClick={() => setSelectedMetric(m.key)}
                      className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                        isActive
                          ? "bg-indigo-600 text-white shadow"
                          : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
                      }`}
                    >
                      {m.label}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Severity Curve (SVG Line Chart) */}
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span className="font-semibold text-slate-300 flex items-center gap-1.5">
                  <Activity className="w-4 h-4 text-indigo-400" />
                  <span>Degradation Trajectory: {currentMetricConfig.fullName}</span>
                </span>
                <span className="text-[11px] font-mono text-slate-500">
                  Click a point to inspect condition details
                </span>
              </div>

              <div className="w-full bg-slate-950 rounded-xl p-3 border border-slate-800/60 overflow-x-auto">
                <svg
                  viewBox={`0 0 ${chartWidth} ${chartHeight}`}
                  className="w-full h-auto min-w-[500px]"
                >
                  <defs>
                    <linearGradient id="curveGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={currentMetricConfig.color} stopOpacity="0.3" />
                      <stop offset="100%" stopColor={currentMetricConfig.color} stopOpacity="0.0" />
                    </linearGradient>
                  </defs>

                  {/* Horizontal grid lines */}
                  {[0.0, 0.25, 0.5, 0.75, 1.0].map((tick) => {
                    const y = padding.top + (1 - (tick - minVal) / valRange) * innerHeight;
                    if (y < padding.top || y > padding.top + innerHeight) return null;
                    return (
                      <g key={tick}>
                        <line
                          x1={padding.left}
                          y1={y}
                          x2={padding.left + innerWidth}
                          y2={y}
                          stroke="#1e293b"
                          strokeDasharray="3 3"
                        />
                        <text
                          x={padding.left - 8}
                          y={y + 3}
                          fill="#64748b"
                          fontSize="9"
                          fontFamily="monospace"
                          textAnchor="end"
                        >
                          {tick.toFixed(2)}
                        </text>
                      </g>
                    );
                  })}

                  {/* Pristine Control Baseline reference line */}
                  {controlY >= padding.top && controlY <= padding.top + innerHeight && (
                    <g>
                      <line
                        x1={padding.left}
                        y1={controlY}
                        x2={padding.left + innerWidth}
                        y2={controlY}
                        stroke="#94a3b8"
                        strokeDasharray="4 4"
                        strokeOpacity="0.5"
                      />
                      <text
                        x={padding.left + innerWidth - 5}
                        y={controlY - 4}
                        fill="#94a3b8"
                        fontSize="8"
                        fontFamily="monospace"
                        textAnchor="end"
                      >
                        Baseline Control ({controlVal.toFixed(4)})
                      </text>
                    </g>
                  )}

                  {/* Area fill */}
                  {areaD && <path d={areaD} fill="url(#curveGradient)" />}

                  {/* Trend line */}
                  {pathD && (
                    <path
                      d={pathD}
                      fill="none"
                      stroke={currentMetricConfig.color}
                      strokeWidth="2.5"
                      strokeLinecap="round"
                    />
                  )}

                  {/* Condition Points */}
                  {points.map((p) => {
                    const isSelected = p.condition.condition_id === selectedCondition?.condition_id;
                    return (
                      <g
                        key={p.condition.condition_id}
                        className="cursor-pointer group"
                        onClick={() => setSelectedConditionId(p.condition.condition_id)}
                      >
                        {/* Outer hover ring */}
                        <circle
                          cx={p.x}
                          cy={p.y}
                          r={isSelected ? 8 : 6}
                          fill={isSelected ? currentMetricConfig.color : "#0f172a"}
                          stroke={currentMetricConfig.color}
                          strokeWidth={isSelected ? 3 : 2}
                          className="transition-all group-hover:r-8"
                        />
                        {/* Value label above point */}
                        <text
                          x={p.x}
                          y={p.y - 10}
                          fill={isSelected ? "#ffffff" : "#94a3b8"}
                          fontSize="10"
                          fontWeight={isSelected ? "bold" : "normal"}
                          fontFamily="monospace"
                          textAnchor="middle"
                        >
                          {p.val.toFixed(4)}
                        </text>

                        {/* X-axis tick & severity label */}
                        <text
                          x={p.x}
                          y={padding.top + innerHeight + 16}
                          fill={isSelected ? "#ffffff" : "#cbd5e1"}
                          fontSize="10"
                          fontWeight="600"
                          fontFamily="monospace"
                          textAnchor="middle"
                        >
                          {p.condition.severity}
                        </text>

                        {/* Parameter label under severity */}
                        <text
                          x={p.x}
                          y={padding.top + innerHeight + 28}
                          fill="#64748b"
                          fontSize="8"
                          fontFamily="monospace"
                          textAnchor="middle"
                        >
                          {p.condition.parameter || "none"}
                        </text>
                      </g>
                    );
                  })}
                </svg>
              </div>
            </div>
          </div>

          {/* Derived Research Findings & Formal Inflection Jump */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="md:col-span-2 p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
              <div className="flex items-center gap-2 text-amber-400 text-xs font-semibold uppercase tracking-wider">
                <TrendingDown className="w-4 h-4" />
                <span>Derived Research Findings & Inflection Analysis</span>
              </div>
              <div className="space-y-2.5">
                {currentItem.findings.map((f, idx) => (
                  <div
                    key={idx}
                    className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 text-xs leading-relaxed text-slate-300 flex items-start gap-2.5"
                  >
                    <span className="w-1.5 h-1.5 rounded-full bg-amber-400 mt-1.5 shrink-0" />
                    <span>{f}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Source Artifact Traceability */}
            <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3 flex flex-col justify-between">
              <div>
                <div className="flex items-center gap-2 text-indigo-400 text-xs font-semibold uppercase tracking-wider">
                  <Database className="w-4 h-4" />
                  <span>Artifact Traceability</span>
                </div>
                <div className="space-y-2 mt-3 text-xs">
                  <div>
                    <div className="text-[10px] text-slate-500 uppercase font-mono">B1 Detailed CSV</div>
                    <code className="text-[11px] text-slate-300 font-mono break-all">
                      {currentItem.source}
                    </code>
                  </div>
                  {currentItem.calibration_source && (
                    <div className="pt-2 border-t border-slate-800/80">
                      <div className="text-[10px] text-slate-500 uppercase font-mono">Severity Calibration</div>
                      <code className="text-[11px] text-slate-300 font-mono break-all">
                        {currentItem.calibration_source}
                      </code>
                    </div>
                  )}
                  <div className="pt-2 border-t border-slate-800/80">
                    <div className="text-[10px] text-slate-500 uppercase font-mono">Evaluated Receipts</div>
                    <div className="text-xs text-slate-300 font-medium">{currentItem.sample_size} validation documents</div>
                  </div>
                </div>
              </div>

              <div className="pt-3 border-t border-slate-800/80 flex items-center gap-1.5 text-[11px] text-emerald-400">
                <CheckCircle2 className="w-3.5 h-3.5 shrink-0" />
                <span>Cryptographically verified read-only</span>
              </div>
            </div>
          </div>

          {/* Condition Comparison Table */}
          <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <FileText className="w-4 h-4 text-indigo-400" />
                <span>Condition Comparison Table</span>
              </h3>
              <span className="text-xs text-slate-500 font-mono">
                Deltas relative to Control (D0_S0_P0)
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400 font-mono uppercase text-[10px]">
                    <th className="py-2.5 px-3">Condition ID</th>
                    <th className="py-2.5 px-3">Severity</th>
                    <th className="py-2.5 px-3">Parameter</th>
                    <th className="py-2.5 px-3">CER (Δ vs Ctrl)</th>
                    <th className="py-2.5 px-3">NED (Δ vs Ctrl)</th>
                    <th className="py-2.5 px-3">Macro F1 (Δ vs Ctrl)</th>
                    <th className="py-2.5 px-3">Entity H-Mean (Δ)</th>
                    <th className="py-2.5 px-3 text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {conditions.map((c) => {
                    const isSelected = c.condition_id === selectedCondition?.condition_id;
                    const isControl = c.severity === "Control";

                    return (
                      <tr
                        key={c.condition_id}
                        onClick={() => setSelectedConditionId(c.condition_id)}
                        className={`cursor-pointer transition-colors ${
                          isSelected
                            ? "bg-indigo-600/15 text-white"
                            : "hover:bg-slate-800/40 text-slate-300"
                        }`}
                      >
                        <td className="py-3 px-3 font-mono font-medium">{c.condition_id}</td>
                        <td className="py-3 px-3">
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold ${
                              isControl
                                ? "bg-slate-800 text-slate-300 border border-slate-700"
                                : "bg-indigo-500/20 text-indigo-300 border border-indigo-500/30"
                            }`}
                          >
                            {c.severity}
                          </span>
                        </td>
                        <td className="py-3 px-3 font-mono text-slate-400">{c.parameter || "none"}</td>
                        {/* CER: lower is better -> positive delta is bad (rose), negative delta is good (emerald) */}
                        <td className="py-3 px-3 font-mono">
                          <span className="font-semibold">{c.cer.toFixed(4)}</span>
                          {!isControl && (
                            <span
                              className={`ml-1.5 text-[10px] ${
                                c.delta_cer > 0 ? "text-rose-400" : "text-emerald-400"
                              }`}
                            >
                              ({c.delta_cer >= 0 ? `+${c.delta_cer.toFixed(4)}` : c.delta_cer.toFixed(4)})
                            </span>
                          )}
                        </td>
                        {/* NED: higher is better -> positive delta is good, negative is bad */}
                        <td className="py-3 px-3 font-mono">
                          <span className="font-semibold">{c.ned.toFixed(4)}</span>
                          {!isControl && (
                            <span
                              className={`ml-1.5 text-[10px] ${
                                c.delta_ned < 0 ? "text-rose-400" : "text-emerald-400"
                              }`}
                            >
                              ({c.delta_ned >= 0 ? `+${c.delta_ned.toFixed(4)}` : c.delta_ned.toFixed(4)})
                            </span>
                          )}
                        </td>
                        {/* Macro F1: higher is better */}
                        <td className="py-3 px-3 font-mono">
                          <span className="font-semibold">{c.macro_f1.toFixed(4)}</span>
                          {!isControl && (
                            <span
                              className={`ml-1.5 text-[10px] ${
                                c.delta_macro_f1 < 0 ? "text-rose-400" : "text-emerald-400"
                              }`}
                            >
                              ({c.delta_macro_f1 >= 0 ? `+${c.delta_macro_f1.toFixed(4)}` : c.delta_macro_f1.toFixed(4)})
                            </span>
                          )}
                        </td>
                        {/* Entity H-Mean: higher is better */}
                        <td className="py-3 px-3 font-mono">
                          <span className="font-semibold">{c.entity_hmean.toFixed(4)}</span>
                          {!isControl && (
                            <span
                              className={`ml-1.5 text-[10px] ${
                                c.delta_entity_hmean < 0 ? "text-rose-400" : "text-emerald-400"
                              }`}
                            >
                              ({c.delta_entity_hmean >= 0 ? `+${c.delta_entity_hmean.toFixed(4)}` : c.delta_entity_hmean.toFixed(4)})
                            </span>
                          )}
                        </td>
                        <td className="py-3 px-3 text-right">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              setSelectedConditionId(c.condition_id);
                            }}
                            className={`px-2.5 py-1 rounded text-[10px] font-medium transition-colors ${
                              isSelected
                                ? "bg-indigo-600 text-white"
                                : "bg-slate-800 text-slate-400 hover:text-white hover:bg-slate-700"
                            }`}
                          >
                            {isSelected ? "Selected" : "Inspect"}
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Selected Condition Details Card */}
          {selectedCondition && (
            <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 border-b border-slate-800/80 pb-3">
                <div className="flex items-center gap-2">
                  <span className="px-2 py-0.5 rounded text-xs font-mono font-bold bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                    {selectedCondition.condition_id}
                  </span>
                  <span className="text-sm font-bold text-white">
                    Condition Details: {selectedCondition.severity} ({selectedCondition.parameter || "none"})
                  </span>
                </div>
                <span className="text-xs text-slate-500 font-mono">
                  {selectedCondition.severity === "Control"
                    ? "Baseline Control Benchmark"
                    : "Monotonic Degradation Severity"}
                </span>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                  <div className="text-[10px] text-slate-500 font-mono uppercase">Normalized CER</div>
                  <div className="text-base font-bold text-slate-100 mt-0.5">
                    {selectedCondition.cer.toFixed(4)}
                  </div>
                  <div className="text-[10px] text-slate-400 mt-0.5">
                    Δ vs Control:{" "}
                    <span
                      className={
                        selectedCondition.delta_cer > 0
                          ? "text-rose-400 font-semibold"
                          : "text-emerald-400"
                      }
                    >
                      {selectedCondition.delta_cer >= 0
                        ? `+${selectedCondition.delta_cer.toFixed(4)}`
                        : selectedCondition.delta_cer.toFixed(4)}
                    </span>
                  </div>
                </div>

                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                  <div className="text-[10px] text-slate-500 font-mono uppercase">Normalized NED</div>
                  <div className="text-base font-bold text-slate-100 mt-0.5">
                    {selectedCondition.ned.toFixed(4)}
                  </div>
                  <div className="text-[10px] text-slate-400 mt-0.5">
                    Δ vs Control:{" "}
                    <span
                      className={
                        selectedCondition.delta_ned < 0
                          ? "text-rose-400 font-semibold"
                          : "text-emerald-400"
                      }
                    >
                      {selectedCondition.delta_ned >= 0
                        ? `+${selectedCondition.delta_ned.toFixed(4)}`
                        : selectedCondition.delta_ned.toFixed(4)}
                    </span>
                  </div>
                </div>

                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                  <div className="text-[10px] text-slate-500 font-mono uppercase">KIE Macro F1</div>
                  <div className="text-base font-bold text-slate-100 mt-0.5">
                    {selectedCondition.macro_f1.toFixed(4)}
                  </div>
                  <div className="text-[10px] text-slate-400 mt-0.5">
                    Δ vs Control:{" "}
                    <span
                      className={
                        selectedCondition.delta_macro_f1 < 0
                          ? "text-rose-400 font-semibold"
                          : "text-emerald-400"
                      }
                    >
                      {selectedCondition.delta_macro_f1 >= 0
                        ? `+${selectedCondition.delta_macro_f1.toFixed(4)}`
                        : selectedCondition.delta_macro_f1.toFixed(4)}
                    </span>
                  </div>
                </div>

                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                  <div className="text-[10px] text-slate-500 font-mono uppercase">SROIE Entity H-Mean</div>
                  <div className="text-base font-bold text-slate-100 mt-0.5">
                    {selectedCondition.entity_hmean.toFixed(4)}
                  </div>
                  <div className="text-[10px] text-slate-400 mt-0.5">
                    Δ vs Control:{" "}
                    <span
                      className={
                        selectedCondition.delta_entity_hmean < 0
                          ? "text-rose-400 font-semibold"
                          : "text-emerald-400"
                      }
                    >
                      {selectedCondition.delta_entity_hmean >= 0
                        ? `+${selectedCondition.delta_entity_hmean.toFixed(4)}`
                        : selectedCondition.delta_entity_hmean.toFixed(4)}
                    </span>
                  </div>
                </div>
              </div>

              {selectedCondition.parameters &&
                Object.keys(selectedCondition.parameters).length > 0 && (
                  <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                    <div className="text-[10px] text-slate-500 font-mono uppercase mb-1.5">
                      Structured Calibration Parameters
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {Object.entries(selectedCondition.parameters).map(([k, v]) => (
                        <span
                          key={k}
                          className="px-2 py-1 rounded bg-slate-900 border border-slate-800 text-[11px] font-mono text-slate-300"
                        >
                          <span className="text-indigo-400">{k}:</span> {String(v)}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
