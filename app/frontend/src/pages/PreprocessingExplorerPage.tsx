import React, { useEffect, useState } from "react";
import {
  PreprocessingConditionGroup,
  PreprocessingExplorerResponse,
  PreprocessingPolicyMetric,
} from "../types";
import { apiResearch } from "../api/client";
import {
  Sparkles,
  ShieldCheck,
  Activity,
  AlertTriangle,
  FileText,
  Layers,
  Database,
  CheckCircle2,
  TrendingDown,
  ArrowRight,
  Sliders,
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
    fullName: "KIE Token Extraction F1",
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

// Color palette for distinct policies
const POLICY_COLORS: Record<string, string> = {
  p_standard_receipt_enhancement: "#38bdf8", // Sky blue
  p_contrast_enhancement: "#a855f7",        // Purple
  p_minimal_cleanup: "#34d399",             // Emerald
  p_aggressive_binarization: "#fb923c",     // Amber/Orange
};

interface PreprocessingExplorerPageProps {
  initialDegradation?: string;
}

export const PreprocessingExplorerPage: React.FC<PreprocessingExplorerPageProps> = ({
  initialDegradation,
}) => {
  const [data, setData] = useState<PreprocessingExplorerResponse | null>(null);
  const [selectedDegCode, setSelectedDegCode] = useState<string>("D6"); // Default to Rotation (notable recovery)
  const [selectedSeverity, setSelectedSeverity] = useState<string>("S2");
  const [selectedMetric, setSelectedMetric] = useState<MetricKey>("cer");
  const [selectedPolicyKey, setSelectedPolicyKey] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadData = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await apiResearch.getPreprocessing();
      setData(res);

      const target = initialDegradation || new URLSearchParams(window.location.search).get("degradation");
      if (target && res.degradations.length > 0) {
        const match = res.degradations.find(
          (d) =>
            d.code.toUpperCase() === target.toUpperCase() ||
            (d.id && d.id.toLowerCase() === target.toLowerCase()) ||
            (d.key && d.key.toLowerCase() === target.toLowerCase()) ||
            d.name.toLowerCase().includes(target.toLowerCase())
        );
        if (match) {
          setSelectedDegCode(match.code);
        }
      }
    } catch (err: any) {
      console.error("Failed to load preprocessing explorer data", err);
      setError("Unable to load B2 preprocessing experiments. Please verify research artifacts are accessible.");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  useEffect(() => {
    if (data && initialDegradation) {
      const match = data.degradations.find(
        (d) =>
          d.code.toUpperCase() === initialDegradation.toUpperCase() ||
          (d.id && d.id.toLowerCase() === initialDegradation.toLowerCase()) ||
          (d.key && d.key.toLowerCase() === initialDegradation.toLowerCase()) ||
          d.name.toLowerCase().includes(initialDegradation.toLowerCase())
      );
      if (match) {
        setSelectedDegCode(match.code);
      }
    }
  }, [initialDegradation, data]);

  // Find matching group for selected degradation & severity
  const currentGroup: PreprocessingConditionGroup | undefined = data?.groups.find(
    (g) => g.degradation_code === selectedDegCode && g.severity === selectedSeverity
  ) || data?.groups[0];

  const currentMetricConfig = METRICS.find((m) => m.key === selectedMetric) || METRICS[0];

  const ctrlVal = currentGroup ? currentGroup.clean_control_metrics[selectedMetric] || 0 : 0;
  const degVal = currentGroup ? currentGroup.degraded_metrics[selectedMetric] || 0 : 0;
  const policies = currentGroup?.policies || [];

  const selectedPolicy: PreprocessingPolicyMetric | undefined =
    policies.find((p) => p.policy_key === selectedPolicyKey) || policies[0];

  // SVG Chart Dimensions & Scale calculations
  const chartWidth = 680;
  const chartHeight = 240;
  const padding = { top: 25, right: 35, bottom: 50, left: 55 };
  const innerWidth = chartWidth - padding.left - padding.right;
  const innerHeight = chartHeight - padding.top - padding.bottom;

interface PlotItem {
  id: string;
  label: string;
  sublabel: string;
  val: number;
  color: string;
  isControl: boolean;
  isDegraded: boolean;
  policy?: PreprocessingPolicyMetric;
}

  // Items to plot on horizontal axis: Clean Control, Degraded, and the 4 policies
  const plotItems: PlotItem[] = currentGroup
    ? [
        {
          id: "clean_control",
          label: "Clean Control",
          sublabel: "B0 Pristine",
          val: ctrlVal,
          color: "#94a3b8",
          isControl: true,
          isDegraded: false,
        },
        {
          id: "degraded",
          label: "Degraded",
          sublabel: `${currentGroup.ref_b1_condition_id}`,
          val: degVal,
          color: "#ef4444",
          isControl: false,
          isDegraded: true,
        },
        ...policies.map((p) => ({
          id: p.policy_key,
          label: p.policy_name.split(" ")[0],
          sublabel: p.policy_name.replace("Enhancement", "Enh.").replace("Binarization", "Bin."),
          val: p.metrics[selectedMetric] || 0,
          color: POLICY_COLORS[p.policy_key] || "#818cf8",
          isControl: false,
          isDegraded: false,
          policy: p,
        })),
      ]
    : [];

  const allVals = plotItems.map((p) => p.val);
  const minVal = Math.max(0, Math.min(...allVals) * 0.85);
  const maxVal = Math.min(1.0, Math.max(...allVals) * 1.15 || 1.0);
  const valRange = maxVal - minVal || 1.0;

  const points = plotItems.map((item, idx) => {
    const x = padding.left + (idx / Math.max(1, plotItems.length - 1)) * innerWidth;
    const y = padding.top + (1 - (item.val - minVal) / valRange) * innerHeight;
    return { ...item, x, y };
  });

  if (isLoading && !data) {
    return (
      <div className="flex flex-col items-center justify-center p-24 text-slate-400 space-y-3">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-500"></div>
        <p className="text-xs font-mono text-slate-500">
          Loading B2 preprocessing evaluation conditions and references...
        </p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="max-w-4xl mx-auto p-6 rounded-2xl bg-rose-500/10 border border-rose-500/20 text-rose-300 space-y-3">
        <div className="flex items-center gap-2 font-semibold">
          <AlertTriangle className="w-5 h-5 text-rose-400" />
          <span>Preprocessing Explorer Unavailable</span>
        </div>
        <p className="text-xs text-rose-200/80">{error}</p>
        <button
          onClick={loadData}
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
            <Sparkles className="w-4 h-4" />
            <span>Track B2 Preprocessing & Recovery Explorer</span>
          </div>
          <h1 className="text-xl md:text-2xl font-bold text-white mt-1">
            Preprocessing Explorer
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Comparative evaluation of 4 preprocessing policies across 32 degradation regimes (D1–D8, S1–S4) relative to Degraded and Clean Control references.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-right">
            <div className="text-[10px] text-slate-500 font-mono uppercase">Total Evaluations</div>
            <div className="text-xs font-semibold text-slate-200">
              {data.total_conditions_evaluated} conditions ({data.canonical_dataset})
            </div>
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
            Preprocessing recovery and regression measurements are derived from frozen experimental runs ({data.canonical_dataset}).
            These results constitute scientific evidence and do <strong>NOT</strong> serve as production preprocessing recommendations or automatic routing rules.
          </div>
        </div>
      </div>

      {/* Selectors: Degradation + Severity */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Degradation Selector (8 types) */}
        <div className="lg:col-span-2 space-y-2">
          <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
            <Layers className="w-3.5 h-3.5 text-indigo-400" />
            <span>Select Degradation Type ({data.degradations.length})</span>
          </label>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {data.degradations.map((d) => {
              const isSelected = d.code === selectedDegCode;
              return (
                <button
                  key={d.code}
                  onClick={() => setSelectedDegCode(d.code)}
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
                      {d.code}
                    </span>
                  </div>
                  <div className="text-xs font-semibold mt-1.5 truncate" title={d.name}>
                    {d.name.replace(" Robustness", "")}
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Severity Selector (S1..S4) */}
        <div className="space-y-2">
          <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
            <Sliders className="w-3.5 h-3.5 text-indigo-400" />
            <span>Severity Level</span>
          </label>
          <div className="grid grid-cols-4 gap-2">
            {data.severities.map((sev) => {
              const isSelected = sev === selectedSeverity;
              return (
                <button
                  key={sev}
                  onClick={() => setSelectedSeverity(sev)}
                  className={`py-3 px-2 rounded-xl text-center font-mono transition-all border ${
                    isSelected
                      ? "bg-indigo-600 text-white border-indigo-500 shadow-md font-bold"
                      : "bg-slate-900/60 border-slate-800 text-slate-300 hover:bg-slate-800"
                  }`}
                >
                  <div className="text-sm">{sev}</div>
                  <div className="text-[10px] text-slate-400 mt-0.5">
                    {sev === "S1" ? "Mild" : sev === "S2" ? "Mod." : sev === "S3" ? "High" : "Extreme"}
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {currentGroup && (
        <div className="space-y-6">
          {/* Main Visualization & Metric Switcher Card */}
          <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-slate-800/80 pb-4">
              <div>
                <div className="flex items-center gap-2">
                  <span className="px-2 py-0.5 rounded text-xs font-mono font-bold bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                    {currentGroup.degradation_code} — {currentGroup.severity}
                  </span>
                  <h2 className="text-base font-bold text-white">
                    {currentGroup.degradation_name} ({currentGroup.severity})
                  </h2>
                </div>
                <div className="flex items-center gap-3 text-xs text-slate-400 mt-1 font-mono">
                  <span>Ref Degraded: {currentGroup.ref_b1_condition_id}</span>
                  <span>•</span>
                  <span>Clean Control: D0_S0_P0</span>
                </div>
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

            {/* SVG Visualizer */}
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span className="font-semibold text-slate-300 flex items-center gap-1.5">
                  <Activity className="w-4 h-4 text-indigo-400" />
                  <span>Comparative Policy Response: {currentMetricConfig.fullName}</span>
                </span>
                <span className="text-[11px] font-mono text-slate-500">
                  Clean Control vs Degraded vs Preprocessing Policies
                </span>
              </div>

              <div className="w-full bg-slate-950 rounded-xl p-3 border border-slate-800/60 overflow-x-auto">
                <svg
                  viewBox={`0 0 ${chartWidth} ${chartHeight}`}
                  className="w-full h-auto min-w-[550px]"
                >
                  {/* Grid lines */}
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

                  {/* Clean Control Reference Line (dashed) */}
                  {points[0] && (
                    <g>
                      <line
                        x1={padding.left}
                        y1={points[0].y}
                        x2={padding.left + innerWidth}
                        y2={points[0].y}
                        stroke="#94a3b8"
                        strokeDasharray="4 4"
                        strokeOpacity="0.4"
                      />
                      <text
                        x={padding.left + innerWidth - 5}
                        y={points[0].y - 4}
                        fill="#94a3b8"
                        fontSize="8"
                        fontFamily="monospace"
                        textAnchor="end"
                      >
                        Clean Control ({ctrlVal.toFixed(4)})
                      </text>
                    </g>
                  )}

                  {/* Degraded Reference Line (dotted) */}
                  {points[1] && (
                    <g>
                      <line
                        x1={padding.left}
                        y1={points[1].y}
                        x2={padding.left + innerWidth}
                        y2={points[1].y}
                        stroke="#ef4444"
                        strokeDasharray="2 2"
                        strokeOpacity="0.5"
                      />
                      <text
                        x={padding.left + innerWidth - 5}
                        y={points[1].y - 4}
                        fill="#ef4444"
                        fontSize="8"
                        fontFamily="monospace"
                        textAnchor="end"
                      >
                        Degraded Reference ({degVal.toFixed(4)})
                      </text>
                    </g>
                  )}

                  {/* Connecting lines between policies */}
                  {points.length > 2 && (
                    <path
                      d={points.slice(2).reduce(
                        (acc, p, idx) => `${acc} ${idx === 0 ? "M" : "L"} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`,
                        ""
                      )}
                      fill="none"
                      stroke="#475569"
                      strokeWidth="1.5"
                      strokeDasharray="2 2"
                    />
                  )}

                  {/* Plotted Points */}
                  {points.map((p) => {
                    const isSelected = p.id === selectedPolicy?.policy_key;
                    return (
                      <g
                        key={p.id}
                        className="cursor-pointer group"
                        onClick={() => {
                          if (p.policy) {
                            setSelectedPolicyKey(p.policy.policy_key);
                          }
                        }}
                      >
                        {/* Drop line to X axis */}
                        <line
                          x1={p.x}
                          y1={p.y}
                          x2={p.x}
                          y2={padding.top + innerHeight}
                          stroke="#334155"
                          strokeDasharray="1 2"
                        />

                        {/* Outer Ring */}
                        <circle
                          cx={p.x}
                          cy={p.y}
                          r={isSelected ? 8 : 6}
                          fill={p.isControl ? "#475569" : p.isDegraded ? "#ef4444" : p.color}
                          stroke={isSelected ? "#ffffff" : "#0f172a"}
                          strokeWidth={isSelected ? 3 : 2}
                          className="transition-all"
                        />

                        {/* Value label */}
                        <text
                          x={p.x}
                          y={p.y - 10}
                          fill={isSelected ? "#ffffff" : "#cbd5e1"}
                          fontSize="10"
                          fontWeight={isSelected ? "bold" : "600"}
                          fontFamily="monospace"
                          textAnchor="middle"
                        >
                          {p.val.toFixed(4)}
                        </text>

                        {/* X-axis label */}
                        <text
                          x={p.x}
                          y={padding.top + innerHeight + 16}
                          fill={isSelected ? "#ffffff" : "#94a3b8"}
                          fontSize="9"
                          fontWeight={isSelected ? "bold" : "500"}
                          textAnchor="middle"
                        >
                          {p.label}
                        </text>

                        {/* X-axis sublabel */}
                        <text
                          x={p.x}
                          y={padding.top + innerHeight + 28}
                          fill="#64748b"
                          fontSize="8"
                          fontFamily="monospace"
                          textAnchor="middle"
                        >
                          {p.sublabel}
                        </text>
                      </g>
                    );
                  })}
                </svg>
              </div>
            </div>
          </div>

          {/* Research Observations Panel & Artifact Traceability */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="md:col-span-2 p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
              <div className="flex items-center gap-2 text-amber-400 text-xs font-semibold uppercase tracking-wider">
                <TrendingDown className="w-4 h-4" />
                <span>Computed Research Observations</span>
              </div>
              <div className="space-y-2.5">
                {currentGroup.findings.map((f, idx) => (
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

            {/* Artifact Traceability */}
            <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3 flex flex-col justify-between">
              <div>
                <div className="flex items-center gap-2 text-indigo-400 text-xs font-semibold uppercase tracking-wider">
                  <Database className="w-4 h-4" />
                  <span>Artifact Traceability</span>
                </div>
                <div className="space-y-2 mt-3 text-xs">
                  <div>
                    <div className="text-[10px] text-slate-500 uppercase font-mono">B2 Detailed CSV</div>
                    <code className="text-[11px] text-slate-300 font-mono break-all">
                      {data.source}
                    </code>
                  </div>
                  <div className="pt-2 border-t border-slate-800/80">
                    <div className="text-[10px] text-slate-500 uppercase font-mono">Degraded Reference (B1)</div>
                    <code className="text-[11px] text-slate-300 font-mono">
                      {currentGroup.ref_b1_condition_id}
                    </code>
                  </div>
                  <div className="pt-2 border-t border-slate-800/80">
                    <div className="text-[10px] text-slate-500 uppercase font-mono">Clean Control (B0)</div>
                    <code className="text-[11px] text-slate-300 font-mono">
                      D0_S0_P0
                    </code>
                  </div>
                </div>
              </div>

              <div className="pt-3 border-t border-slate-800/80 flex items-center gap-1.5 text-[11px] text-emerald-400">
                <CheckCircle2 className="w-3.5 h-3.5 shrink-0" />
                <span>Cryptographically verified read-only</span>
              </div>
            </div>
          </div>

          {/* Detailed Policy Comparison Table */}
          <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <FileText className="w-4 h-4 text-indigo-400" />
                <span>Policy Comparison Table ({currentGroup.degradation_code} — {currentGroup.severity})</span>
              </h3>
              <span className="text-xs text-slate-500 font-mono">
                Deltas relative to Degraded Condition ({currentGroup.ref_b1_condition_id})
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400 font-mono uppercase text-[10px]">
                    <th className="py-2.5 px-3">Condition / Policy</th>
                    <th className="py-2.5 px-3">CER (Δ vs Deg)</th>
                    <th className="py-2.5 px-3">NED (Δ vs Deg)</th>
                    <th className="py-2.5 px-3">Macro F1 (Δ vs Deg)</th>
                    <th className="py-2.5 px-3">Entity H-Mean (Δ)</th>
                    <th className="py-2.5 px-3">Rel. Recovery</th>
                    <th className="py-2.5 px-3 text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {/* Clean Control Reference Row */}
                  <tr className="bg-slate-950/40 text-slate-400">
                    <td className="py-3 px-3">
                      <div className="flex items-center gap-2 font-mono">
                        <span className="w-2 h-2 rounded-full bg-slate-500" />
                        <span className="font-semibold text-slate-300">Clean Control (D0_S0_P0)</span>
                      </div>
                    </td>
                    <td className="py-3 px-3 font-mono text-slate-300">
                      {currentGroup.clean_control_metrics.cer.toFixed(4)}
                    </td>
                    <td className="py-3 px-3 font-mono text-slate-300">
                      {currentGroup.clean_control_metrics.ned.toFixed(4)}
                    </td>
                    <td className="py-3 px-3 font-mono text-slate-300">
                      {currentGroup.clean_control_metrics.macro_f1.toFixed(4)}
                    </td>
                    <td className="py-3 px-3 font-mono text-slate-300">
                      {currentGroup.clean_control_metrics.entity_hmean.toFixed(4)}
                    </td>
                    <td className="py-3 px-3 font-mono text-slate-500">—</td>
                    <td className="py-3 px-3 text-right text-slate-500 font-mono text-[10px]">Reference</td>
                  </tr>

                  {/* Degraded Reference Row */}
                  <tr className="bg-rose-950/10 text-slate-300">
                    <td className="py-3 px-3">
                      <div className="flex items-center gap-2 font-mono">
                        <span className="w-2 h-2 rounded-full bg-rose-500" />
                        <span className="font-semibold text-rose-300">
                          Degraded ({currentGroup.ref_b1_condition_id})
                        </span>
                      </div>
                    </td>
                    <td className="py-3 px-3 font-mono font-semibold text-rose-200">
                      {currentGroup.degraded_metrics.cer.toFixed(4)}
                    </td>
                    <td className="py-3 px-3 font-mono text-slate-300">
                      {currentGroup.degraded_metrics.ned.toFixed(4)}
                    </td>
                    <td className="py-3 px-3 font-mono text-slate-300">
                      {currentGroup.degraded_metrics.macro_f1.toFixed(4)}
                    </td>
                    <td className="py-3 px-3 font-mono text-slate-300">
                      {currentGroup.degraded_metrics.entity_hmean.toFixed(4)}
                    </td>
                    <td className="py-3 px-3 font-mono text-slate-500">0.0%</td>
                    <td className="py-3 px-3 text-right text-slate-500 font-mono text-[10px]">Baseline</td>
                  </tr>

                  {/* 4 Preprocessing Policies */}
                  {policies.map((pol) => {
                    const isSelected = pol.policy_key === selectedPolicy?.policy_key;
                    const dotColor = POLICY_COLORS[pol.policy_key] || "#818cf8";

                    return (
                      <tr
                        key={pol.policy_key}
                        onClick={() => setSelectedPolicyKey(pol.policy_key)}
                        className={`cursor-pointer transition-colors ${
                          isSelected
                            ? "bg-indigo-600/15 text-white"
                            : "hover:bg-slate-800/40 text-slate-300"
                        }`}
                      >
                        <td className="py-3 px-3 font-mono font-medium">
                          <div className="flex items-center gap-2">
                            <span
                              className="w-2 h-2 rounded-full shrink-0"
                              style={{ backgroundColor: dotColor }}
                            />
                            <span>{pol.policy_name}</span>
                          </div>
                        </td>

                        {/* CER: lower is better -> negative delta vs degraded is good (emerald), positive is bad (rose) */}
                        <td className="py-3 px-3 font-mono">
                          <span className="font-semibold">{pol.metrics.cer.toFixed(4)}</span>
                          <span
                            className={`ml-1.5 text-[10px] ${
                              pol.delta_cer_vs_degraded < 0 ? "text-emerald-400" : "text-rose-400"
                            }`}
                          >
                            ({pol.delta_cer_vs_degraded >= 0 ? `+${pol.delta_cer_vs_degraded.toFixed(4)}` : pol.delta_cer_vs_degraded.toFixed(4)})
                          </span>
                        </td>

                        {/* NED: higher is better -> positive delta is good */}
                        <td className="py-3 px-3 font-mono">
                          <span className="font-semibold">{pol.metrics.ned.toFixed(4)}</span>
                          <span
                            className={`ml-1.5 text-[10px] ${
                              pol.delta_ned_vs_degraded >= 0 ? "text-emerald-400" : "text-rose-400"
                            }`}
                          >
                            ({pol.delta_ned_vs_degraded >= 0 ? `+${pol.delta_ned_vs_degraded.toFixed(4)}` : pol.delta_ned_vs_degraded.toFixed(4)})
                          </span>
                        </td>

                        {/* Macro F1: higher is better */}
                        <td className="py-3 px-3 font-mono">
                          <span className="font-semibold">{pol.metrics.macro_f1.toFixed(4)}</span>
                          <span
                            className={`ml-1.5 text-[10px] ${
                              pol.delta_macro_f1_vs_degraded >= 0 ? "text-emerald-400" : "text-rose-400"
                            }`}
                          >
                            ({pol.delta_macro_f1_vs_degraded >= 0 ? `+${pol.delta_macro_f1_vs_degraded.toFixed(4)}` : pol.delta_macro_f1_vs_degraded.toFixed(4)})
                          </span>
                        </td>

                        {/* Entity H-Mean: higher is better */}
                        <td className="py-3 px-3 font-mono">
                          <span className="font-semibold">{pol.metrics.entity_hmean.toFixed(4)}</span>
                          <span
                            className={`ml-1.5 text-[10px] ${
                              pol.delta_entity_hmean_vs_degraded >= 0 ? "text-emerald-400" : "text-rose-400"
                            }`}
                          >
                            ({pol.delta_entity_hmean_vs_degraded >= 0 ? `+${pol.delta_entity_hmean_vs_degraded.toFixed(4)}` : pol.delta_entity_hmean_vs_degraded.toFixed(4)})
                          </span>
                        </td>

                        {/* Relative Recovery Rate */}
                        <td className="py-3 px-3 font-mono font-semibold">
                          {pol.relative_recovery_rate !== null && pol.relative_recovery_rate !== undefined ? (
                            <span
                              className={
                                pol.relative_recovery_rate > 0
                                  ? "text-emerald-400"
                                  : pol.relative_recovery_rate < 0
                                  ? "text-rose-400"
                                  : "text-slate-400"
                              }
                            >
                              {(pol.relative_recovery_rate * 100.0).toFixed(1)}%
                            </span>
                          ) : (
                            <span className="text-slate-500" title="Degradation CER impact <= 0.005 (guard applied)">
                              —
                            </span>
                          )}
                        </td>

                        <td className="py-3 px-3 text-right">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              setSelectedPolicyKey(pol.policy_key);
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

          {/* Selected Policy Inspection Card */}
          {selectedPolicy && (
            <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 border-b border-slate-800/80 pb-3">
                <div className="flex items-center gap-2">
                  <span
                    className="w-2.5 h-2.5 rounded-full shrink-0"
                    style={{ backgroundColor: POLICY_COLORS[selectedPolicy.policy_key] || "#818cf8" }}
                  />
                  <span className="text-sm font-bold text-white">
                    {selectedPolicy.policy_name} ({selectedPolicy.condition_id})
                  </span>
                </div>
                <div className="text-xs text-slate-400 font-mono">
                  Relative Error Recovery:{" "}
                  {selectedPolicy.relative_recovery_rate !== null && selectedPolicy.relative_recovery_rate !== undefined ? (
                    <span className="font-bold text-emerald-400">
                      {(selectedPolicy.relative_recovery_rate * 100.0).toFixed(1)}%
                    </span>
                  ) : (
                    <span className="text-slate-500">— (guard: negligible baseline CER impact)</span>
                  )}
                </div>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                  <div className="text-[10px] text-slate-500 font-mono uppercase">Normalized CER</div>
                  <div className="text-base font-bold text-slate-100 mt-0.5">
                    {selectedPolicy.metrics.cer.toFixed(4)}
                  </div>
                  <div className="text-[10px] text-slate-400 mt-0.5">
                    Δ vs Degraded:{" "}
                    <span
                      className={
                        selectedPolicy.delta_cer_vs_degraded < 0
                          ? "text-emerald-400 font-semibold"
                          : "text-rose-400"
                      }
                    >
                      {selectedPolicy.delta_cer_vs_degraded >= 0
                        ? `+${selectedPolicy.delta_cer_vs_degraded.toFixed(4)}`
                        : selectedPolicy.delta_cer_vs_degraded.toFixed(4)}
                    </span>
                  </div>
                  <div className="text-[9px] text-slate-500 mt-0.5">
                    Δ vs Clean: {selectedPolicy.delta_cer_vs_clean >= 0 ? `+${selectedPolicy.delta_cer_vs_clean.toFixed(4)}` : selectedPolicy.delta_cer_vs_clean.toFixed(4)}
                  </div>
                </div>

                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                  <div className="text-[10px] text-slate-500 font-mono uppercase">Normalized NED</div>
                  <div className="text-base font-bold text-slate-100 mt-0.5">
                    {selectedPolicy.metrics.ned.toFixed(4)}
                  </div>
                  <div className="text-[10px] text-slate-400 mt-0.5">
                    Δ vs Degraded:{" "}
                    <span
                      className={
                        selectedPolicy.delta_ned_vs_degraded >= 0
                          ? "text-emerald-400 font-semibold"
                          : "text-rose-400"
                      }
                    >
                      {selectedPolicy.delta_ned_vs_degraded >= 0
                        ? `+${selectedPolicy.delta_ned_vs_degraded.toFixed(4)}`
                        : selectedPolicy.delta_ned_vs_degraded.toFixed(4)}
                    </span>
                  </div>
                </div>

                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                  <div className="text-[10px] text-slate-500 font-mono uppercase">KIE Macro F1</div>
                  <div className="text-base font-bold text-slate-100 mt-0.5">
                    {selectedPolicy.metrics.macro_f1.toFixed(4)}
                  </div>
                  <div className="text-[10px] text-slate-400 mt-0.5">
                    Δ vs Degraded:{" "}
                    <span
                      className={
                        selectedPolicy.delta_macro_f1_vs_degraded >= 0
                          ? "text-emerald-400 font-semibold"
                          : "text-rose-400"
                      }
                    >
                      {selectedPolicy.delta_macro_f1_vs_degraded >= 0
                        ? `+${selectedPolicy.delta_macro_f1_vs_degraded.toFixed(4)}`
                        : selectedPolicy.delta_macro_f1_vs_degraded.toFixed(4)}
                    </span>
                  </div>
                </div>

                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                  <div className="text-[10px] text-slate-500 font-mono uppercase">Entity H-Mean</div>
                  <div className="text-base font-bold text-slate-100 mt-0.5">
                    {selectedPolicy.metrics.entity_hmean.toFixed(4)}
                  </div>
                  <div className="text-[10px] text-slate-400 mt-0.5">
                    Δ vs Degraded:{" "}
                    <span
                      className={
                        selectedPolicy.delta_entity_hmean_vs_degraded >= 0
                          ? "text-emerald-400 font-semibold"
                          : "text-rose-400"
                      }
                    >
                      {selectedPolicy.delta_entity_hmean_vs_degraded >= 0
                        ? `+${selectedPolicy.delta_entity_hmean_vs_degraded.toFixed(4)}`
                        : selectedPolicy.delta_entity_hmean_vs_degraded.toFixed(4)}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
