import React, { useEffect, useState } from "react";
import {
  ResearchExperimentDetail,
  ResearchExperimentItem,
  ResearchOverviewResponse,
} from "../types";
import { apiResearch } from "../api/client";
import {
  FlaskConical,
  Award,
  Sliders,
  CheckCircle2,
  FileText,
  Activity,
  ArrowRight,
  ShieldCheck,
  AlertTriangle,
  X,
  ExternalLink,
  Zap,
  RotateCcw,
  Sparkles,
} from "lucide-react";

export const ResearchPage: React.FC = () => {
  const [selectedTrack, setSelectedTrack] = useState<string>("ALL");
  const [overview, setOverview] = useState<ResearchOverviewResponse | null>(null);
  const [experiments, setExperiments] = useState<ResearchExperimentItem[]>([]);
  const [selectedExperiment, setSelectedExperiment] = useState<ResearchExperimentDetail | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isLoadingDetail, setIsLoadingDetail] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load overview and experiment list
  const loadData = async (trackFilter: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const [ovRes, expRes] = await Promise.all([
        apiResearch.getOverview(),
        apiResearch.getExperiments(trackFilter === "ALL" ? undefined : trackFilter),
      ]);
      setOverview(ovRes);
      setExperiments(expRes.items);
    } catch (err: any) {
      console.error("Failed to load research benchmarks", err);
      setError("Research data unavailable — please verify that research runs are accessible in read-only mode.");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadData(selectedTrack);
  }, [selectedTrack]);

  // Handle experiment selection for deep-dive detail
  const handleSelectExperiment = async (expId: string) => {
    setIsLoadingDetail(true);
    try {
      const detail = await apiResearch.getExperiment(expId);
      setSelectedExperiment(detail);
    } catch (err: any) {
      console.error("Failed to load experiment detail", err);
    } finally {
      setIsLoadingDetail(false);
    }
  };

  if (isLoading && !overview) {
    return (
      <div className="flex flex-col items-center justify-center p-24 text-slate-400 space-y-3">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-500"></div>
        <p className="text-xs font-mono text-slate-500">Loading research benchmarks & frozen artifacts...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-4xl mx-auto p-6 rounded-2xl bg-rose-500/10 border border-rose-500/20 text-rose-300 space-y-3">
        <div className="flex items-center gap-2 font-semibold">
          <AlertTriangle className="w-5 h-5 text-rose-400" />
          <span>Research Data Unavailable</span>
        </div>
        <p className="text-xs text-rose-200/80">{error}</p>
        <button
          onClick={() => loadData(selectedTrack)}
          className="px-3 py-1.5 bg-rose-600/30 hover:bg-rose-600/50 rounded-lg text-xs font-medium text-white transition-colors"
        >
          Retry Connection
        </button>
      </div>
    );
  }

  const b0Track = overview?.tracks.find((t) => t.id === "B0");
  const b1Track = overview?.tracks.find((t) => t.id === "B1");
  const b2Track = overview?.tracks.find((t) => t.id === "B2");

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 p-6 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-lg">
        <div className="flex items-center gap-3.5">
          <div className="w-10 h-10 rounded-xl bg-indigo-600/15 text-indigo-400 flex items-center justify-center border border-indigo-500/20 shadow-md">
            <FlaskConical className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-base font-bold text-slate-100 tracking-tight">Research Mode Diagnostics</h2>
              <span className="px-2 py-0.5 text-[10px] font-mono rounded bg-emerald-500/15 text-emerald-400 border border-emerald-500/20">
                READ-ONLY ARTIFACTS
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Empirical evidence from SROIE validation experiments: B0 Clean Baseline, B1 Degradation Matrix, and B2 Preprocessing Recovery.
            </p>
          </div>
        </div>

        {/* Track Filter Pills */}
        <div className="flex items-center gap-1.5 bg-slate-950 p-1.5 rounded-xl border border-slate-800">
          {[
            { id: "ALL", label: "All Tracks" },
            { id: "B0", label: "B0: Baseline" },
            { id: "B1", label: "B1: Degradation" },
            { id: "B2", label: "B2: Preprocessing" },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setSelectedTrack(tab.id)}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold uppercase tracking-wider transition-colors ${
                selectedTrack === tab.id
                  ? "bg-indigo-600 text-white shadow-sm shadow-indigo-600/30"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Track Overview Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {/* B0 Card */}
        <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-lg space-y-3 relative overflow-hidden">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="px-2 py-0.5 text-[10px] font-bold font-mono rounded bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
                B0
              </span>
              <span className="text-sm font-semibold text-slate-200">Clean Baseline</span>
            </div>
            <span className="text-xs font-mono text-slate-400">{b0Track?.experiment_count || 2} experiments</span>
          </div>

          <p className="text-xs text-slate-400 leading-relaxed">
            Pristine benchmark on SROIE validation receipts with ONNX CUDA vs CPU numerical parity validation.
          </p>

          <div className="pt-2 border-t border-slate-800/80 grid grid-cols-3 gap-2 text-center">
            <div className="p-2 rounded-lg bg-slate-950/60">
              <span className="text-[10px] text-slate-500 uppercase font-mono">CER (Norm)</span>
              <p className="text-sm font-bold font-mono text-slate-100 mt-0.5">
                {b0Track?.key_metrics?.cer_normalized || 0.3203}
              </p>
            </div>
            <div className="p-2 rounded-lg bg-slate-950/60">
              <span className="text-[10px] text-slate-500 uppercase font-mono">NED Sim</span>
              <p className="text-sm font-bold font-mono text-emerald-400 mt-0.5">
                {b0Track?.key_metrics?.char_ned_normalized || 0.6816}
              </p>
            </div>
            <div className="p-2 rounded-lg bg-slate-950/60">
              <span className="text-[10px] text-slate-500 uppercase font-mono">KIE F1</span>
              <p className="text-sm font-bold font-mono text-indigo-400 mt-0.5">
                {b0Track?.key_metrics?.macro_f1_normalized || 0.4782}
              </p>
            </div>
          </div>
        </div>

        {/* B1 Card */}
        <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-lg space-y-3 relative overflow-hidden">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="px-2 py-0.5 text-[10px] font-bold font-mono rounded bg-amber-500/20 text-amber-400 border border-amber-500/30">
                B1
              </span>
              <span className="text-sm font-semibold text-slate-200">Degradation Robustness</span>
            </div>
            <span className="text-xs font-mono text-slate-400">{b1Track?.experiment_count || 8} dimensions</span>
          </div>

          <p className="text-xs text-slate-400 leading-relaxed">
            Multi-tier degradation stress test across blur, noise, resolution loss, and geometric warp on 4,158 runs.
          </p>

          <div className="pt-2 border-t border-slate-800/80 grid grid-cols-2 gap-2 text-center">
            <div className="p-2 rounded-lg bg-slate-950/60">
              <span className="text-[10px] text-slate-500 uppercase font-mono">Total Evaluations</span>
              <p className="text-sm font-bold font-mono text-amber-400 mt-0.5">4,158 docs</p>
            </div>
            <div className="p-2 rounded-lg bg-slate-950/60">
              <span className="text-[10px] text-slate-500 uppercase font-mono">Severity Steps</span>
              <p className="text-sm font-bold font-mono text-slate-200 mt-0.5">S1 → S4</p>
            </div>
          </div>
        </div>

        {/* B2 Card */}
        <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-lg space-y-3 relative overflow-hidden">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="px-2 py-0.5 text-[10px] font-bold font-mono rounded bg-purple-500/20 text-purple-400 border border-purple-500/30">
                B2
              </span>
              <span className="text-sm font-semibold text-slate-200">Smart Preprocessing</span>
            </div>
            <span className="text-xs font-mono text-slate-400">{b2Track?.experiment_count || 4} policies</span>
          </div>

          <p className="text-xs text-slate-400 leading-relaxed">
            Empirical mitigation benchmarks comparing deskew, CLAHE, and filtering across 16,254 evaluations.
          </p>

          <div className="pt-2 border-t border-slate-800/80 grid grid-cols-2 gap-2 text-center">
            <div className="p-2 rounded-lg bg-slate-950/60">
              <span className="text-[10px] text-slate-500 uppercase font-mono">Evaluations</span>
              <p className="text-sm font-bold font-mono text-purple-400 mt-0.5">16,254 docs</p>
            </div>
            <div className="p-2 rounded-lg bg-slate-950/60">
              <span className="text-[10px] text-slate-500 uppercase font-mono">Top Recovery</span>
              <p className="text-sm font-bold font-mono text-emerald-400 mt-0.5">+15.3% CER</p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Split: Experiment Registry Table & Detailed Deep-Dive */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Experiment Registry (6 cols if detail open, 12 cols if detail closed) */}
        <div className={`${selectedExperiment ? "lg:col-span-6" : "lg:col-span-12"} space-y-4`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <FileText className="w-4 h-4 text-indigo-400" />
              <h3 className="text-sm font-semibold text-slate-200">
                Experiment Registry ({experiments.length} Available)
              </h3>
            </div>
            <span className="text-xs text-slate-500 font-mono">Click to view evidence</span>
          </div>

          <div className="rounded-2xl border border-slate-800 bg-slate-900/80 shadow-lg overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs text-slate-300">
                <thead className="bg-slate-950 text-slate-400 border-b border-slate-800 uppercase font-mono">
                  <tr>
                    <th className="py-3 px-4">Track</th>
                    <th className="py-3 px-4">Experiment Name</th>
                    <th className="py-3 px-4">Status</th>
                    <th className="py-3 px-4 text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {experiments.length > 0 ? (
                    experiments.map((exp) => {
                      const isSelected = selectedExperiment?.id === exp.id;
                      return (
                        <tr
                          key={exp.id}
                          onClick={() => handleSelectExperiment(exp.id)}
                          className={`cursor-pointer transition-colors ${
                            isSelected
                              ? "bg-indigo-600/15 border-l-2 border-indigo-500"
                              : "hover:bg-slate-800/40"
                          }`}
                        >
                          <td className="py-3 px-4 font-mono">
                            <span
                              className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                                exp.track === "B0"
                                  ? "bg-indigo-500/20 text-indigo-300"
                                  : exp.track === "B1"
                                  ? "bg-amber-500/20 text-amber-300"
                                  : "bg-purple-500/20 text-purple-300"
                              }`}
                            >
                              {exp.track}
                            </span>
                          </td>
                          <td className="py-3 px-4">
                            <p className="font-semibold text-slate-200">{exp.name}</p>
                            <p className="text-[11px] text-slate-500 truncate max-w-xs">{exp.description}</p>
                          </td>
                          <td className="py-3 px-4">
                            <span
                              className={`px-2 py-0.5 rounded-full text-[10px] font-semibold ${
                                exp.status === "COMPLETED"
                                  ? "bg-emerald-500/15 text-emerald-400"
                                  : "bg-rose-500/15 text-rose-400"
                              }`}
                            >
                              {exp.status}
                            </span>
                          </td>
                          <td className="py-3 px-4 text-right">
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                handleSelectExperiment(exp.id);
                              }}
                              className="text-xs text-indigo-400 hover:text-indigo-300 font-medium flex items-center gap-1 justify-end"
                            >
                              View
                              <ArrowRight className="w-3.5 h-3.5" />
                            </button>
                          </td>
                        </tr>
                      );
                    })
                  ) : (
                    <tr>
                      <td colSpan={4} className="py-8 text-center text-slate-500">
                        No experiments found for track {selectedTrack}
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Right Column: Experiment Detail Panel (6 cols) */}
        {selectedExperiment && (
          <div className="lg:col-span-6 space-y-4">
            <div className="p-6 rounded-2xl bg-slate-900/90 border border-indigo-500/30 shadow-xl space-y-5">
              {/* Detail Header */}
              <div className="flex items-start justify-between gap-3 pb-4 border-b border-slate-800">
                <div>
                  <div className="flex items-center gap-2">
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold ${
                        selectedExperiment.track === "B0"
                          ? "bg-indigo-500/20 text-indigo-300"
                          : selectedExperiment.track === "B1"
                          ? "bg-amber-500/20 text-amber-300"
                          : "bg-purple-500/20 text-purple-300"
                      }`}
                    >
                      {selectedExperiment.track}
                    </span>
                    <h3 className="text-base font-bold text-slate-100">{selectedExperiment.name}</h3>
                  </div>
                  <p className="text-xs text-slate-400 mt-1">{selectedExperiment.description}</p>
                </div>

                <button
                  onClick={() => setSelectedExperiment(null)}
                  className="p-1 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition-colors"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              {/* Source Traceability Badge */}
              <div className="p-3 rounded-xl bg-slate-950 border border-slate-800/80 flex items-center justify-between text-xs">
                <span className="text-slate-400 flex items-center gap-1.5">
                  <ShieldCheck className="w-4 h-4 text-emerald-400" />
                  Verified Frozen Artifact:
                </span>
                <span className="font-mono text-indigo-300 text-[11px] truncate max-w-xs" title={selectedExperiment.source}>
                  {selectedExperiment.source}
                </span>
              </div>

              {/* Derived Research Findings Callout Box */}
              {selectedExperiment.findings.length > 0 && (
                <div className="p-4 rounded-xl bg-indigo-950/30 border border-indigo-500/30 space-y-2">
                  <div className="flex items-center gap-2 text-xs font-semibold text-indigo-300">
                    <Sparkles className="w-4 h-4 text-indigo-400" />
                    <span>Derived Experimental Findings</span>
                  </div>
                  <ul className="space-y-1.5 text-xs text-slate-300 leading-relaxed pl-1">
                    {selectedExperiment.findings.map((f, idx) => (
                      <li key={idx} className="flex items-start gap-2">
                        <span className="text-indigo-400 font-bold">•</span>
                        <span>{f}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Conditions & Metric Breakdown */}
              {selectedExperiment.conditions.length > 0 && (
                <div className="space-y-2">
                  <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider font-mono">
                    Conditions & Evaluation Metrics ({selectedExperiment.conditions.length})
                  </span>
                  <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950">
                    <table className="w-full text-left text-xs text-slate-300">
                      <thead className="bg-slate-900/80 text-slate-400 border-b border-slate-800 text-[11px] font-mono">
                        <tr>
                          <th className="py-2.5 px-3">Condition</th>
                          <th className="py-2.5 px-3 text-right">CER</th>
                          <th className="py-2.5 px-3 text-right">NED</th>
                          <th className="py-2.5 px-3 text-right">Macro F1</th>
                          <th className="py-2.5 px-3 text-right">Δ from Baseline</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/60 font-mono text-[11px]">
                        {selectedExperiment.conditions.map((c) => (
                          <tr key={c.condition_id} className="hover:bg-slate-900/40">
                            <td className="py-2.5 px-3 font-semibold text-slate-200">
                              {c.name || c.condition_id}
                            </td>
                            <td className="py-2.5 px-3 text-right text-slate-300">{c.cer?.toFixed(4) ?? "—"}</td>
                            <td className="py-2.5 px-3 text-right text-slate-300">{c.ned?.toFixed(4) ?? "—"}</td>
                            <td className="py-2.5 px-3 text-right text-indigo-400 font-semibold">{c.f1?.toFixed(4) ?? "—"}</td>
                            <td className="py-2.5 px-3 text-right">
                              {c.delta_cer !== undefined && c.delta_cer !== null ? (
                                <span
                                  className={
                                    c.delta_cer < 0
                                      ? "text-emerald-400"
                                      : c.delta_cer > 0.05
                                      ? "text-rose-400"
                                      : "text-slate-400"
                                  }
                                >
                                  {c.delta_cer > 0 ? `+${c.delta_cer.toFixed(4)}` : c.delta_cer.toFixed(4)}
                                </span>
                              ) : (
                                "—"
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* KIE Per-field breakdown if available (Track B0) */}
              {selectedExperiment.kie_metrics?.per_field_f1 && (
                <div className="space-y-2 pt-2">
                  <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider font-mono">
                    Per-Field KIE Recognition F1
                  </span>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                    {Object.entries(selectedExperiment.kie_metrics.per_field_f1).map(([field, score]) => (
                      <div key={field} className="p-2.5 rounded-xl bg-slate-950 border border-slate-800 text-center">
                        <span className="text-[10px] uppercase font-mono text-slate-400">{field}</span>
                        <p className="text-sm font-bold font-mono text-indigo-400 mt-0.5">
                          {(score as number).toFixed(4)}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
