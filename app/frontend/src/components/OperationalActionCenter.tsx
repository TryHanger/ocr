import React from "react";
import {
  DocumentOperationsResponse,
  OperationalAction,
  OperationalIssue,
  IssueSeverity,
  ResolutionStatus,
} from "../types";
import {
  AlertOctagon,
  AlertTriangle,
  AlertCircle,
  Info,
  CheckCircle2,
  Clock,
  ArrowRight,
  ExternalLink,
  RotateCw,
  Eye,
  Edit3,
  ShieldAlert,
  FlaskConical,
  Activity,
  Layers,
  Database,
  CheckCircle,
} from "lucide-react";

interface OperationalActionCenterProps {
  operations: DocumentOperationsResponse | null;
  isLoading?: boolean;
  error?: string | null;
  onActionClick: (action: OperationalAction) => void;
  onRefresh?: () => void;
}

export const OperationalActionCenter: React.FC<OperationalActionCenterProps> = ({
  operations,
  isLoading = false,
  error = null,
  onActionClick,
  onRefresh,
}) => {
  if (isLoading) {
    return (
      <div className="p-4 rounded-2xl bg-slate-900/90 border border-slate-800 text-xs flex items-center justify-center gap-3 py-8 text-slate-400">
        <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-indigo-500"></div>
        <span>Evaluating operational issues and available actions...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 rounded-2xl bg-rose-950/40 border border-rose-800/60 text-xs text-rose-300 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <AlertOctagon className="w-4 h-4 text-rose-400 shrink-0" />
          <span>Failed to load operational guidance: {error}</span>
        </div>
        {onRefresh && (
          <button
            onClick={onRefresh}
            className="px-2.5 py-1 rounded-lg bg-rose-900/60 hover:bg-rose-800/80 text-rose-200 border border-rose-700/60 transition-colors"
          >
            Retry
          </button>
        )}
      </div>
    );
  }

  if (!operations) {
    return null;
  }

  const { overall_status, total_issues, blocking_issues, issues, summary } = operations;

  const renderStatusBadge = (status: ResolutionStatus) => {
    switch (status) {
      case "open":
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full font-bold uppercase tracking-wider text-[11px] bg-amber-950/80 text-amber-300 border border-amber-800/70">
            <Clock className="w-3.5 h-3.5" />
            Open (Pending Review)
          </span>
        );
      case "in_review":
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full font-bold uppercase tracking-wider text-[11px] bg-blue-950/80 text-blue-300 border border-blue-800/70">
            <Edit3 className="w-3.5 h-3.5" />
            In Review (Underway)
          </span>
        );
      case "resolved":
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full font-bold uppercase tracking-wider text-[11px] bg-emerald-950/80 text-emerald-300 border border-emerald-800/70">
            <CheckCircle2 className="w-3.5 h-3.5" />
            Resolved
          </span>
        );
    }
  };

  const getSeverityBadge = (severity: IssueSeverity) => {
    switch (severity) {
      case "critical":
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-rose-950/90 text-rose-300 border border-rose-700/70">
            <AlertOctagon className="w-3 h-3 text-rose-400" />
            Critical
          </span>
        );
      case "high":
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-rose-950/60 text-rose-300 border border-rose-800/50">
            <AlertTriangle className="w-3 h-3 text-rose-400" />
            High
          </span>
        );
      case "medium":
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-amber-950/70 text-amber-300 border border-amber-800/50">
            <AlertCircle className="w-3 h-3 text-amber-400" />
            Medium
          </span>
        );
      case "low":
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-slate-800 text-slate-300 border border-slate-700">
            <Info className="w-3 h-3 text-slate-400" />
            Low
          </span>
        );
      case "info":
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-emerald-950/70 text-emerald-300 border border-emerald-800/50">
            <CheckCircle className="w-3 h-3 text-emerald-400" />
            Info
          </span>
        );
    }
  };

  const getActionIcon = (type: string) => {
    switch (type) {
      case "open_field":
        return <Edit3 className="w-3.5 h-3.5" />;
      case "open_quality_evidence":
        return <Activity className="w-3.5 h-3.5" />;
      case "open_validation":
        return <ShieldAlert className="w-3.5 h-3.5" />;
      case "open_research_evidence":
        return <FlaskConical className="w-3.5 h-3.5" />;
      case "open_review":
        return <CheckCircle2 className="w-3.5 h-3.5" />;
      case "retry_processing":
        return <RotateCw className="w-3.5 h-3.5" />;
      default:
        return <Eye className="w-3.5 h-3.5" />;
    }
  };

  return (
    <div className="p-4 rounded-2xl bg-slate-900/90 border border-slate-800 text-xs shadow-lg space-y-4">
      {/* 1. Header Bar: Title, Resolution Status, and Counts */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-800/80">
        <div className="flex items-center gap-2.5 flex-wrap">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-indigo-400" />
            <span className="font-semibold text-slate-200 tracking-wide uppercase text-[11px]">
              Operational Action & Resolution Center
            </span>
          </div>

          {renderStatusBadge(overall_status)}

          {blocking_issues > 0 && (
            <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-rose-950/70 text-rose-300 border border-rose-800/50">
              {blocking_issues} blocking issue(s)
            </span>
          )}
        </div>

        <div className="flex items-center gap-2 text-[11px] text-slate-400">
          <span className="font-mono">{summary}</span>
        </div>
      </div>

      {/* 2. Empty State (Zero Issues) */}
      {issues.length === 0 ? (
        <div className="py-6 px-4 rounded-xl bg-slate-950/50 border border-slate-800/60 text-center space-y-2">
          <div className="w-8 h-8 rounded-full bg-emerald-950/80 border border-emerald-700/60 text-emerald-400 flex items-center justify-center mx-auto">
            <CheckCircle2 className="w-5 h-5" />
          </div>
          <div className="text-slate-200 font-semibold text-xs">
            Zero Operational Issues Detected
          </div>
          <div className="text-slate-400 text-[11px] max-w-lg mx-auto">
            Document met all automated quality, extraction confidence, and domain validation criteria.
            No manual operator intervention or remediation is required.
          </div>
        </div>
      ) : (
        /* 3. Operational Issues List */
        <div className="space-y-3">
          {issues.map((iss) => (
            <div
              key={iss.id}
              className={`p-3.5 rounded-xl border transition-all space-y-3 ${
                iss.severity === "critical" || iss.severity === "high"
                  ? "bg-slate-950/90 border-rose-900/40"
                  : iss.severity === "medium"
                  ? "bg-slate-950/90 border-amber-900/40"
                  : "bg-slate-950/70 border-slate-800"
              }`}
            >
              {/* Issue Header: Severity, Code, Title, Field */}
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2 flex-wrap">
                  {getSeverityBadge(iss.severity)}
                  <span className="font-mono text-[11px] font-semibold text-slate-200">
                    {iss.code}
                  </span>
                  <span className="text-slate-400 text-[11px]">·</span>
                  <span className="text-slate-200 font-medium text-[11px]">{iss.title}</span>
                </div>

                <div className="flex items-center gap-2">
                  {iss.field && (
                    <span className="px-2 py-0.5 rounded bg-indigo-950/70 text-indigo-300 border border-indigo-800/50 text-[10px] font-mono">
                      Field: <strong className="text-indigo-200">{iss.field}</strong>
                    </span>
                  )}
                  <span className="text-[10px] text-slate-500 font-mono">
                    Source: {iss.source}
                  </span>
                </div>
              </div>

              {/* Description & Observed vs Threshold */}
              <div className="text-slate-300 text-[11px] leading-relaxed">
                {iss.description}
                {(iss.observed_value !== null && iss.observed_value !== undefined) &&
                  (iss.threshold !== null && iss.threshold !== undefined) && (
                    <div className="mt-1.5 inline-flex items-center gap-3 px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-800 font-mono text-[10px]">
                      <span className="text-slate-400">
                        Observed: <strong className="text-rose-300">{iss.observed_value}</strong>
                      </span>
                      <span className="text-slate-600">|</span>
                      <span className="text-slate-400">
                        Configured Threshold: <strong className="text-emerald-300">{iss.threshold}</strong>
                      </span>
                    </div>
                  )}
              </div>

              {/* Tri-Layer Evidence Container (Factual Only) */}
              {(iss.evidence.production || iss.evidence.historical || iss.evidence.research) && (
                <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800/80 space-y-1.5 text-[11px]">
                  <div className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                    <Database className="w-3 h-3 text-slate-500" />
                    Supporting Evidence
                  </div>

                  {iss.evidence.production && (
                    <div className="flex items-start gap-2 text-slate-300">
                      <span className="text-[10px] font-mono uppercase text-indigo-400 font-semibold shrink-0 w-20">
                        Production:
                      </span>
                      <span className="text-slate-300 text-[11px]">{iss.evidence.production}</span>
                    </div>
                  )}

                  {iss.evidence.historical && (
                    <div className="flex items-start gap-2 text-slate-300">
                      <span className="text-[10px] font-mono uppercase text-emerald-400 font-semibold shrink-0 w-20">
                        Historical:
                      </span>
                      <span className="text-slate-300 text-[11px]">{iss.evidence.historical}</span>
                    </div>
                  )}

                  {iss.evidence.research && (
                    <div className="flex items-start gap-2 text-slate-300">
                      <span className="text-[10px] font-mono uppercase text-purple-400 font-semibold shrink-0 w-20">
                        Research:
                      </span>
                      <span className="text-slate-300 text-[11px]">{iss.evidence.research}</span>
                    </div>
                  )}
                </div>
              )}

              {/* Available Actions Bar */}
              {iss.available_actions.length > 0 && (
                <div className="pt-2 border-t border-slate-800/60 flex flex-wrap items-center gap-2">
                  <span className="text-[10px] text-slate-500 font-semibold uppercase mr-1">
                    Actions:
                  </span>
                  {iss.available_actions.map((act) => (
                    <button
                      key={act.id}
                      onClick={() => onActionClick(act)}
                      disabled={!act.enabled}
                      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium transition-all active:scale-95 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 ${
                        act.type === "open_review"
                          ? "bg-indigo-600 hover:bg-indigo-500 text-white shadow-sm shadow-indigo-600/20"
                          : act.type === "open_field"
                          ? "bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700"
                          : act.type === "open_research_evidence"
                          ? "bg-purple-950/70 hover:bg-purple-900/80 text-purple-300 border border-purple-800/60"
                          : act.type === "retry_processing"
                          ? "bg-amber-950/60 hover:bg-amber-900/80 text-amber-300 border border-amber-800/60"
                          : "bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700"
                      }`}
                      title={act.description}
                      aria-label={act.label}
                    >
                      {getActionIcon(act.type)}
                      <span>{act.label}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
