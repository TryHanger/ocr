import React from "react";
import { DocumentStatus } from "../types";
import { CheckCircle2, AlertTriangle, Loader2, XCircle, Clock } from "lucide-react";

interface StatusBadgeProps {
  status: DocumentStatus | string;
  size?: "sm" | "md";
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status, size = "md" }) => {
  const isSm = size === "sm";

  switch (status) {
    case "completed_automatic":
      return (
        <span
          className={`inline-flex items-center gap-1.5 font-medium rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 ${
            isSm ? "px-2 py-0.5 text-xs" : "px-2.5 py-1 text-xs"
          }`}
        >
          <CheckCircle2 className={isSm ? "w-3 h-3" : "w-3.5 h-3.5"} />
          Automatic
        </span>
      );

    case "completed_manual":
      return (
        <span
          className={`inline-flex items-center gap-1.5 font-medium rounded-full bg-teal-500/10 text-teal-400 border border-teal-500/20 ${
            isSm ? "px-2 py-0.5 text-xs" : "px-2.5 py-1 text-xs"
          }`}
        >
          <CheckCircle2 className={isSm ? "w-3 h-3" : "w-3.5 h-3.5"} />
          Completed (Manual)
        </span>
      );

    case "manual_review":
      return (
        <span
          className={`inline-flex items-center gap-1.5 font-medium rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20 ${
            isSm ? "px-2 py-0.5 text-xs" : "px-2.5 py-1 text-xs"
          }`}
        >
          <AlertTriangle className={isSm ? "w-3 h-3" : "w-3.5 h-3.5"} />
          Manual Review
        </span>
      );

    case "error":
      return (
        <span
          className={`inline-flex items-center gap-1.5 font-medium rounded-full bg-rose-500/10 text-rose-400 border border-rose-500/20 ${
            isSm ? "px-2 py-0.5 text-xs" : "px-2.5 py-1 text-xs"
          }`}
        >
          <XCircle className={isSm ? "w-3 h-3" : "w-3.5 h-3.5"} />
          Error
        </span>
      );

    case "uploaded":
      return (
        <span
          className={`inline-flex items-center gap-1.5 font-medium rounded-full bg-slate-500/10 text-slate-400 border border-slate-500/20 ${
            isSm ? "px-2 py-0.5 text-xs" : "px-2.5 py-1 text-xs"
          }`}
        >
          <Clock className={isSm ? "w-3 h-3" : "w-3.5 h-3.5"} />
          Uploaded
        </span>
      );

    default:
      // In-flight processing states
      return (
        <span
          className={`inline-flex items-center gap-1.5 font-medium rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 ${
            isSm ? "px-2 py-0.5 text-xs" : "px-2.5 py-1 text-xs"
          }`}
        >
          <Loader2 className={`${isSm ? "w-3 h-3" : "w-3.5 h-3.5"} animate-spin`} />
          {status.replace("_", " ")}
        </span>
      );
  }
};
