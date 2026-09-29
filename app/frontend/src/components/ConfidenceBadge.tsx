import React from "react";

interface ConfidenceBadgeProps {
  confidence?: number | null;
  size?: "sm" | "md";
}

export const ConfidenceBadge: React.FC<ConfidenceBadgeProps> = ({ confidence, size = "md" }) => {
  if (confidence === undefined || confidence === null) {
    return <span className="text-slate-500 text-xs">—</span>;
  }

  const val = Math.round(confidence);
  let colorCls = "bg-rose-500/10 text-rose-400 border-rose-500/20";
  if (val >= 90) {
    colorCls = "bg-emerald-500/10 text-emerald-400 border-emerald-500/20";
  } else if (val >= 70) {
    colorCls = "bg-amber-500/10 text-amber-400 border-amber-500/20";
  }

  const isSm = size === "sm";

  return (
    <span
      className={`inline-flex items-center font-mono font-medium rounded border ${colorCls} ${
        isSm ? "px-1.5 py-0.5 text-xs" : "px-2 py-0.5 text-xs"
      }`}
    >
      {val}%
    </span>
  );
};
