import React from "react";
import { DocumentItem } from "../types";
import { StatusBadge } from "./StatusBadge";
import { ConfidenceBadge } from "./ConfidenceBadge";
import { Eye, Edit3, RotateCw, FileText } from "lucide-react";

interface DocumentTableProps {
  documents: DocumentItem[];
  onInspect: (id: string) => void;
  onReview: (id: string) => void;
  onRetry: (id: string) => void;
  isLoading?: boolean;
}

export const DocumentTable: React.FC<DocumentTableProps> = ({
  documents,
  onInspect,
  onReview,
  onRetry,
  isLoading = false,
}) => {
  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-12 text-slate-400">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-500"></div>
      </div>
    );
  }

  if (documents.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center p-12 text-slate-500 border border-dashed border-slate-800 rounded-xl">
        <FileText className="w-10 h-10 mb-2 stroke-1" />
        <p className="text-sm font-medium">No documents found matching criteria</p>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-900/60 shadow-lg">
      <table className="w-full text-left text-sm text-slate-300">
        <thead className="bg-slate-950/80 text-xs uppercase tracking-wider text-slate-400 border-b border-slate-800">
          <tr>
            <th className="py-3 px-4">Document</th>
            <th className="py-3 px-4">Type</th>
            <th className="py-3 px-4">Status</th>
            <th className="py-3 px-4">Confidence</th>
            <th className="py-3 px-4">Processing Time</th>
            <th className="py-3 px-4">Uploaded</th>
            <th className="py-3 px-4 text-right">Actions</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800/60 font-sans">
          {documents.map((doc) => (
            <tr
              key={doc.id}
              className="hover:bg-slate-800/40 transition-colors group cursor-pointer"
              onClick={() => onInspect(doc.id)}
            >
              <td className="py-3.5 px-4 font-medium text-slate-100 flex items-center gap-2">
                <FileText className="w-4 h-4 text-indigo-400 shrink-0" />
                <span className="truncate max-w-xs">{doc.filename}</span>
              </td>
              <td className="py-3.5 px-4 capitalize text-slate-400 text-xs">
                {doc.document_type}
              </td>
              <td className="py-3.5 px-4">
                <StatusBadge status={doc.status} size="sm" />
              </td>
              <td className="py-3.5 px-4">
                <ConfidenceBadge confidence={doc.confidence} size="sm" />
              </td>
              <td className="py-3.5 px-4 text-xs font-mono text-slate-400">
                {doc.processing_duration_ms ? `${doc.processing_duration_ms} ms` : "—"}
              </td>
              <td className="py-3.5 px-4 text-xs text-slate-400">
                {new Date(doc.created_at).toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                  second: "2-digit",
                })}
              </td>
              <td className="py-3.5 px-4 text-right" onClick={(e) => e.stopPropagation()}>
                <div className="flex items-center justify-end gap-1.5">
                  <button
                    onClick={() => onInspect(doc.id)}
                    className="p-1.5 text-slate-400 hover:text-slate-100 hover:bg-slate-800 rounded transition-colors"
                    title="Inspect Document"
                  >
                    <Eye className="w-4 h-4" />
                  </button>
                  {doc.status === "manual_review" && (
                    <button
                      onClick={() => onReview(doc.id)}
                      className="px-2.5 py-1 text-xs font-medium bg-amber-500/20 text-amber-300 hover:bg-amber-500/30 rounded border border-amber-500/30 flex items-center gap-1 transition-colors"
                      title="Review Extracted Fields"
                    >
                      <Edit3 className="w-3.5 h-3.5" />
                      Review
                    </button>
                  )}
                  {(doc.status === "error" || doc.status === "manual_review") && (
                    <button
                      onClick={() => onRetry(doc.id)}
                      className="p-1.5 text-slate-400 hover:text-indigo-400 hover:bg-slate-800 rounded transition-colors"
                      title="Retry Processing"
                    >
                      <RotateCw className="w-4 h-4" />
                    </button>
                  )}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};
