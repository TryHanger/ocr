import React, { useEffect, useState } from "react";
import { DocumentItem } from "../types";
import { apiDocuments } from "../api/client";
import { DocumentTable } from "../components/DocumentTable";
import { Search, Filter, RefreshCw } from "lucide-react";

interface DocumentsPageProps {
  onInspect: (id: string) => void;
  onReview: (id: string) => void;
}

export const DocumentsPage: React.FC<DocumentsPageProps> = ({ onInspect, onReview }) => {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [total, setTotal] = useState(0);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState<string>("");
  const [typeFilter, setTypeFilter] = useState<string>("");
  const [isLoading, setIsLoading] = useState(true);
  const [page, setPage] = useState(1);
  const limit = 20;

  const loadDocuments = async () => {
    try {
      const res = await apiDocuments.list({
        search: search || undefined,
        status: statusFilter || undefined,
        document_type: typeFilter || undefined,
        limit,
        offset: (page - 1) * limit,
      });
      setDocuments(res.items);
      setTotal(res.total);
    } catch (err) {
      console.error("Failed to load documents", err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadDocuments();
    // Poll every 2 seconds to update in-flight processing jobs
    const interval = setInterval(loadDocuments, 2000);
    return () => clearInterval(interval);
  }, [search, statusFilter, typeFilter, page]);

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Search and Filters Bar */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4 bg-slate-900/60 p-4 rounded-2xl border border-slate-800">
        <div className="relative w-full sm:w-80">
          <Search className="w-4 h-4 absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-500" />
          <input
            type="text"
            placeholder="Search by filename..."
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setPage(1);
            }}
            className="w-full pl-9 pr-4 py-2 bg-slate-950 border border-slate-800 rounded-xl text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500 transition-colors"
          />
        </div>

        {/* Filter Pills */}
        <div className="flex items-center gap-2 overflow-x-auto w-full sm:w-auto">
          {[
            { id: "", label: "All Statuses" },
            { id: "completed_automatic", label: "Automatic" },
            { id: "manual_review", label: "Manual Review" },
            { id: "processing", label: "Processing" },
            { id: "error", label: "Errors" },
          ].map((item) => (
            <button
              key={item.id}
              onClick={() => {
                setStatusFilter(item.id);
                setPage(1);
              }}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition-colors ${
                statusFilter === item.id
                  ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/20"
                  : "bg-slate-800/70 text-slate-400 hover:text-slate-200"
              }`}
            >
              {item.label}
            </button>
          ))}

          <button
            onClick={() => loadDocuments()}
            className="p-2 text-slate-400 hover:text-slate-200 hover:bg-slate-800 rounded-lg transition-colors ml-1"
            title="Refresh list"
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Main Table */}
      <DocumentTable
        documents={documents}
        onInspect={onInspect}
        onReview={onReview}
        onRetry={async (id) => {
          await apiDocuments.retry(id);
          loadDocuments();
        }}
        isLoading={isLoading}
      />

      {/* Pagination Controls */}
      {total > limit && (
        <div className="flex items-center justify-between text-xs text-slate-400 px-2">
          <span>
            Showing {(page - 1) * limit + 1} to {Math.min(page * limit, total)} of {total} documents
          </span>
          <div className="flex gap-2">
            <button
              disabled={page <= 1}
              onClick={() => setPage((p) => Math.max(p - 1, 1))}
              className="px-3 py-1.5 bg-slate-900 border border-slate-800 rounded-lg disabled:opacity-40 disabled:cursor-not-allowed hover:bg-slate-800 transition-colors"
            >
              Previous
            </button>
            <button
              disabled={page * limit >= total}
              onClick={() => setPage((p) => p + 1)}
              className="px-3 py-1.5 bg-slate-900 border border-slate-800 rounded-lg disabled:opacity-40 disabled:cursor-not-allowed hover:bg-slate-800 transition-colors"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
