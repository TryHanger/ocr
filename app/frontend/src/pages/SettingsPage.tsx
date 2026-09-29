import React, { useEffect, useState } from "react";
import { apiSystem } from "../api/client";
import { Settings, Shield, Cpu, Database, CheckCircle2 } from "lucide-react";

export const SettingsPage: React.FC = () => {
  const [versionData, setVersionData] = useState<{
    version: string;
    app_name: string;
    environment: string;
  } | null>(null);
  const [healthStatus, setHealthStatus] = useState<string>("Checking...");

  useEffect(() => {
    const check = async () => {
      try {
        const v = await apiSystem.getVersion();
        setVersionData(v);
        const h = await apiSystem.getHealth();
        setHealthStatus(h.status);
      } catch (err) {
        setHealthStatus("Unreachable");
      }
    };
    check();
  }, []);

  return (
    <div className="space-y-8 max-w-4xl mx-auto">
      <div>
        <h2 className="text-base font-semibold text-slate-100 flex items-center gap-2">
          <Settings className="w-5 h-5 text-indigo-400" />
          System Settings & Policies
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          Review environment parameters, automation thresholds, and ML engine configurations.
        </p>
      </div>

      {/* System Status Section */}
      <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4">
        <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2">
          <Cpu className="w-4 h-4 text-indigo-400" />
          Runtime Environment
        </h3>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
          <div className="p-3 bg-slate-950 rounded-xl border border-slate-800">
            <span className="text-slate-500">Application</span>
            <p className="font-semibold text-slate-200 mt-1">
              {versionData?.app_name || "Document AI"}
            </p>
          </div>
          <div className="p-3 bg-slate-950 rounded-xl border border-slate-800">
            <span className="text-slate-500">Version</span>
            <p className="font-mono text-slate-200 mt-1">{versionData?.version || "0.1.0"}</p>
          </div>
          <div className="p-3 bg-slate-950 rounded-xl border border-slate-800">
            <span className="text-slate-500">Environment</span>
            <p className="font-mono text-slate-200 mt-1 capitalize">
              {versionData?.environment || "development"}
            </p>
          </div>
          <div className="p-3 bg-slate-950 rounded-xl border border-slate-800">
            <span className="text-slate-500">Health Probe</span>
            <p className="font-semibold text-emerald-400 mt-1 capitalize">{healthStatus}</p>
          </div>
        </div>
      </div>

      {/* ML Providers */}
      <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4">
        <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2">
          <Database className="w-4 h-4 text-indigo-400" />
          ML Adapters & Storage Driver
        </h3>
        <div className="space-y-3 text-xs">
          <div className="flex items-center justify-between p-3 bg-slate-950 rounded-xl border border-slate-800">
            <div>
              <p className="font-semibold text-slate-200">OCR Engine Provider</p>
              <p className="text-slate-500">RapidOCR PP-OCRv6 ONNX Runtime (CPUExecutionProvider default)</p>
            </div>
            <span className="px-2.5 py-1 bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded-lg font-mono">
              Active
            </span>
          </div>

          <div className="flex items-center justify-between p-3 bg-slate-950 rounded-xl border border-slate-800">
            <div>
              <p className="font-semibold text-slate-200">KIE Semantic Extractor</p>
              <p className="text-slate-500">Research RuleBasedKIE (SROIE Provenance & Tokens)</p>
            </div>
            <span className="px-2.5 py-1 bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded-lg font-mono">
              Active
            </span>
          </div>

          <div className="flex items-center justify-between p-3 bg-slate-950 rounded-xl border border-slate-800">
            <div>
              <p className="font-semibold text-slate-200">File Storage Driver</p>
              <p className="text-slate-500">LocalStorageDriver (/data/documents) with path sanitization</p>
            </div>
            <span className="px-2.5 py-1 bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 rounded-lg font-mono">
              Local FS
            </span>
          </div>
        </div>
      </div>

      {/* Automation Policy Thresholds */}
      <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4">
        <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2">
          <Shield className="w-4 h-4 text-indigo-400" />
          Automation Policy (Document Type: Receipt)
        </h3>
        <p className="text-xs text-slate-400">
          Documents with required fields above these confidence thresholds pass automatically without operator review.
        </p>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs pt-1">
          <div className="p-3 bg-slate-950 rounded-xl border border-slate-800">
            <span className="text-slate-400 font-medium">Company</span>
            <p className="text-lg font-bold font-mono text-emerald-400 mt-1">90%</p>
            <span className="text-[10px] text-slate-500">Required</span>
          </div>
          <div className="p-3 bg-slate-950 rounded-xl border border-slate-800">
            <span className="text-slate-400 font-medium">Date</span>
            <p className="text-lg font-bold font-mono text-emerald-400 mt-1">90%</p>
            <span className="text-[10px] text-slate-500">Required</span>
          </div>
          <div className="p-3 bg-slate-950 rounded-xl border border-slate-800">
            <span className="text-slate-400 font-medium">Address</span>
            <p className="text-lg font-bold font-mono text-emerald-400 mt-1">85%</p>
            <span className="text-[10px] text-slate-500">Optional</span>
          </div>
          <div className="p-3 bg-slate-950 rounded-xl border border-slate-800">
            <span className="text-slate-400 font-medium">Total</span>
            <p className="text-lg font-bold font-mono text-emerald-400 mt-1">95%</p>
            <span className="text-[10px] text-slate-500">Required + Valid</span>
          </div>
        </div>
      </div>
    </div>
  );
};
