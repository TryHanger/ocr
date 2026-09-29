import React from "react";
import {
  LayoutDashboard,
  FileStack,
  CheckSquare,
  BarChart3,
  FlaskConical,
  Sliders,
  Sparkles,
  Settings,
  Upload,
  Cpu,
  RotateCcw,
} from "lucide-react";

export type NavTab = "dashboard" | "documents" | "review" | "analytics" | "improvement" | "research" | "degradations" | "preprocessing" | "settings";


interface AppShellProps {
  currentTab: NavTab;
  onSelectTab: (tab: NavTab) => void;
  onOpenUpload: () => void;
  manualReviewCount?: number;
  appVersion?: string;
  children: React.ReactNode;
}

export const AppShell: React.FC<AppShellProps> = ({
  currentTab,
  onSelectTab,
  onOpenUpload,
  manualReviewCount = 0,
  appVersion = "0.1.0",
  children,
}) => {
  const navItems = [
    { id: "dashboard" as NavTab, label: "Control Center", icon: LayoutDashboard },
    { id: "documents" as NavTab, label: "Documents", icon: FileStack },
    {
      id: "review" as NavTab,
      label: "Manual Review",
      icon: CheckSquare,
      badge: manualReviewCount > 0 ? manualReviewCount : undefined,
    },
    { id: "analytics" as NavTab, label: "Analytics", icon: BarChart3 },
    { id: "research" as NavTab, label: "Research Mode", icon: FlaskConical },
    { id: "settings" as NavTab, label: "Settings", icon: Settings },
  ];

  return (
    <div className="flex h-screen bg-slate-950 text-slate-100 antialiased overflow-hidden">
      {/* Sidebar */}
      <aside className="w-64 bg-slate-900/90 border-r border-slate-800 flex flex-col shrink-0">
        {/* Brand Header */}
        <div className="h-16 flex items-center px-6 border-b border-slate-800 gap-3">
          <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center text-white shadow-md shadow-indigo-600/30">
            <Cpu className="w-5 h-5" />
          </div>
          <div>
            <h1 className="font-semibold text-sm leading-tight text-slate-100">Document AI</h1>
            <p className="text-[11px] text-slate-400 font-mono">Control Center</p>
          </div>
        </div>

        {/* Navigation Items */}
        {/* Navigation Items */}
        <nav className="flex-1 p-3 space-y-4 overflow-y-auto">
          {/* Operations Section */}
          <div className="space-y-1">
            <span className="px-3 text-[10px] font-bold uppercase tracking-wider text-slate-500">
              Operations
            </span>
            {[
              { id: "dashboard" as NavTab, label: "Control Center", icon: LayoutDashboard },
              { id: "documents" as NavTab, label: "Documents", icon: FileStack },
              {
                id: "review" as NavTab,
                label: "Manual Review",
                icon: CheckSquare,
                badge: manualReviewCount > 0 ? manualReviewCount : undefined,
              },
              { id: "analytics" as NavTab, label: "Operations Analytics", icon: BarChart3 },
            ].map((item) => {
              const Icon = item.icon;
              const isActive = currentTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => onSelectTab(item.id)}
                  className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs font-medium transition-colors ${
                    isActive
                      ? "bg-indigo-600/15 text-indigo-400 border border-indigo-500/20"
                      : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
                  }`}
                >
                  <div className="flex items-center gap-2.5">
                    <Icon className={`w-4 h-4 ${isActive ? "text-indigo-400" : "text-slate-400"}`} />
                    <span>{item.label}</span>
                  </div>
                  {item.badge !== undefined && (
                    <span className="px-1.5 py-0.5 text-[10px] font-semibold rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/30">
                      {item.badge}
                    </span>
                  )}
                </button>
              );
            })}
          </div>

          {/* Continuous Improvement Section */}
          <div className="space-y-1">
            <span className="px-3 text-[10px] font-bold uppercase tracking-wider text-slate-500">
              Improvement
            </span>
            {[
              { id: "improvement" as NavTab, label: "Improvement Loop", icon: RotateCcw, badge: "MVP-10" },
            ].map((item) => {
              const Icon = item.icon;
              const isActive = currentTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => onSelectTab(item.id)}
                  className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs font-medium transition-colors ${
                    isActive
                      ? "bg-indigo-600/15 text-indigo-400 border border-indigo-500/20"
                      : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
                  }`}
                >
                  <div className="flex items-center gap-2.5">
                    <Icon className={`w-4 h-4 ${isActive ? "text-indigo-400" : "text-slate-400"}`} />
                    <span>{item.label}</span>
                  </div>
                  <span className="px-1.5 py-0.5 text-[9px] font-mono rounded bg-purple-500/20 text-purple-300 border border-purple-500/30 uppercase">
                    {item.badge}
                  </span>
                </button>
              );
            })}
          </div>

          {/* Research Section */}
          <div className="space-y-1">
            <span className="px-3 text-[10px] font-bold uppercase tracking-wider text-slate-500">
              Research
            </span>
            {[
              { id: "research" as NavTab, label: "Research Overview", icon: FlaskConical, badge: "B0–B2" },
              { id: "degradations" as NavTab, label: "Degradation Explorer", icon: Sliders, badge: "B1" },
              { id: "preprocessing" as NavTab, label: "Preprocessing Explorer", icon: Sparkles, badge: "B2" },
            ].map((item) => {
              const Icon = item.icon;
              const isActive = currentTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => onSelectTab(item.id)}
                  className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs font-medium transition-colors ${
                    isActive
                      ? "bg-indigo-600/15 text-indigo-400 border border-indigo-500/20"
                      : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
                  }`}
                >
                  <div className="flex items-center gap-2.5">
                    <Icon className={`w-4 h-4 ${isActive ? "text-indigo-400" : "text-slate-400"}`} />
                    <span>{item.label}</span>
                  </div>
                  <span className="px-1.5 py-0.5 text-[9px] font-mono rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 uppercase">
                    {item.badge}
                  </span>
                </button>
              );
            })}
          </div>

          {/* System Section */}
          <div className="space-y-1">
            <span className="px-3 text-[10px] font-bold uppercase tracking-wider text-slate-500">
              System
            </span>
            {[
              { id: "settings" as NavTab, label: "Settings", icon: Settings },
            ].map((item) => {
              const Icon = item.icon;
              const isActive = currentTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => onSelectTab(item.id)}
                  className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs font-medium transition-colors ${
                    isActive
                      ? "bg-indigo-600/15 text-indigo-400 border border-indigo-500/20"
                      : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
                  }`}
                >
                  <div className="flex items-center gap-2.5">
                    <Icon className={`w-4 h-4 ${isActive ? "text-indigo-400" : "text-slate-400"}`} />
                    <span>{item.label}</span>
                  </div>
                </button>
              );
            })}
          </div>
        </nav>

        {/* Sidebar Footer */}
        <div className="p-4 border-t border-slate-800 text-xs text-slate-500 flex items-center justify-between">
          <span>Version {appVersion}</span>
          <span className="inline-flex items-center gap-1 text-emerald-400">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            Online
          </span>
        </div>
      </aside>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top Header */}
        <header className="h-16 bg-slate-900/60 border-b border-slate-800 px-8 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-2">
            <h2 className="text-base font-semibold text-slate-200 capitalize">
              {currentTab === "dashboard" ? "Control Center" : currentTab.replace("_", " ")}
            </h2>
          </div>

          <div className="flex items-center gap-4">
            <button
              onClick={onOpenUpload}
              className="flex items-center gap-2 px-3.5 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-medium transition-all shadow-md shadow-indigo-600/20 active:scale-95"
            >
              <Upload className="w-3.5 h-3.5" />
              Upload Document
            </button>
          </div>
        </header>

        {/* Content Body */}
        <main className="flex-1 overflow-y-auto p-8">{children}</main>
      </div>
    </div>
  );
};
