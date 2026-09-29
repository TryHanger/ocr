import React, { useEffect, useState } from "react";
import { AppShell, NavTab } from "./components/AppShell";
import { DashboardPage } from "./pages/DashboardPage";
import { DocumentsPage } from "./pages/DocumentsPage";
import { DocumentInspectorPage } from "./pages/DocumentInspectorPage";
import { ManualReviewPage } from "./pages/ManualReviewPage";
import { AnalyticsPage } from "./pages/AnalyticsPage";
import { ResearchPage } from "./pages/ResearchPage";
import { DegradationExplorerPage } from "./pages/DegradationExplorerPage";
import { PreprocessingExplorerPage } from "./pages/PreprocessingExplorerPage";
import { SettingsPage } from "./pages/SettingsPage";
import { UploadModal } from "./components/UploadModal";
import { apiAnalytics, apiSystem } from "./api/client";

export const App: React.FC = () => {
  const [currentTab, setCurrentTab] = useState<NavTab>("dashboard");
  const [inspectedDocId, setInspectedDocId] = useState<string | null>(null);
  const [researchDegradation, setResearchDegradation] = useState<string | null>(null);
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [manualReviewCount, setManualReviewCount] = useState(0);
  const [appVersion, setAppVersion] = useState("0.1.0");

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const tabParam = params.get("tab") as NavTab | null;
    const degParam = params.get("degradation");
    if (degParam) {
      setResearchDegradation(degParam);
    }
    if (
      tabParam &&
      [
        "dashboard",
        "documents",
        "review",
        "analytics",
        "research",
        "degradations",
        "preprocessing",
        "settings",
      ].includes(tabParam)
    ) {
      setCurrentTab(tabParam);
    } else if (degParam) {
      setCurrentTab("degradations");
    }
  }, []);

  const refreshCounts = async () => {
    try {
      const overview = await apiAnalytics.getOverview();
      setManualReviewCount(overview.kpis.manual_review_count);
    } catch {
      // ignore
    }
  };

  useEffect(() => {
    refreshCounts();
    const interval = setInterval(refreshCounts, 4000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    apiSystem.getVersion().then((v) => setAppVersion(v.version)).catch(() => {});
  }, []);

  const handleInspect = (id: string) => {
    setInspectedDocId(id);
  };

  const handleReview = (id: string) => {
    setInspectedDocId(null);
    setCurrentTab("review");
  };

  const handleNavigateToResearch = (track: "B1" | "B2", degradationCode: string) => {
    setInspectedDocId(null);
    setResearchDegradation(degradationCode);
    const targetTab: NavTab = track === "B1" ? "degradations" : "preprocessing";
    setCurrentTab(targetTab);

    // Sync browser address bar with deep-link query parameter
    try {
      const url = new URL(window.location.href);
      url.searchParams.set("degradation", degradationCode);
      url.searchParams.set("tab", targetTab);
      window.history.pushState({}, "", url.toString());
    } catch {
      // ignore in non-browser environments
    }
  };

  const handleUploadSuccess = (docId: string) => {
    setInspectedDocId(docId);
    refreshCounts();
  };

  return (
    <AppShell
      currentTab={currentTab}
      onSelectTab={(tab) => {
        setInspectedDocId(null);
        setCurrentTab(tab);
      }}
      onOpenUpload={() => setIsUploadOpen(true)}
      manualReviewCount={manualReviewCount}
      appVersion={appVersion}
    >
      {inspectedDocId ? (
        <DocumentInspectorPage
          documentId={inspectedDocId}
          onBack={() => setInspectedDocId(null)}
          onNavigateToResearch={handleNavigateToResearch}
        />
      ) : (
        <>
          {currentTab === "dashboard" && (
            <DashboardPage
              onInspect={handleInspect}
              onReview={handleReview}
              onViewAllDocuments={() => setCurrentTab("documents")}
              onNavigateToReview={() => setCurrentTab("review")}
              onNavigateToAnalytics={() => setCurrentTab("analytics")}
              onNavigateToResearch={() => setCurrentTab("research")}
            />
          )}

          {currentTab === "documents" && (
            <DocumentsPage onInspect={handleInspect} onReview={handleReview} />
          )}

          {currentTab === "review" && <ManualReviewPage onInspect={handleInspect} />}

          {currentTab === "analytics" && <AnalyticsPage />}

          {currentTab === "research" && <ResearchPage />}

          {currentTab === "degradations" && (
            <DegradationExplorerPage initialDegradation={researchDegradation || undefined} />
          )}

          {currentTab === "preprocessing" && (
            <PreprocessingExplorerPage initialDegradation={researchDegradation || undefined} />
          )}

          {currentTab === "settings" && <SettingsPage />}
        </>
      )}

      <UploadModal
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
        onSuccess={handleUploadSuccess}
      />
    </AppShell>
  );
};

export default App;
