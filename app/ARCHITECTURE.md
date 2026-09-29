# Document AI Control Center — System Architecture

## 1. Context & Architectural Boundaries

The system maintains a strict separation between experimental science and operational software:

```text
┌────────────────────────────────────────────────────────┐
│            RESEARCH REPOSITORY (READ-ONLY)            │
│  src/ (RapidOCR, RuleBasedKIE, Preprocessing P1..P5)   │
│  data/SROIE2019/ (Canonical validation receipts)       │
│  experiments/runs/ (B0, B1, B2 empirical results)      │
└──────────────────────────┬─────────────────────────────┘
                           │ Dynamic Adapters (Read-Only)
                           ▼
┌────────────────────────────────────────────────────────┐
│           PRODUCT LAYER (ocr/app/ - INDEPENDENT)       │
│                                                        │
│  FastAPI (REST API v1) ──┐                             │
│                          ├──> PostgreSQL (Asyncpg)     │
│  Pipeline Worker ────────┘                             │
│                          └──> LocalStorage (/data)     │
│                                                        │
│  React SPA (Vite + Tailwind + SVG Overlay Canvas)      │
└────────────────────────────────────────────────────────┘
```

---

## 2. Core Subsystems

### 2.1 Storage Layer (`app.services.storage`)
- Defined via `FileStorage` protocol.
- `LocalStorageDriver` provides safe file handling, preventing directory traversal attacks via absolute resolution checks.
- Structured storage path: `/data/documents/{yyyy}/{mm}/{uuid}/{filename}`.

### 2.2 ML / OCR / KIE Adapter Subsystem (`app.adapters`)
- Uses Python `Protocol` definitions (`OCRProvider`, `KIEProvider`, `QualityAnalyzer`, `PreprocessingProvider`, `ResearchMetricsProvider`).
- `ResearchRapidOCRAdapter`: Loads PP-OCRv6 ONNX models in worker thread pool, normalizes token bounding boxes to relative float range $[0.0, 1.0]$.
- `ResearchRuleBasedKIEAdapter`: Executes SROIE semantic rules for `company`, `date`, `address`, `total`, and computes union bounding boxes from token provenance.
- `QualityAnalysisAdapter`: Evaluates Laplacian variance (blur), standard deviation (contrast), difference of medians (noise), and Hough line angle (skew).
- `ResearchPreprocessingAdapter`: Applies adaptive deskew, CLAHE, and denoising in accordance with B1/B2 findings.
- `DiskResearchMetricsAdapter`: Directly ingests real B0, B1, and B2 JSON/CSV reports from `experiments/runs/`.

### 2.3 Confidence & Decision Engines (`app.services`)
- **`ConfidenceEngine`**:
  - Token-level OCR confidence.
  - Field-level KIE confidence.
  - Holistic document confidence: $0.60 \times \text{Fields} + 0.25 \times \text{Tokens} + 0.15 \times \text{Quality}$.
- **`DecisionEngine`**:
  - Enforces document type automation policy (Receipt):
    - `company >= 0.90` (Required)
    - `date >= 0.90` (Required)
    - `total >= 0.95` (Required, validated numeric)
    - `address >= 0.85` (Optional)
    - `quality_score >= 0.50` (Quality Gate)
  - Routes to `AUTOMATIC` or `MANUAL_REVIEW` with structured root cause (`ReviewReason`).

### 2.4 State Machine & Idempotency
- Discrete, controlled lifecycle:
  `UPLOADED` $\to$ `PROCESSING` $\to$ `QUALITY_ANALYZED` $\to$ `PREPROCESSED` $\to$ `OCR_COMPLETED` $\to$ `KIE_COMPLETED` $\to$ `DECISION_MADE` $\to$ `COMPLETED_AUTOMATIC` or `MANUAL_REVIEW`.
- Re-executing a job clears existing extracted fields in transaction before persisting new extractions.
- Append-only `AuditEvent` table preserves full audit history.

---

## 3. Frontend Architecture

- **React 18 + TypeScript + Vite + Tailwind CSS**.
- **Responsive Layout (`AppShell`)**:
  - Control Center Dashboard (KPIs, stream, root causes).
  - Documents Queue (filtering, pagination, search).
  - Manual Review Station (split review workflow).
  - Document Inspector (tabbed detail, zoomable document viewer with SVG overlays).
  - Analytics (visual aggregations).
  - Research Mode (live benchmarks from B0/B1/B2).
  - Settings (system metadata and threshold inspection).
- **DocumentViewer**:
  - Scales $[0.0, 1.0]$ normalized bounding box coordinates into percentage-based SVG elements (`left: x1 * 100%`, `top: y1 * 100%`).
  - Supports interactive field cross-highlighting on hover/click.
