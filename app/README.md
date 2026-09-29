# Document AI Control Center

Production-ready Control Center platform for document intake, scan quality inspection, smart preprocessing, OCR text extraction, semantic Key Information Extraction (KIE), multi-tier confidence evaluation, human-in-the-loop manual review, analytics, and research diagnostics.

---

## 1. High-Level Architecture

The product is strictly decoupled from the research layer:
- **Research Layer** (`/repo/src`, `/repo/data`, `/repo/experiments`): read-only, containing empirical models and baseline benchmark records.
- **Product Layer** (`ocr/app`): independent, API-first full-stack system with Clean Architecture principles.

```text
DOCUMENT
   │
   ▼
Document Intake (MIME & Magic Bytes Inspection, Secure Storage)
   │
   ▼
Quality Analyzer (Laplacian Sharpness, RMS Contrast, Noise, Deskew Angle)
   │
   ▼
Smart Preprocessing (Adaptive Deskew, CLAHE, Denoise based on B1/B2 findings)
   │
   ▼
OCR Engine (RapidOCR PP-OCRv6 Small Models with [0.0, 1.0] Normalized BBoxes)
   │
   ▼
KIE Extraction (RuleBasedKIE SROIE with Token Provenance Mapping)
   │
   ▼
Confidence Engine (Token, Field, Document-level Scoring)
   │
   ▼
Decision Engine (Automation Policy & Quality Gate Check)
   │
   ├────────────────────────┐
   ▼                        ▼
AUTOMATIC PASS        MANUAL REVIEW
   │                        │
   └────────────┬───────────┘
                ▼
      Final Business Result
```

---

## 2. Directory Layout

```text
ocr/app/
├── backend/
│   ├── app/
│   │   ├── api/             # FastAPI REST endpoints (/api/v1/ and system)
│   │   ├── core/            # Config, Enums, JSON Logging, Security utilities
│   │   ├── db/              # SQLAlchemy async engine, session factory, base model
│   │   ├── models/          # Relational entities (Document, Page, Job, Field, Quality, Review, Audit)
│   │   ├── schemas/         # Normalized Pydantic v2 DTOs (0..1 coordinates)
│   │   ├── adapters/        # Clean Protocol bridges to RapidOCR, KIE, Quality, Preprocessing
│   │   ├── services/        # Storage, Confidence, Decision, Document, Pipeline, Review, Analytics
│   │   ├── workers/         # Background pipeline worker executing state-machine jobs
│   │   └── main.py          # FastAPI application entrypoint
│   ├── alembic/             # Database migrations
│   ├── tests/               # 15 Unit, integration, and real SROIE E2E tests
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── src/
│   │   ├── api/             # Typed API client
│   │   ├── components/      # AppShell, DocumentViewer (SVG overlays), DocumentTable, Badges
│   │   ├── pages/           # Dashboard, Documents, Inspector, Review, Analytics, Research, Settings
│   │   └── types/           # TypeScript schema definitions
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   ├── nginx.conf
│   └── Dockerfile
│
├── scripts/
│   ├── update.sh            # Safe self-update script for Linux/macOS
│   └── update.ps1           # Safe self-update script for Windows PowerShell
├── docker-compose.yml       # Production services: postgres, backend, worker, frontend
├── .env.example
└── README.md
```

---

## 3. Quick Start with Docker

### Prerequisites
- Docker Engine 24+ & Docker Compose v2.20+
- Host ports `3000` (frontend), `8000` (backend), `5432` (postgres) free

### Running the System
```bash
cd ocr/app

# 1. Copy environment variables
cp .env.example .env

# 2. Build and launch all containers
docker compose up --build -d

# 3. Check container status
docker compose ps
```

Once running:
- **Web UI Control Center**: [http://localhost:3000](http://localhost:3000)
- **FastAPI OpenAPI Swagger**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check Probe**: [http://localhost:8000/health](http://localhost:8000/health)
- **Readiness Check Probe**: [http://localhost:8000/ready](http://localhost:8000/ready)

---

## 4. Local Development (Without Docker)

### Backend Setup
```bash
cd ocr/app/backend

# Create virtual environment (Python 3.11)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run backend tests
python -m pytest tests/ -v

# Run FastAPI server
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# In a separate terminal, run background worker
python -m app.workers.pipeline_worker
```

### Frontend Setup
```bash
cd ocr/app/frontend

# Install dependencies
npm install

# Run dev server with API proxy
npm run dev

# Production build test
npm run build
```

---

## 5. Automated Self-Update Mechanism

The application includes an automated, zero-downtime update script (`update.sh` / `update.ps1`) that:
1. Creates an automated SQL dump backup of PostgreSQL data.
2. Pulls the latest commits from the current Git branch.
3. Builds updated container images.
4. Executes Alembic schema migrations.
5. Gracefully restarts containers.
6. Polls `/health` and `/ready` probes to verify system integrity. If health checks fail, alerts the operator.

### Running Self-Update
```bash
# On Linux / macOS
./scripts/update.sh

# On Windows PowerShell
.\scripts\update.ps1
```

---

## 6. End-to-End Pipeline & Idempotency

- Every document processing job follows an explicit Finite State Machine:
  `UPLOADED` $\to$ `PROCESSING` $\to$ `QUALITY_ANALYZED` $\to$ `PREPROCESSED` $\to$ `OCR_COMPLETED` $\to$ `KIE_COMPLETED` $\to$ `DECISION_MADE` $\to$ `COMPLETED_AUTOMATIC` or `MANUAL_REVIEW`.
- **Idempotency Guarantee**: If a job is retried after worker restart or failure, existing extracted fields for that document are cleared atomically in transaction before new ones are saved. No duplicate documents, pages, or audit records are created.
- **Append-Only Audit Trail**: Every action (upload, ocr, kie, operator field correction, approval, rejection) records an immutable entry in `audit_events`.
