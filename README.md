# ⬡ MEIS — Mineral Exploration Intelligence System

> A production-grade RAG (Retrieval-Augmented Generation) platform for mining engineers.  
> Upload geological survey PDFs, geochemical datasets, and satellite/hyperspectral imagery —  
> then query across all of it in plain language powered by Claude.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (HTML/JS)                        │
│  Upload UI · Query Chat · Collection Stats · Alteration Legend   │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP/REST
┌──────────────────────────▼──────────────────────────────────────┐
│                     FastAPI BACKEND                              │
│                                                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │ /api/ingest │  │ /api/query   │  │ /api/analysis          │  │
│  │             │  │              │  │                        │  │
│  │ PDF         │  │ RAG Engine   │  │ Mineral Zone Detector  │  │
│  │ Processor   │  │              │  │ Exploration Decision   │  │
│  │ Geochemical │  │ Retrieval    │  │ Support                │  │
│  │ Processor   │  │    +         │  └────────────────────────┘  │
│  │ Hyperspect. │  │ Generation   │                              │
│  │ Processor   │  │              │  ┌────────────────────────┐  │
│  └──────┬──────┘  └──────┬───────┘  │   Anthropic Claude     │  │
│         │                │          │   (LLM Client)         │  │
│  ┌──────▼──────┐         │          └────────────────────────┘  │
│  │  ChromaDB   │◄────────┘                                      │
│  │ Vector Store│                                                 │
│  │             │  sentence-transformers embeddings (local)       │
│  │ geological  │                                                 │
│  │ mineral     │                                                 │
│  │ hyperspect. │                                                 │
│  └─────────────┘                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key design decisions:**
- **Local embeddings** via `sentence-transformers` → no embedding API cost, runs offline
- **ChromaDB** for persistent vector storage → no external DB to manage
- **Three specialized collections** → domain-specific retrieval with cross-collection fallback
- **Modular processors** → each file type has its own processor class (Open/Closed Principle)
- **Streaming support** → real-time token streaming for the query endpoint

---

## Project Structure

```
mineral-exploration-rag/
│
├── backend/                    # FastAPI application
│   ├── main.py                 # App entry point, lifespan, CORS, routes
│   │
│   ├── core/                   # Core infrastructure
│   │   ├── config.py           # Pydantic-settings config (env vars)
│   │   ├── llm_client.py       # Anthropic Claude wrapper (query + streaming)
│   │   ├── rag_engine.py       # RAG pipeline orchestrator
│   │   └── vector_store_manager.py  # ChromaDB collections manager
│   │
│   ├── ingestion/              # Data ingestion processors
│   │   ├── pdf_processor.py         # PDF → chunks (PyMuPDF)
│   │   ├── mineral_dataset_processor.py  # CSV/JSON geochemical data
│   │   └── hyperspectral_processor.py    # Imagery + spectral analysis
│   │
│   ├── api/routes/             # FastAPI route handlers
│   │   ├── health.py           # GET /api/health
│   │   ├── ingest.py           # POST /api/ingest/upload, GET /api/ingest/stats
│   │   ├── query.py            # POST /api/query/
│   │   └── analysis.py         # POST /api/analysis/mineral-zones
│   │
│   ├── models/
│   │   └── schemas.py          # Pydantic request/response schemas
│   │
│   └── utils/
│       ├── logger.py           # Structured logging setup
│       └── text_chunker.py     # Sentence-aware text chunking
│
├── frontend/
│   └── index.html              # Single-file SPA (vanilla JS)
│
├── data/
│   ├── sample_reports/         # Place your PDF reports here
│   ├── mineral_datasets/       # Place your CSV/JSON datasets here
│   └── hyperspectral/          # Place your imagery here
│
├── scripts/
│   └── seed_sample_data.py     # Load synthetic data for instant testing
│
├── tests/
│   └── test_core.py            # Pytest test suite (no API key needed)
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── requirements.txt
├── .env.example                # → copy to .env and fill in your API key
├── pytest.ini
└── README.md
```

---

## Step 1 — Get Your Anthropic API Key

1. Go to **https://console.anthropic.com/**
2. Sign in or create a free account
3. Click **"API Keys"** in the left sidebar
4. Click **"Create Key"** → give it a name like `meis-dev`
5. **Copy the key immediately** — it's only shown once
6. It starts with `sk-ant-api03-...`

> **Cost note:** Claude Opus 4 is the most capable but costs more.  
> For development, use `claude-haiku-4-5-20251001` (cheapest, fastest) by changing  
> `ANTHROPIC_MODEL=claude-haiku-4-5-20251001` in your `.env` file.  
> Switch to `claude-opus-4-5` for production quality.

---

## Step 2 — Local Setup (Recommended for Development)

### Prerequisites
- Python **3.11+** → check with `python --version`
- pip → check with `pip --version`
- Git

### 2a. Clone & configure

```bash
git clone https://github.com/YOUR_USERNAME/mineral-exploration-rag.git
cd mineral-exploration-rag

# Copy the environment template
cp .env.example .env
```

Open `.env` in any editor and set your API key:
```
ANTHROPIC_API_KEY=sk-ant-api03-YOUR-KEY-HERE
```

### 2b. Create a virtual environment

```bash
# Create
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 2c. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on PyTorch:** The `torch` package in requirements.txt installs the CPU version.  
> If you have an NVIDIA GPU, install the CUDA version instead:  
> `pip install torch --index-url https://download.pytorch.org/whl/cu121`

> **First run:** `sentence-transformers` will download the `all-MiniLM-L6-v2` model  
> (~90 MB) automatically on first startup. This happens once.

### 2d. Seed sample data (optional but recommended)

This loads synthetic geological data so you can test immediately without uploading files:

```bash
cd backend
python ../scripts/seed_sample_data.py
```

You should see:
```
✅ Seeding complete! Total chunks in vector store: 8
   geological_reports: 5 chunks
   mineral_datasets: 2 chunks
   hyperspectral_data: 1 chunks
```

### 2e. Start the server

```bash
# From the backend/ directory
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 2f. Open the application

- **Frontend:** http://localhost:8000
- **API Docs (Swagger):** http://localhost:8000/api/docs
- **Health check:** http://localhost:8000/api/health

---

## Step 3 — Docker Setup (For Deployment)

```bash
# Build and start
cd docker
docker-compose up --build

# Run in background
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

The `.env` file at the project root is automatically loaded by Docker Compose.

---

## Step 4 — Run the Test Suite

Tests use mocks — no API key needed:

```bash
# From the project root
pytest tests/ -v
```

Expected output:
```
tests/test_core.py::TestHealth::test_health_check_returns_200 PASSED
tests/test_core.py::TestQuery::test_basic_query_returns_answer PASSED
tests/test_core.py::TestQuery::test_query_with_geological_type PASSED
tests/test_core.py::TestIngest::test_stats_endpoint PASSED
tests/test_core.py::TestAnalysis::test_mineral_zone_analysis PASSED
tests/test_core.py::TestTextChunker::test_long_text_produces_multiple_chunks PASSED
tests/test_core.py::TestMineralDatasetProcessor::test_csv_processing PASSED
...
```

---

## Usage Guide

### Uploading Files

| File Type | Category | What happens |
|-----------|----------|-------------|
| `.pdf` | Report | Text extracted page-by-page, cleaned, chunked |
| `.txt` | Report | Chunked directly |
| `.csv` | Dataset | Geochemical stats + anomaly analysis generated |
| `.json` | Dataset | Parsed as samples array, same as CSV |
| `.tif` / `.tiff` | Imagery | Band statistics extracted |
| `.png` / `.jpg` | Imagery | RGB analysis + iron oxide proxy |
| `.hdr` | Imagery | Full ENVI hyperspectral SAM analysis |

### Query Modes

| Mode | Best for |
|------|----------|
| `all` | General questions across all data types |
| `geological` | Rock formation, stratigraphy, alteration questions |
| `mineral` | Geochemical anomalies, grade, pathfinder elements |
| `hyperspectral` | Spectral mineral mapping, alteration zones |
| `decision` | "Should we drill here?" type questions |

### Example Queries

```
What minerals are associated with the main fault zone?

Identify potential Au-Cu porphyry targets from the geochemical data.

What alteration assemblage was detected in the hyperspectral imagery?

Compare the mineralization styles between the two prospect areas.

What are the highest-grade drill intercepts and where are they located?

Summarize the exploration recommendations across all available data.
```

### REST API Examples

```bash
# Query the knowledge base
curl -X POST http://localhost:8000/api/query/ \
  -H "Content-Type: application/json" \
  -d '{"query": "What gold grades were found near the fault?", "query_type": "all"}'

# Upload a PDF report
curl -X POST http://localhost:8000/api/ingest/upload \
  -F "file=@my_report.pdf" \
  -F "category=report"

# Mineral zone analysis
curl -X POST http://localhost:8000/api/analysis/mineral-zones \
  -H "Content-Type: application/json" \
  -d '{"data_summary": "Au 2.3ppm, As 840ppm, Sb 24ppm near NW fault", "include_report": true}'

# System stats
curl http://localhost:8000/api/ingest/stats
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ANTHROPIC_API_KEY not set` | Make sure `.env` exists with your key |
| `chromadb` errors on Windows | Run `pip install chromadb --upgrade` |
| `torch` install slow | Install CPU-only: `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| Port 8000 in use | Change `PORT=8001` in `.env` and restart |
| Empty search results | Run `python scripts/seed_sample_data.py` to load sample data |
| PDF extracts no text | File may be scanned — OCR support requires `pytesseract` |

---

## For Your Resume

**AI Engineer / ML Engineer description:**

> **Mineral Exploration Intelligence System** — Built a production RAG system for mining engineers integrating geological survey PDFs, geochemical datasets, and hyperspectral satellite imagery. Implemented modular ingestion pipeline with domain-specific processors (PyMuPDF, spectral analysis, geochemical anomaly detection), ChromaDB vector store with three domain collections, local sentence-transformers embeddings, and a FastAPI backend with streaming support. LLM layer uses Anthropic Claude with domain-expert system prompts for geological reasoning. Includes React-style SPA frontend, Docker deployment, and a test suite with dependency injection mocking.
>
> **Stack:** Python 3.11 · FastAPI · ChromaDB · sentence-transformers · Anthropic Claude API · PyMuPDF · Pandas · NumPy · Docker

**Key talking points in interviews:**
- Why local embeddings? Cost efficiency + offline capability for field environments
- Why three collections instead of one? Enables domain-specific retrieval tuning and targeted queries
- RAG vs fine-tuning tradeoff? RAG chosen for updateability — new reports ingestable without retraining
- How do you handle geochemical structured data in a text-based RAG? Dataset-to-text summarization with domain-specific statistics and anomaly narratives
- Chunking strategy? Sentence-aware with overlap, preserving decimal numbers in geological measurements

---

## License

MIT — see LICENSE file.
