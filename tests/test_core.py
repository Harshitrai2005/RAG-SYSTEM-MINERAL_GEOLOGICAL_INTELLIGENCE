"""
Test Suite — Core RAG & API Functionality

Run with:
    cd backend
    pytest ../tests/ -v

Tests use dependency injection overrides to mock the vector store and LLM,
so no API key is needed to run the test suite.
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_vector_store():
    """A minimal VectorStoreManager mock."""
    vs = MagicMock()
    vs.query.return_value = [
        {
            "id": "test_001",
            "text": "Gold mineralization at 2.3 g/t Au associated with the NW fault.",
            "metadata": {"source": "test_report.pdf", "doc_type": "geological_report", "page": 3},
            "similarity": 0.85,
            "collection": "geological_reports",
        }
    ]
    vs.multi_collection_query.return_value = vs.query.return_value
    vs.get_collection_stats.return_value = {
        "geological_reports": {"count": 5},
        "mineral_datasets": {"count": 3},
        "hyperspectral_data": {"count": 2},
    }
    vs.add_documents.return_value = 4
    return vs


@pytest.fixture
def mock_llm():
    """A minimal LLMClient mock that returns a canned geological answer."""
    llm = MagicMock()
    llm.query.return_value = {
        "answer": "Based on the retrieved context, the NW fault zone hosts gold mineralization at 2.3 g/t Au.",
        "usage": {"input_tokens": 200, "output_tokens": 80},
        "model": "claude-opus-4-5",
    }
    llm.analyze_mineral_zones.return_value = llm.query.return_value
    llm.generate_exploration_report.return_value = "Exploration Report: Target A is high priority."
    return llm


@pytest.fixture
def client(mock_vector_store, mock_llm):
    """FastAPI test client with mocked dependencies."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

    from main import app
    from api.routes.query import get_rag_engine
    from api.routes.ingest import get_vector_store
    from api.routes.analysis import get_rag_engine as get_rag_analysis
    from core.rag_engine import RAGEngine

    mock_rag = RAGEngine(vector_store=mock_vector_store, llm_client=mock_llm)

    app.dependency_overrides[get_rag_engine] = lambda: mock_rag
    app.dependency_overrides[get_rag_analysis] = lambda: mock_rag
    app.dependency_overrides[get_vector_store] = lambda: mock_vector_store

    # Inject mock vector store into app state
    app.state.vector_store_manager = mock_vector_store

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


# ── Health Tests ──────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_check_returns_200(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


# ── Query Tests ───────────────────────────────────────────────────────────────

class TestQuery:
    def test_basic_query_returns_answer(self, client):
        response = client.post(
            "/api/query/",
            json={"query": "What gold mineralization exists near the NW fault?", "query_type": "all"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert len(data["answer"]) > 10
        assert "sources" in data
        assert "chunks_retrieved" in data

    def test_query_with_geological_type(self, client):
        response = client.post(
            "/api/query/",
            json={"query": "Describe the rock formations", "query_type": "geological", "top_k": 3},
        )
        assert response.status_code == 200

    def test_query_with_decision_mode(self, client):
        response = client.post(
            "/api/query/",
            json={"query": "Should we drill the western target?", "query_type": "decision"},
        )
        assert response.status_code == 200

    def test_query_too_short_returns_422(self, client):
        response = client.post("/api/query/", json={"query": "Au"})
        assert response.status_code == 422

    def test_query_empty_returns_422(self, client):
        response = client.post("/api/query/", json={"query": ""})
        assert response.status_code == 422


# ── Ingest Tests ──────────────────────────────────────────────────────────────

class TestIngest:
    def test_stats_endpoint(self, client):
        response = client.get("/api/ingest/stats")
        assert response.status_code == 200
        data = response.json()
        assert "collections" in data
        assert "total_documents" in data

    def test_upload_invalid_extension(self, client, tmp_path):
        bad_file = tmp_path / "test.xyz"
        bad_file.write_text("some content")
        with open(bad_file, "rb") as f:
            response = client.post(
                "/api/ingest/upload",
                data={"category": "report"},
                files={"file": ("test.xyz", f, "application/octet-stream")},
            )
        assert response.status_code == 400

    def test_upload_valid_text_report(self, client, tmp_path, mock_vector_store):
        txt_file = tmp_path / "geo_report.txt"
        txt_file.write_text(
            "GEOLOGICAL REPORT\n"
            "The study area hosts gold mineralization in quartz veins. "
            "Assays returned 3.4 g/t Au over 2.5 m in trench TR-001. "
            "The host rock is a Cretaceous granodiorite with phyllic alteration. "
            "Structural controls include NW-trending faults intersecting NE splays. "
            "Recommendation: drill test the fault intersection at 200 m depth. " * 5
        )
        with open(txt_file, "rb") as f:
            response = client.post(
                "/api/ingest/upload",
                data={"category": "report"},
                files={"file": ("geo_report.txt", f, "text/plain")},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "chunks_added" in data


# ── Analysis Tests ────────────────────────────────────────────────────────────

class TestAnalysis:
    def test_mineral_zone_analysis(self, client):
        response = client.post(
            "/api/analysis/mineral-zones",
            json={
                "data_summary": "Soil sampling returned Au anomaly of 2.3 ppm coinciding "
                                "with As 840 ppm and Sb 24 ppm. Located adjacent to NW fault.",
                "include_report": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "analysis" in data
        assert "model" in data

    def test_mineral_zone_with_report(self, client):
        response = client.post(
            "/api/analysis/mineral-zones",
            json={
                "data_summary": "High-grade Au in quartz veins. Cu-Mo anomaly at depth.",
                "include_report": True,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("report") is not None

    def test_deposit_models_endpoint(self, client):
        response = client.get("/api/analysis/deposit-models")
        assert response.status_code == 200
        data = response.json()
        assert "deposit_models" in data
        assert "spectral_minerals" in data
        assert len(data["deposit_models"]) > 0


# ── Text Chunker Unit Tests ───────────────────────────────────────────────────

class TestTextChunker:
    def test_short_text_returns_single_chunk(self):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
        from utils.text_chunker import TextChunker
        chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
        text = "Gold mineralization at 2.3 g/t Au in quartz veins."
        chunks = chunker.split(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_produces_multiple_chunks(self):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
        from utils.text_chunker import TextChunker
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        text = "Gold is present. " * 50
        chunks = chunker.split(text)
        assert len(chunks) > 1

    def test_chunks_respect_max_size(self):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
        from utils.text_chunker import TextChunker
        chunker = TextChunker(chunk_size=200, chunk_overlap=30)
        text = "Copper porphyry deposit with chalcopyrite-bornite assemblage. " * 30
        chunks = chunker.split(text)
        for chunk in chunks:
            assert len(chunk) <= 250  # small tolerance for sentence boundary


# ── Geochemical Dataset Processor Tests ──────────────────────────────────────

class TestMineralDatasetProcessor:
    def test_csv_processing(self, tmp_path):
        import sys, os, csv
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
        from ingestion.mineral_dataset_processor import MineralDatasetProcessor

        csv_file = tmp_path / "assay_data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["SampleID", "Au", "Cu", "As", "Lithology"])
            writer.writeheader()
            writer.writerows([
                {"SampleID": "S001", "Au": "2.34", "Cu": "145", "As": "820", "Lithology": "Granodiorite"},
                {"SampleID": "S002", "Au": "0.12", "Cu": "45", "As": "42", "Lithology": "Schist"},
                {"SampleID": "S003", "Au": "0.98", "Cu": "312", "As": "380", "Lithology": "Granodiorite"},
            ])

        processor = MineralDatasetProcessor()
        chunks = processor.process_file(csv_file)

        assert len(chunks) > 0
        # Should have at least an overview chunk
        overview = next((c for c in chunks if "Dataset Overview" in c["metadata"].get("section", "")), None)
        assert overview is not None
        assert "Au" in overview["text"]

    def test_json_processing(self, tmp_path):
        import sys, os, json
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
        from ingestion.mineral_dataset_processor import MineralDatasetProcessor

        json_file = tmp_path / "mineral_data.json"
        with open(json_file, "w") as f:
            json.dump([
                {"Sample": "J001", "Au": 1.2, "Cu": 200, "As": 500},
                {"Sample": "J002", "Au": 0.05, "Cu": 45, "As": 18},
            ], f)

        processor = MineralDatasetProcessor()
        chunks = processor.process_file(json_file)
        assert len(chunks) > 0
