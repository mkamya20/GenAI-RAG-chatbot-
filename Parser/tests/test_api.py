"""
Tests for FastAPI endpoints.

Uses TestClient for synchronous testing of async endpoints.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from app import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_vector_store():
    """Mock vector store operations."""
    with patch("routers.health.count_chunks") as mock_count, \
         patch("routers.chat.retrieve_chunks") as mock_retrieve, \
         patch("routers.pdfs.get_all_documents") as mock_docs:
        
        mock_count.return_value = 100
        mock_docs.return_value = {
            "test.pdf": {"chunk_count": 50, "pages": {1, 2, 3}},
            "other.pdf": {"chunk_count": 50, "pages": {1, 2}},
        }
        mock_retrieve.return_value = [
            {
                "id": "chunk_1",
                "text": "Sample text about gravitational waves.",
                "metadata": {
                    "filename": "test.pdf",
                    "page_numbers": [1],
                    "title": "Test Doc",
                },
                "chunk_index": 0,
                "distance": 0.5,
            }
        ]
        
        yield {
            "retrieve": mock_retrieve,
            "count": mock_count,
            "docs": mock_docs,
        }


@pytest.fixture
def mock_azure():
    """Mock Azure OpenAI client."""
    with patch("routers.health.azure_client") as mock_health, \
         patch("routers.chat.azure_client") as mock_chat:
        mock_health.is_available.return_value = True
        mock_chat.is_available.return_value = True
        mock_chat.chat_completion.return_value = "This is a test response about gravitational waves."
        yield mock_chat


class TestHealthEndpoint:
    """Tests for /api/health endpoint."""

    def test_health_returns_ok(self, client, mock_vector_store, mock_azure):
        response = client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "chromadb_available" in data
        assert "azure_openai_available" in data
        assert "total_chunks" in data

    def test_health_shows_chunk_count(self, client, mock_vector_store, mock_azure):
        response = client.get("/api/health")
        
        data = response.json()
        assert data["total_chunks"] == 100


class TestApiInfoEndpoint:
    """Tests for /api endpoint."""

    def test_returns_api_info(self, client):
        response = client.get("/api")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data


class TestSearchEndpoint:
    """Tests for /api/search endpoint."""

    def test_search_returns_results(self, client, mock_vector_store):
        response = client.post("/api/search", json={
            "query": "gravitational waves",
            "top_k": 5
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "total" in data
        assert data["total"] >= 0

    def test_search_respects_top_k(self, client, mock_vector_store):
        response = client.post("/api/search", json={
            "query": "test query",
            "top_k": 3
        })
        
        assert response.status_code == 200
        mock_vector_store["retrieve"].assert_called_once()
        call_kwargs = mock_vector_store["retrieve"].call_args
        assert call_kwargs[1]["top_k"] == 3

    def test_search_with_filename_filter(self, client, mock_vector_store):
        response = client.post("/api/search", json={
            "query": "test",
            "top_k": 5,
            "filter_filename": "specific.pdf"
        })
        
        assert response.status_code == 200


class TestChatEndpoint:
    """Tests for /api/chat endpoint."""

    def test_chat_returns_answer(self, client, mock_vector_store, mock_azure):
        response = client.post("/api/chat", json={
            "query": "What are gravitational waves?",
            "top_k": 5,
            "use_rag": True
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "chunks_used" in data

    def test_chat_without_rag(self, client, mock_vector_store, mock_azure):
        response = client.post("/api/chat", json={
            "query": "What are gravitational waves?",
            "top_k": 5,
            "use_rag": False
        })
        
        assert response.status_code == 200


class TestPdfsEndpoint:
    """Tests for /api/pdfs endpoints (read-only)."""

    def test_list_pdfs(self, client, mock_vector_store):
        response = client.get("/api/pdfs")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_pdf_info(self, client, mock_vector_store):
        response = client.get("/api/pdfs/test.pdf")
        
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "test.pdf"
        assert "chunk_count" in data

    def test_get_nonexistent_pdf(self, client, mock_vector_store):
        mock_vector_store["docs"].return_value = {}
        
        response = client.get("/api/pdfs/nonexistent.pdf")
        
        assert response.status_code == 404

    def test_list_pdfs_filters_non_pdfs(self, client, mock_vector_store):
        """Verify that only .pdf files are returned, not talk posts."""
        mock_vector_store["docs"].return_value = {
            "paper.pdf": {"chunk_count": 10, "pages": {1, 2}},
            "talk_post_123": {"chunk_count": 5, "pages": set()},
        }
        
        response = client.get("/api/pdfs")
        
        assert response.status_code == 200
        data = response.json()
        filenames = [doc["filename"] for doc in data]
        assert "paper.pdf" in filenames
        assert "talk_post_123" not in filenames