"""
Shared pytest fixtures for all tests.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Test collection name (isolated from production)
TEST_COLLECTION_NAME = "test_collection"

# Fake embedding dimension (matches text-embedding-3-small)
EMBEDDING_DIM = 1536


def make_fake_embedding(text: str) -> list[float]:
    """Generate a deterministic fake embedding based on text hash."""
    hash_val = hash(text)
    return [(hash_val >> i & 0xFF) / 255.0 for i in range(EMBEDDING_DIM)]


@pytest.fixture(autouse=True)
def reset_clients_after_test():
    """Reset all client singletons after each test."""
    yield
    
    from vector_store import reset_client as reset_vector_client
    from azure_client import reset_client as reset_azure_client
    
    reset_vector_client()
    reset_azure_client()


@pytest.fixture
def mock_embeddings():
    """Mock Azure OpenAI embedding calls."""
    with patch("azure_client.generate_embedding") as mock_single, \
         patch("azure_client.generate_embeddings") as mock_batch:
        
        mock_single.side_effect = lambda text: make_fake_embedding(text)
        mock_batch.side_effect = lambda texts, **kwargs: [
            make_fake_embedding(t) for t in texts
        ]
        
        yield {
            "single": mock_single,
            "batch": mock_batch,
        }


@pytest.fixture
def test_collection_name():
    """Return isolated test collection name."""
    return TEST_COLLECTION_NAME


@pytest.fixture
def cleanup_test_collection():
    """Clean up test collection after test."""
    yield TEST_COLLECTION_NAME
    
    try:
        from vector_store import get_client
        client = get_client()
        client.delete_collection(TEST_COLLECTION_NAME)
    except Exception:
        pass  # Collection may not exist


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing vector store."""
    return [
        {
            "id": "test_chunk_1",
            "text": "Gravitational waves are ripples in spacetime.",
            "embedding": make_fake_embedding("Gravitational waves are ripples in spacetime."),
            "metadata": {
                "filename": "test_doc.pdf",
                "page_numbers": "[1]",
                "title": "Test Document",
                "chunk_index": "0",
                "source_type": "pdf",
            }
        },
        {
            "id": "test_chunk_2",
            "text": "LIGO detects gravitational waves using laser interferometry.",
            "embedding": make_fake_embedding("LIGO detects gravitational waves using laser interferometry."),
            "metadata": {
                "filename": "test_doc.pdf",
                "page_numbers": "[1]",
                "title": "Test Document",
                "chunk_index": "1",
                "source_type": "pdf",
            }
        },
        {
            "id": "test_chunk_3",
            "text": "Black hole mergers produce strong gravitational wave signals.",
            "embedding": make_fake_embedding("Black hole mergers produce strong gravitational wave signals."),
            "metadata": {
                "filename": "another_doc.pdf",
                "page_numbers": "[5]",
                "title": "Another Document",
                "chunk_index": "0",
                "source_type": "pdf",
            }
        },
    ]


@pytest.fixture
def sample_pdf_text():
    """Sample text that might come from a PDF."""
    return """Introduction to Gravitational Waves

Gravitational waves are ripples in the fabric of spacetime caused by 
accelerating massive objects. They were first predicted by Albert Einstein 
in 1916 as part of his general theory of relativity.

Detection Methods

The Laser Interferometer Gravitational-Wave Observatory (LIGO) uses 
laser interferometry to detect these tiny distortions in spacetime. 
The detector arms are 4 kilometers long and can measure changes smaller 
than a proton's width.
"""