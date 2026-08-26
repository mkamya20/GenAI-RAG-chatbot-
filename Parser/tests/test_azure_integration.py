"""
Integration tests for Azure OpenAI connectivity.

These tests hit the real Azure API and require valid credentials.
Skip in CI unless credentials are available.

Run manually:
    pytest tests/test_azure_integration.py -v

Or directly:
    python tests/test_azure_integration.py
"""

import sys
from pathlib import Path

# Add project root to path (for running as standalone script)
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import azure_client


# Skip all tests in this module if Azure not configured
pytestmark = pytest.mark.skipif(
    not azure_client.is_available(),
    reason="Azure OpenAI credentials not configured"
)


class TestAzureIntegration:
    """Integration tests that hit real Azure OpenAI API."""

    def test_client_creation(self):
        """Test Azure OpenAI client can be created."""
        client = azure_client.get_client()
        assert client is not None

    def test_single_embedding(self):
        """Test single embedding generation."""
        embedding = azure_client.generate_embedding("Hello, world!")
        
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

    def test_batch_embeddings(self):
        """Test batch embedding generation."""
        texts = ["Hello", "World", "Test"]
        embeddings = azure_client.generate_embeddings(texts, show_progress=False)
        
        assert len(embeddings) == 3
        assert all(len(e) == len(embeddings[0]) for e in embeddings)

    def test_chat_completion(self):
        """Test chat completion."""
        response = azure_client.chat_completion(
            messages=[
                {"role": "user", "content": "Say 'hello' and nothing else."}
            ],
            max_tokens=10
        )
        
        assert isinstance(response, str)
        assert len(response) > 0


# Allow running directly as a script
if __name__ == "__main__":
    print("Testing Azure OpenAI connection...\n")

    if not azure_client.is_available():
        print("✗ Azure OpenAI credentials not configured in .env")
        exit(1)

    print("✓ Azure OpenAI credentials configured")

    try:
        azure_client.get_client()
        print("✓ Azure OpenAI client created")
    except Exception as e:
        print(f"✗ Failed to create client: {e}")
        exit(1)

    try:
        embedding = azure_client.generate_embedding("Hello, world!")
        print(f"✓ Embedding generation successful (dimension: {len(embedding)})")
    except Exception as e:
        print(f"✗ Embedding generation failed: {e}")
        exit(1)

    try:
        texts = ["Hello", "World", "Test"]
        embeddings = azure_client.generate_embeddings(texts, show_progress=False)
        print(f"✓ Batch embedding successful ({len(embeddings)} embeddings)")
    except Exception as e:
        print(f"✗ Batch embedding failed: {e}")
        exit(1)

    try:
        response = azure_client.chat_completion(
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
            max_tokens=10
        )
        print(f"✓ Chat completion successful: {response}")
    except Exception as e:
        print(f"✗ Chat completion failed: {e}")
        exit(1)

    print("\n✓ All Azure OpenAI tests passed!")