"""
Azure OpenAI client wrapper for embeddings and chat completions.
"""

from openai import AzureOpenAI

from config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    EMBEDDING_BATCH_SIZE,
)
from logging_config import get_logger

logger = get_logger(__name__)


class _AzureClientHolder:
    """Holds the Azure OpenAI client singleton. Use get_client() to access."""
    
    def __init__(self):
        self.client: AzureOpenAI | None = None
    
    def get(self) -> AzureOpenAI:
        if self.client is None:
            self.client = AzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION,
                azure_endpoint=AZURE_OPENAI_ENDPOINT
            )
            logger.debug("Azure OpenAI client initialized")
        return self.client
    
    def reset(self) -> None:
        """Reset the client. Primarily for testing."""
        self.client = None


_AZURE_HOLDER = _AzureClientHolder()


def get_client() -> AzureOpenAI:
    """
    Get the Azure OpenAI client singleton.

    Returns:
        AzureOpenAI client instance
    """
    return _AZURE_HOLDER.get()


def reset_client() -> None:
    """
    Reset the Azure OpenAI client singleton.
    
    Primarily for testing to ensure isolation between tests.
    """
    _AZURE_HOLDER.reset()


def is_available() -> bool:
    """
    Check if Azure OpenAI is configured and available.

    Returns:
        True if credentials are configured
    """
    return bool(AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT)


def generate_embedding(text: str) -> list[float]:
    """
    Generate embedding for a single text string.

    Args:
        text: Text to embed

    Returns:
        Embedding vector
    """
    client = get_client()
    response = client.embeddings.create(
        model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        input=text
    )
    return response.data[0].embedding


def generate_embeddings(
    texts: list[str],
    batch_size: int = EMBEDDING_BATCH_SIZE,
    show_progress: bool = True
) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.

    Args:
        texts: List of text strings to embed
        batch_size: Number of texts per API call
        show_progress: Whether to log progress

    Returns:
        List of embedding vectors
    """
    client = get_client()
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            input=batch
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

        if show_progress:
            logger.info(f"Embedded batch {i // batch_size + 1}/{total_batches}")

    return all_embeddings


def chat_completion(
    messages: list[dict],
    max_tokens: int = 800
) -> str:
    """
    Generate a chat completion.

    Args:
        messages: List of message dicts with 'role' and 'content'
        max_tokens: Maximum tokens in response

    Returns:
        Assistant's response text
    """
    client = get_client()
    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=messages,
        max_completion_tokens=max_tokens
    )
    return response.choices[0].message.content