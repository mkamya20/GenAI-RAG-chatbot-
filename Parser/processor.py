"""
Shared processing utilities for chunking, embedding, and storage.

This module provides orchestration between text processing, embedding generation,
and vector storage. It serves as the bridge between source-specific processors
(pdf_processor, talk_post_processor) and the underlying services (azure_client,
vector_store).
"""

import time

from langchain_text_splitters import RecursiveCharacterTextSplitter

from azure_client import generate_embeddings
from vector_store import store_chunks
from config import DEFAULT_BATCH_SIZE
from logging_config import get_logger

logger = get_logger(__name__)

# Default separators for text splitting, ordered by preference
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def split_into_chunks(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str] | None = None
) -> list[str]:
    """
    Split text into overlapping chunks using recursive character splitting.

    Args:
        text: Text to split
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        separators: List of separators to split on, in order of preference.
                   Defaults to paragraph breaks, line breaks, sentences, words.

    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=separators or DEFAULT_SEPARATORS
    )
    return splitter.split_text(text)


def embed_and_store_batched(
    chunks: list[dict],
    collection_name: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    delay: float = 0.0
) -> int:
    """
    Generate embeddings and store chunks in batches.

    Processes chunks in batches to manage memory and API rate limits.
    Each batch generates embeddings via Azure OpenAI and stores results
    in ChromaDB.

    Args:
        chunks: List of chunk dicts, each with 'id', 'text', 'metadata' keys.
                Embeddings will be added in place.
        collection_name: ChromaDB collection name (uses default if None)
        batch_size: Number of chunks to process per batch
        delay: Seconds to wait between batches (helps avoid rate limiting)

    Returns:
        Total number of chunks stored
    """
    if not chunks:
        return 0

    total_batches = (len(chunks) + batch_size - 1) // batch_size
    total_stored = 0

    for batch_num, i in enumerate(range(0, len(chunks), batch_size), start=1):
        batch = chunks[i:i + batch_size]
        total_stored += _process_batch(batch, collection_name, batch_num, total_batches)
        
        # Delay between batches (skip after last batch)
        if delay > 0 and batch_num < total_batches:
            logger.debug(f"Waiting {delay}s before next batch...")
            time.sleep(delay)

    return total_stored


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Generate embeddings for chunks without storing.

    Useful when you need to embed chunks but handle storage separately.

    Args:
        chunks: List of chunk dicts with 'text' key

    Returns:
        Same chunks with 'embedding' key added
    """
    if not chunks:
        return chunks

    texts = [chunk["text"] for chunk in chunks]
    embeddings = generate_embeddings(texts, show_progress=False)

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    return chunks


def _process_batch(
    batch: list[dict],
    collection_name: str | None,
    batch_num: int,
    total_batches: int
) -> int:
    """
    Process a single batch: generate embeddings and store.

    Args:
        batch: List of chunk dicts to process
        collection_name: ChromaDB collection name
        batch_num: Current batch number (1-indexed)
        total_batches: Total number of batches

    Returns:
        Number of chunks stored
    """
    if not batch:
        return 0

    texts = [chunk["text"] for chunk in batch]

    logger.info(
        f"Batch {batch_num}/{total_batches}: "
        f"Generating embeddings for {len(texts)} chunks..."
    )
    embeddings = generate_embeddings(texts, show_progress=False)

    for chunk, embedding in zip(batch, embeddings):
        chunk["embedding"] = embedding

    store_chunks(batch, collection_name=collection_name)
    logger.info(f"Batch {batch_num}/{total_batches}: Stored {len(batch)} chunks")

    return len(batch)