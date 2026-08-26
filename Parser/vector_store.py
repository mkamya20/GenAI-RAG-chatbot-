"""
ChromaDB vector store operations.
"""

import ast

import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError

from config import (
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_NAME,
    DEFAULT_TOP_K,
    EMBEDDING_BATCH_SIZE,
)
from azure_client import generate_embedding
from logging_config import get_logger

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Client Management
# -----------------------------------------------------------------------------

class _ChromaClientHolder:
    """Holds the ChromaDB client singleton. Use get_client() to access."""
    
    def __init__(self):
        self.client: chromadb.PersistentClient | None = None
    
    def get(self) -> chromadb.PersistentClient:
        if self.client is None:
            self.client = chromadb.PersistentClient(
                path=CHROMA_DB_PATH,
                settings=Settings(anonymized_telemetry=False)
            )
            logger.debug(f"ChromaDB client initialized at {CHROMA_DB_PATH}")
        return self.client
    
    def reset(self) -> None:
        """Reset the client. Primarily for testing."""
        self.client = None


_CHROMA_HOLDER = _ChromaClientHolder()


def get_client() -> chromadb.PersistentClient:
    """
    Get the ChromaDB client singleton.

    Returns:
        ChromaDB client instance
    """
    return _CHROMA_HOLDER.get()


def reset_client() -> None:
    """
    Reset the ChromaDB client singleton.
    
    Primarily for testing to ensure isolation between tests.
    """
    _CHROMA_HOLDER.reset()


def get_or_create_collection(collection_name: str | None = None) -> chromadb.Collection:
    """
    Get or create a ChromaDB collection.

    Args:
        collection_name: Name of the collection (uses config default if None)

    Returns:
        ChromaDB collection
    """
    name = collection_name or CHROMA_COLLECTION_NAME
    client = get_client()
    try:
        collection = client.get_collection(name=name)
        logger.debug(f"Using existing collection: {name}")
    except NotFoundError:
        collection = client.create_collection(name=name)
        logger.info(f"Created new collection: {name}")
    return collection


# -----------------------------------------------------------------------------
# Store Operations
# -----------------------------------------------------------------------------

def store_chunks(
    chunks: list[dict],
    collection_name: str | None = None,
    batch_size: int = EMBEDDING_BATCH_SIZE
) -> int:
    """
    Store chunks with embeddings in ChromaDB.

    Args:
        chunks: List of chunk dicts with 'id', 'text', 'embedding', 'metadata'
        collection_name: Collection name (uses config default if None)
        batch_size: Number of chunks per batch

    Returns:
        Number of chunks stored
    """
    if not chunks:
        return 0

    collection = get_or_create_collection(collection_name)

    ids = [chunk["id"] for chunk in chunks]
    texts = [chunk["text"] for chunk in chunks]
    embeddings = [chunk["embedding"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    total_batches = (len(ids) + batch_size - 1) // batch_size

    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i + batch_size]
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]
        batch_metadatas = metadatas[i:i + batch_size]

        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_texts,
            metadatas=batch_metadatas
        )
        logger.info(f"Stored batch {i // batch_size + 1}/{total_batches}")

    return len(chunks)


# -----------------------------------------------------------------------------
# Retrieve Operations
# -----------------------------------------------------------------------------

def retrieve_chunks(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    collection_name: str | None = None,
    pdf_slots: int | None = None,
    wiki_slots: int | None = None,
) -> list[dict]:
    """
    Retrieve top-k most similar chunks for a query.

    Uses multi-stage retrieval to ensure both authoritative source types
    (PDFs and wiki pages) are always represented. Reserves dedicated slots
    for each, then backfills any unused slots from remaining source types.

    Args:
        query: Search query text
        top_k: Number of results to return
        collection_name: Collection name (uses config default if None)
        pdf_slots: Slots reserved for PDF results.
                   Defaults to ceil(top_k * 0.6), e.g. 3 of 5.
        wiki_slots: Slots reserved for wiki results.
                    Defaults to top_k - pdf_slots, e.g. 2 of 5.

    Returns:
        List of chunk dicts sorted by similarity (best first)
    """
    if pdf_slots is None:
        pdf_slots = -(-top_k * 3 // 5)  # ceil(top_k * 0.6)
    if wiki_slots is None:
        wiki_slots = top_k - pdf_slots

    query_embedding = generate_embedding(query)
    collection = get_or_create_collection(collection_name)

    # Stage 1: retrieve from PDFs
    pdf_chunks = _query_by_source(collection, query_embedding, "pdf", pdf_slots)

    # Stage 2: retrieve from wiki pages
    wiki_chunks = _query_by_source(collection, query_embedding, "wiki", wiki_slots)

    # Backfill: if either pool returned fewer than its slots, give the
    # surplus to the other pool so we still aim for top_k total
    filled = len(pdf_chunks) + len(wiki_chunks)
    backfill_needed = top_k - filled

    backfill_chunks = []
    if backfill_needed > 0:
        # Try the pool that had capacity (fewer results than slots)
        seen_ids = {c["id"] for c in pdf_chunks + wiki_chunks}

        if len(pdf_chunks) < pdf_slots:
            # Wiki had more room; pull extra wiki
            extra = _query_by_source(
                collection, query_embedding, "wiki",
                wiki_slots + backfill_needed
            )
            backfill_chunks = [c for c in extra if c["id"] not in seen_ids]
        else:
            # Pull extra PDFs
            extra = _query_by_source(
                collection, query_embedding, "pdf",
                pdf_slots + backfill_needed
            )
            backfill_chunks = [c for c in extra if c["id"] not in seen_ids]

    # Merge and sort by distance (lower = more similar)
    merged = pdf_chunks + wiki_chunks + backfill_chunks
    merged.sort(key=lambda c: c["distance"] if c["distance"] is not None else float("inf"))

    logger.debug(
        f"Retrieved {len(merged)} chunks for query "
        f"(pdf={len(pdf_chunks)}, wiki={len(wiki_chunks)}, "
        f"backfill={len(backfill_chunks)})"
    )
    return merged[:top_k]


def _query_by_source(
    collection: chromadb.Collection,
    query_embedding: list[float],
    source_type: str,
    n_results: int,
) -> list[dict]:
    """
    Query a collection filtered to a single source type.

    Args:
        collection: ChromaDB collection
        query_embedding: Pre-computed query embedding
        source_type: Value to filter on (e.g. 'pdf', 'wiki', 'talk_post')
        n_results: Maximum number of results

    Returns:
        List of chunk dicts, may be shorter than n_results
    """
    if n_results <= 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"source_type": source_type}
    )
    return _unpack_query_results(results)


def get_all_documents() -> dict[str, dict]:
    """
    Get summary info for all documents in the collection.

    Returns:
        Dict mapping filename to {'chunk_count': int, 'pages': set}
    """
    collection = get_or_create_collection()
    results = collection.get()

    docs = {}
    for i, chunk_id in enumerate(results['ids']):
        metadata = results['metadatas'][i] if results['metadatas'] else {}
        filename = metadata.get('filename', 'Unknown')

        if filename not in docs:
            docs[filename] = {
                'chunk_count': 0,
                'pages': set()
            }

        docs[filename]['chunk_count'] += 1
        pages = _parse_page_numbers(metadata.get('page_numbers', '[]'))
        docs[filename]['pages'].update(pages)

    return docs


def get_existing_post_ids(collection_name: str | None = None) -> set[str]:
    """
    Get all existing talk post IDs from the database.
    
    Useful for deduplication when ingesting talk posts.
    
    Args:
        collection_name: ChromaDB collection name
        
    Returns:
        Set of post_id strings that already exist in the database
    """
    collection = get_or_create_collection(collection_name)
    results = collection.get()
    
    post_ids = set()
    for i, _ in enumerate(results['ids']):
        metadata = results['metadatas'][i] if results['metadatas'] else {}
        if metadata.get('source_type') == 'talk_post' and 'post_id' in metadata:
            post_ids.add(metadata['post_id'])
    
    return post_ids


def count_chunks(collection_name: str | None = None) -> int:
    """
    Get total chunk count in collection.

    Args:
        collection_name: Collection name (uses config default if None)

    Returns:
        Number of chunks
    """
    collection = get_or_create_collection(collection_name)
    return collection.count()


# -----------------------------------------------------------------------------
# Delete Operations
# -----------------------------------------------------------------------------

def delete_by_filename(filename: str, collection_name: str | None = None) -> int:
    """
    Delete all chunks for a given filename.

    Args:
        filename: Filename to delete chunks for
        collection_name: Collection name (uses config default if None)

    Returns:
        Number of chunks deleted
    """
    collection = get_or_create_collection(collection_name)
    return _delete_by_metadata(collection, "filename", filename)


def delete_by_source_type(source_type: str, collection_name: str | None = None) -> int:
    """
    Delete all chunks for a given source type.

    Args:
        source_type: Source type to delete ('pdf' or 'talk_post')
        collection_name: Collection name (uses config default if None)

    Returns:
        Number of chunks deleted
    """
    collection = get_or_create_collection(collection_name)
    return _delete_by_metadata(collection, "source_type", source_type)


def delete_by_post_id(post_id: str, collection_name: str | None = None) -> int:
    """
    Delete all chunks for a specific talk post.
    
    Args:
        post_id: The post ID to delete
        collection_name: ChromaDB collection name
        
    Returns:
        Number of chunks deleted
    """
    collection = get_or_create_collection(collection_name)
    return _delete_by_metadata(collection, "post_id", post_id)


# -----------------------------------------------------------------------------
# Private Helpers
# -----------------------------------------------------------------------------

def _unpack_query_results(results: dict) -> list[dict]:
    """
    Convert raw ChromaDB query results into chunk dicts.

    Args:
        results: Raw results from collection.query()

    Returns:
        List of chunk dicts with id, text, metadata, chunk_index, distance
    """
    chunks = []
    if not results['ids'] or not results['ids'][0]:
        return chunks

    for i in range(len(results['ids'][0])):
        raw_meta = results['metadatas'][0][i]
        chunk = {
            "id": results['ids'][0][i],
            "text": results['documents'][0][i],
            "metadata": {
                "filename": raw_meta.get("filename", ""),
                "page_numbers": _parse_page_numbers(
                    raw_meta.get("page_numbers", "[]")
                ),
                "title": raw_meta.get("title") or None,
                "url": raw_meta.get("url") or None,
                "source_type": raw_meta.get("source_type", ""),
            },
            "chunk_index": int(raw_meta.get("chunk_index", 0)),
            "distance": results['distances'][0][i] if 'distances' in results else None
        }
        chunks.append(chunk)

    return chunks


def _delete_by_metadata(
    collection: chromadb.Collection,
    key: str,
    value: str
) -> int:
    """
    Delete all chunks matching a metadata key-value pair.

    Args:
        collection: ChromaDB collection
        key: Metadata key to match (e.g., 'filename', 'source_type')
        value: Value to match

    Returns:
        Number of chunks deleted
    """
    results = collection.get()
    chunk_ids_to_delete = [
        chunk_id
        for i, chunk_id in enumerate(results['ids'])
        if results['metadatas'][i].get(key) == value
    ]

    if chunk_ids_to_delete:
        collection.delete(ids=chunk_ids_to_delete)
        logger.info(f"Deleted {len(chunk_ids_to_delete)} chunks where {key}={value}")

    return len(chunk_ids_to_delete)


def _parse_page_numbers(page_str: str) -> list:
    """
    Safely parse page numbers from stored string representation.
    
    Args:
        page_str: String representation of page numbers (e.g., "[1, 2, 3]")
    
    Returns:
        List of page numbers, or empty list if parsing fails
    """
    try:
        pages = ast.literal_eval(page_str)
        return pages if isinstance(pages, list) else []
    except (ValueError, SyntaxError):
        return []
