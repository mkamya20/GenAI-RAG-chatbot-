"""
Talk post processing library.

Processes Zooniverse Talk posts, chunks, generates embeddings via Azure OpenAI,
and stores in ChromaDB vector database.
"""

import csv
import uuid
from pathlib import Path

from config import (
    CHROMA_COLLECTION_NAME,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_BATCH_SIZE,
)
from processor import split_into_chunks, embed_and_store_batched, embed_chunks
from vector_store import store_chunks, get_existing_post_ids, delete_by_post_id
from logging_config import get_logger

logger = get_logger(__name__)

# ---------------------
# Constants
# ---------------------

# GRAVITYbot's own content - exclude to prevent indexing AI-generated summaries
# These posts are LLM-generated summaries that would pollute RAG retrieval
GRAVITYBOT_BOARD_ID = 6872
GRAVITYBOT_USER_ID = 2877652

# Processing constants
LOG_INTERVAL = 100  # Log progress every N rows


def chunk_talk_post(
    post_id: str,
    title: str,
    content: str,
    author: str = "",
    date: str = "",
    url: str = "",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> list[dict]:
    """
    Chunk a single talk post and prepare metadata (no embedding).

    Args:
        post_id: Unique identifier for the post
        title: Post title
        content: Post content/text
        author: Post author
        date: Post date
        url: Post URL
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Overlap between chunks

    Returns:
        List of chunk dictionaries (without embeddings)
    """
    full_text = f"{title}\n\n{content}" if title else content

    text_chunks = split_into_chunks(full_text, chunk_size, chunk_overlap)

    chunks = []
    for i, chunk_text in enumerate(text_chunks):
        chunk_id = f"talk_post_{post_id}_{i}"
        chunks.append({
            "id": chunk_id,
            "text": chunk_text,
            "metadata": {
                "source_type": "talk_post",
                "post_id": post_id,
                "title": title,
                "author": author,
                "date": date,
                "url": url,
                "chunk_index": str(i),
                "filename": f"talk_post_{post_id}",
                "page_numbers": "[]"
            }
        })

    return chunks


def _normalize_row(row: dict) -> dict | None:
    """
    Normalize a CSV row to standard field names.
    
    Handles both Gravity Spy format and generic CSV format.
    
    Args:
        row: Raw CSV row dictionary
        
    Returns:
        Normalized dict with id, title, content, author, date, url keys,
        or None if row should be skipped
    """
    # Gravity Spy format
    if "comment_id" in row:
        content = row.get("comment_body", "").strip()
        if not content:
            return None
            
        board_id = row.get("board_id", "")
        discussion_id = row.get("discussion_id", "")
        if board_id and discussion_id:
            url = f"https://www.zooniverse.org/projects/zooniverse/gravity-spy/talk/{board_id}/{discussion_id}"
        else:
            url = ""
            
        return {
            "id": row.get("comment_id", str(uuid.uuid4())),
            "title": row.get("discussion_title", ""),
            "content": content,
            "author": row.get("comment_user_login", ""),
            "date": row.get("comment_created_at", ""),
            "url": url,
        }
    
    # Generic format
    content = (
        row.get("content") or 
        row.get("text") or 
        row.get("body", "")
    ).strip()
    
    if not content:
        return None
    
    return {
        "id": row.get("id") or row.get("post_id") or str(uuid.uuid4()),
        "title": row.get("title", ""),
        "content": content,
        "author": row.get("author", ""),
        "date": row.get("date") or row.get("created_at") or row.get("posted_at", ""),
        "url": row.get("url") or row.get("link", ""),
    }


def _should_exclude_row(row: dict) -> bool:
    """
    Check if a row should be excluded from indexing.
    
    Filters out GRAVITYbot's own posts to prevent indexing AI-generated
    summaries, which would pollute RAG retrieval results.
    
    Args:
        row: Raw CSV row dictionary
        
    Returns:
        True if row should be excluded, False otherwise
    """
    # Only applies to Gravity Spy format
    if "comment_id" not in row:
        return False
    
    # Check board_id
    try:
        board_id = int(row.get("board_id") or 0)
        if board_id == GRAVITYBOT_BOARD_ID:
            return True
    except (ValueError, TypeError):
        pass
    
    # Check user_id
    try:
        user_id = int(row.get("comment_user_id") or 0)
        if user_id == GRAVITYBOT_USER_ID:
            return True
    except (ValueError, TypeError):
        pass
    
    return False


def ingest_talk_posts_from_csv(
    csv_path: str,
    collection_name: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    batch_size: int = DEFAULT_BATCH_SIZE,
    delay: float = 0.0,
    on_duplicate: str = "skip"
) -> int:
    """
    Ingest talk posts from a CSV file into ChromaDB.

    Processes the CSV in batches to manage memory. Each batch:
    1. Reads rows from CSV
    2. Chunks all posts
    3. Generates embeddings in a single batched call
    4. Stores in ChromaDB
    
    Automatically filters out GRAVITYbot's own posts (board 6872, user 2877652)
    to prevent indexing AI-generated summaries.

    Expected CSV columns (Gravity Spy format):
    - comment_id, comment_body, discussion_title, comment_user_login,
      comment_created_at, board_id, discussion_id

    Or generic format:
    - id (or post_id): Unique identifier
    - content (or text, body): Post content
    - title: Post title (optional)
    - author: Post author (optional)
    - date (or created_at, posted_at): Post date (optional)
    - url (or link): Post URL (optional)

    Args:
        csv_path: Path to CSV file
        collection_name: ChromaDB collection name
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap between chunks
        batch_size: Number of rows to process per batch
        delay: Seconds to wait between batches (helps avoid rate limiting)
        on_duplicate: How to handle posts that already exist in the database
            - "skip": Skip duplicates (default, fastest)
            - "replace": Delete existing chunks and re-ingest
            - "allow": Allow duplicates (not recommended)

    Returns:
        Total number of chunks stored
    """
    if on_duplicate not in ("skip", "replace", "allow"):
        raise ValueError(f"on_duplicate must be 'skip', 'replace', or 'allow', got '{on_duplicate}'")
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    name = collection_name or CHROMA_COLLECTION_NAME
    
    # Load existing post IDs if we need to check for duplicates
    existing_post_ids: set[str] = set()
    if on_duplicate in ("skip", "replace"):
        existing_post_ids = get_existing_post_ids(collection_name=name)
        if existing_post_ids:
            logger.info(f"Found {len(existing_post_ids)} existing posts in database")
    
    # Count total rows for progress logging
    with open(csv_file, "r", encoding="utf-8") as f:
        total_rows = sum(1 for _ in f) - 1  # Subtract header
    
    total_batches = (total_rows + batch_size - 1) // batch_size
    logger.info(f"Processing {total_rows} rows in ~{total_batches} batches of {batch_size}")

    total_chunks_stored = 0
    rows_processed = 0
    rows_skipped = 0
    rows_excluded = 0  # GRAVITYbot posts
    rows_duplicate = 0  # Already in database
    batch_num = 0
    current_batch_chunks: list[dict] = []
    current_batch_row_count = 0

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            rows_processed += 1
            
            # Filter out GRAVITYbot's own posts
            if _should_exclude_row(row):
                rows_excluded += 1
                continue
            
            normalized = _normalize_row(row)
            if normalized is None:
                rows_skipped += 1
                continue
            
            post_id = normalized["id"]
            
            # Handle duplicates
            if post_id in existing_post_ids:
                if on_duplicate == "skip":
                    rows_duplicate += 1
                    continue
                elif on_duplicate == "replace":
                    delete_by_post_id(post_id, collection_name=name)
                    rows_duplicate += 1  # Still count it, but we'll re-add
                # "allow" falls through and creates duplicate
            
            chunks = chunk_talk_post(
                post_id=normalized["id"],
                title=normalized["title"],
                content=normalized["content"],
                author=normalized["author"],
                date=normalized["date"],
                url=normalized["url"],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            current_batch_chunks.extend(chunks)
            current_batch_row_count += 1
            
            # Log progress periodically
            if rows_processed % LOG_INTERVAL == 0:
                logger.info(f"Read {rows_processed}/{total_rows} rows...")
            
            # Process batch when full
            if current_batch_row_count >= batch_size:
                batch_num += 1
                total_chunks_stored += embed_and_store_batched(
                    current_batch_chunks,
                    collection_name=name,
                    batch_size=len(current_batch_chunks),  # Process as single batch
                    delay=delay
                )
                current_batch_chunks = []
                current_batch_row_count = 0
        
        # Process remaining chunks
        if current_batch_chunks:
            batch_num += 1
            total_chunks_stored += embed_and_store_batched(
                current_batch_chunks,
                collection_name=name,
                batch_size=len(current_batch_chunks),
                delay=delay
            )

    logger.info(
        f"Complete: {rows_processed} rows processed, {rows_skipped} skipped (empty), "
        f"{rows_excluded} excluded (GRAVITYbot), {rows_duplicate} duplicates ({on_duplicate}), "
        f"{total_chunks_stored} chunks stored"
    )
    
    return total_chunks_stored


def add_single_talk_post(
    post_id: str,
    title: str,
    content: str,
    author: str = "",
    date: str = "",
    url: str = "",
    collection_name: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> int:
    """
    Add a single talk post to ChromaDB.

    Args:
        post_id: Unique identifier for the post
        title: Post title
        content: Post content
        author: Post author
        date: Post date
        url: Post URL
        collection_name: ChromaDB collection name
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap between chunks

    Returns:
        Number of chunks created
    """
    chunks = chunk_talk_post(
        post_id=post_id,
        title=title,
        content=content,
        author=author,
        date=date,
        url=url,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    if not chunks:
        return 0

    embed_chunks(chunks)

    name = collection_name or CHROMA_COLLECTION_NAME
    store_chunks(chunks, collection_name=name)

    logger.info(f"Added talk post '{title}' with {len(chunks)} chunks")
    return len(chunks)