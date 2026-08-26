"""
Wiki.js page processing library.

Two-stage pipeline mirroring the PDF workflow:
  1. Fetch: wiki-sync downloads pages from Wiki.js GraphQL API to data/wiki/
  2. Ingest: wiki command reads saved pages, chunks, embeds, and stores in ChromaDB

Each page is saved as a JSON file containing the page content and metadata.
The GraphQL endpoint and optional API token are configured via environment
variables (WIKI_GRAPHQL_URL, WIKI_API_TOKEN).
"""

import html
import json
import re
import ssl
import urllib.error
import urllib.request
from pathlib import Path

from config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    WIKI_GRAPHQL_URL,
    WIKI_API_TOKEN,
    WIKI_PUBLIC_URL,
    WIKI_DIR,
)
from processor import split_into_chunks, embed_and_store_batched
from logging_config import get_logger

logger = get_logger(__name__)

# ---------------------
# Constants
# ---------------------

LOG_INTERVAL = 50  # Log progress every N pages

# SSL context for internal traffic (matches redirector pattern)
_ssl_context = ssl.create_default_context()
_ssl_context.check_hostname = False
_ssl_context.verify_mode = ssl.CERT_NONE

# ---------------------
# GraphQL Queries
# ---------------------

LIST_PAGES_QUERY = """
{
    pages {
        list (orderBy: TITLE) {
            id
            path
            title
            updatedAt
        }
    }
}
"""

SINGLE_PAGE_QUERY = """
query ($id: Int!) {
    pages {
        single (id: $id) {
            id
            path
            title
            description
            content
            contentType
            updatedAt
            tags {
                tag
            }
        }
    }
}
"""


# ---------------------
# GraphQL Client
# ---------------------

def _graphql_request(query: str, variables: dict | None = None) -> dict:
    """
    Execute a GraphQL request against the Wiki.js API.

    Args:
        query: GraphQL query string
        variables: Optional query variables

    Returns:
        The 'data' portion of the GraphQL response

    Raises:
        RuntimeError: If the request fails or returns GraphQL errors
    """
    if not WIKI_GRAPHQL_URL:
        raise RuntimeError("WIKI_GRAPHQL_URL is not configured")

    payload = json.dumps({
        "query": query,
        "variables": variables or {},
    }).encode()

    headers = {"Content-Type": "application/json"}
    if WIKI_API_TOKEN:
        headers["Authorization"] = f"Bearer {WIKI_API_TOKEN}"

    req = urllib.request.Request(
        WIKI_GRAPHQL_URL,
        data=payload,
        headers=headers,
    )

    try:
        with urllib.request.urlopen(req, timeout=30, context=_ssl_context) as resp:
            data = json.loads(resp.read())
    except urllib.error.URLError as exc:
        raise RuntimeError(f"GraphQL request failed: {exc}") from exc

    if "errors" in data:
        msg = data["errors"][0].get("message", str(data["errors"]))
        raise RuntimeError(f"GraphQL error: {msg}")

    return data.get("data", {})


# ---------------------
# Stage 1: Fetch (wiki-sync)
# ---------------------

def fetch_page_list() -> list[dict]:
    """
    Fetch the list of all wiki pages (id, path, title, updatedAt).

    Returns:
        List of page summary dicts
    """
    data = _graphql_request(LIST_PAGES_QUERY)
    pages = data.get("pages", {}).get("list", [])
    logger.info(f"Found {len(pages)} wiki pages")
    return pages


def fetch_page_content(page_id: int) -> dict | None:
    """
    Fetch the full content of a single wiki page.

    Args:
        page_id: Wiki.js page ID

    Returns:
        Page dict with title, path, content, contentType, etc.,
        or None if the page could not be fetched
    """
    try:
        data = _graphql_request(SINGLE_PAGE_QUERY, {"id": page_id})
        return data.get("pages", {}).get("single")
    except RuntimeError as exc:
        logger.warning(f"Failed to fetch page {page_id}: {exc}")
        return None


def sync_wiki_pages(
    output_dir: str | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    """
    Fetch all wiki pages and save them as JSON files to disk.

    Each page is saved as data/wiki/{page_id}.json containing the full
    page content and metadata from the GraphQL API.

    Args:
        output_dir: Directory to save pages (defaults to WIKI_DIR)
        dry_run: If True, list pages without downloading
        force: If True, re-download even if file already exists

    Returns:
        Dict with keys: total, downloaded, skipped, failed
    """
    output_path = Path(output_dir) if output_dir else WIKI_DIR
    output_path.mkdir(parents=True, exist_ok=True)

    page_list = fetch_page_list()

    stats = {"total": len(page_list), "downloaded": 0, "skipped": 0, "failed": 0}

    if dry_run:
        existing = set(p.stem for p in output_path.glob("*.json"))
        new_count = sum(1 for p in page_list if str(p["id"]) not in existing)
        logger.info(
            f"Dry run: {len(page_list)} pages in wiki, "
            f"{len(existing)} already on disk, {new_count} new"
        )
        return stats

    for i, page_summary in enumerate(page_list, start=1):
        page_id = page_summary["id"]
        dest = output_path / f"{page_id}.json"

        if dest.exists() and not force:
            stats["skipped"] += 1
            continue

        if i % LOG_INTERVAL == 0:
            logger.info(f"Fetching page {i}/{len(page_list)}...")

        page = fetch_page_content(page_id)
        if page is None:
            stats["failed"] += 1
            continue

        dest.write_text(json.dumps(page, indent=2, ensure_ascii=False), encoding="utf-8")
        stats["downloaded"] += 1

    logger.info(
        f"Wiki sync complete: {stats['downloaded']} downloaded, "
        f"{stats['skipped']} skipped, {stats['failed']} failed"
    )
    return stats


# ---------------------
# Stage 2: Ingest (wiki)
# ---------------------

def strip_html_to_text(html_content: str) -> str:
    """
    Convert HTML content to plain text suitable for embedding.

    Preserves paragraph structure as double newlines and strips all tags.
    Decodes HTML entities.

    Args:
        html_content: Raw HTML string

    Returns:
        Clean plain text
    """
    text = html_content

    # Convert block-level elements to newlines
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(
        r"</(?:p|div|h[1-6]|li|tr|blockquote)>",
        "\n\n",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"</(?:td|th)>", "\t", text, flags=re.IGNORECASE)

    # Strip all remaining tags
    text = re.sub(r"<[^>]+>", "", text)

    # Decode HTML entities
    text = html.unescape(text)

    # Normalize whitespace: collapse runs of spaces/tabs on each line
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _extract_chunks_from_wiki_page(
    page: dict,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    """
    Extract and chunk text from a single wiki page dict.

    Args:
        page: Page dict loaded from a saved JSON file
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Overlap between chunks

    Returns:
        List of chunk dicts (without embeddings)
    """
    content = page.get("content", "")
    content_type = page.get("contentType", "")

    # Strip HTML if needed
    if content_type == "html" or content.strip().startswith("<"):
        content = strip_html_to_text(content)

    if not content or not content.strip():
        return []

    page_id = page["id"]
    title = page.get("title", "")
    path = page.get("path", "")
    tags = [t["tag"] for t in page.get("tags", [])]
    url = f"{WIKI_PUBLIC_URL}/{path}" if WIKI_PUBLIC_URL else ""

    # Prepend title for embedding context
    full_text = f"{title}\n\n{content}" if title else content

    text_chunks = split_into_chunks(full_text, chunk_size, chunk_overlap)

    chunks = []
    for i, chunk_text in enumerate(text_chunks):
        chunk_id = f"wiki_{page_id}_{i}"
        chunks.append({
            "id": chunk_id,
            "text": chunk_text,
            "metadata": {
                "source_type": "wiki",
                "page_id": str(page_id),
                "title": title,
                "path": path,
                "tags": ", ".join(tags),
                "url": url,
                "chunk_index": str(i),
                "filename": title or f"wiki_{page_id}",
                "page_numbers": "[]",
            },
        })

    return chunks


def process_wiki_pages(
    input_dir: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    chunk_batch_size: int = 500,
    delay: float = 0.0,
) -> int:
    """
    Process saved wiki pages: read JSON files, chunk, embed, store in ChromaDB.

    Mirrors the process_pdfs() pattern: reads from a directory of files on disk,
    extracts and chunks text, then embeds and stores in batches.

    Args:
        input_dir: Directory containing wiki JSON files (defaults to WIKI_DIR)
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Overlap between chunks in characters
        chunk_batch_size: Number of chunks to process per embedding batch
        delay: Seconds to wait between batches (helps avoid rate limiting)

    Returns:
        Total number of chunks stored
    """
    input_path = Path(input_dir) if input_dir else WIKI_DIR

    if not input_path.exists():
        logger.warning(f"Directory not found: {input_path}")
        return 0

    json_files = sorted(input_path.glob("*.json"))

    if not json_files:
        logger.warning(f"No wiki JSON files found in {input_path}")
        return 0

    logger.info(f"Found {len(json_files)} wiki page file(s)")

    all_chunks: list[dict] = []

    for i, json_path in enumerate(json_files, start=1):
        if i % LOG_INTERVAL == 0:
            logger.info(f"Processing wiki page {i}/{len(json_files)}: {json_path.name}")

        try:
            page = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Failed to read {json_path.name}: {exc}")
            continue

        chunks = _extract_chunks_from_wiki_page(page, chunk_size, chunk_overlap)

        if chunks:
            all_chunks.extend(chunks)

    if not all_chunks:
        logger.warning("No chunks extracted from any wiki pages")
        return 0

    logger.info(f"Total chunks to process: {len(all_chunks)}")

    total_stored = embed_and_store_batched(
        all_chunks,
        batch_size=chunk_batch_size,
        delay=delay,
    )

    logger.info(
        f"Complete: {len(json_files)} wiki pages processed, "
        f"{total_stored} chunks stored"
    )
    return total_stored