"""
Zotero sync module for the Gravity Spy RAG chatbot.

Fetches PDF attachments from a Zotero group library and saves them
to the local PDF directory for subsequent ingestion into ChromaDB.

Usage via CLI:
    python cli.py zotero-sync
    python cli.py zotero-sync --collection RMXHTMEV
    python cli.py zotero-sync --dry-run
"""

from pathlib import Path

from pyzotero import zotero

from config import ZOTERO_API_KEY, ZOTERO_GROUP_ID, ZOTERO_COLLECTION_KEY, PDF_DIR
from logging_config import get_logger

logger = get_logger(__name__)


def create_client():
    """Create and return a pyzotero client for the configured group library."""
    if not ZOTERO_API_KEY or not ZOTERO_GROUP_ID:
        raise ValueError(
            "ZOTERO_API_KEY and ZOTERO_GROUP_ID must be set in environment"
        )

    return zotero.Zotero(
        library_id=ZOTERO_GROUP_ID,
        library_type="group",
        api_key=ZOTERO_API_KEY,
    )


def fetch_items(client, collection_key=None):
    """
    Fetch all items from the group library, optionally filtered to a collection.

    Returns the full item list including child attachments.
    """
    if collection_key:
        logger.info("Fetching items from collection '%s'", collection_key)
        items = client.everything(client.collection_items(collection_key))
    else:
        logger.info("Fetching all items from group library")
        items = client.everything(client.items())

    logger.info("Retrieved %d items from Zotero", len(items))
    return items


def extract_pdf_items(items):
    """
    Extract PDF attachment items from a list of Zotero items.

    Zotero stores PDFs as child attachment items of bibliographic entries.
    collection_items() returns both parents and children, so we filter
    for attachment items with contentType 'application/pdf'.

    Returns a list of (filename, item_key) tuples.
    """
    pdf_items = []

    for item in items:
        data = item.get("data", {})

        if data.get("itemType") != "attachment":
            continue

        if data.get("contentType") != "application/pdf":
            continue

        filename = data.get("filename", f"{item['key']}.pdf")
        pdf_items.append((filename, item["key"]))

    logger.info("Found %d PDF attachments", len(pdf_items))
    return pdf_items


def sync_pdfs(client, collection_key=None, output_dir=None, dry_run=False,
              force=False):
    """
    Sync PDFs from Zotero to the local PDF directory.

    Args:
        client: pyzotero client instance.
        collection_key: Optional Zotero collection key to filter by.
        output_dir: Directory to save PDFs. Defaults to config.PDF_DIR.
        dry_run: If True, list what would be downloaded without downloading.
        force: If True, re-download even if the file already exists locally.

    Returns:
        dict with keys: downloaded, skipped, failed, total.
    """
    output_path = Path(output_dir) if output_dir else Path(PDF_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    items = fetch_items(client, collection_key)
    pdf_items = extract_pdf_items(items)

    results = {"downloaded": 0, "skipped": 0, "failed": 0, "total": len(pdf_items)}

    if not pdf_items:
        logger.warning("No PDF attachments found in Zotero")
        return results

    for filename, item_key in pdf_items:
        filepath = output_path / filename

        if filepath.exists() and not force:
            logger.debug("Skipping '%s' (already exists)", filename)
            if dry_run:
                print(f"  SKIP (exists): {filename}")
            results["skipped"] += 1
            continue

        if dry_run:
            action = "REPLACE" if filepath.exists() else "DOWNLOAD"
            print(f"  {action}: {filename}")
            results["downloaded"] += 1
            continue

        try:
            client.dump(item_key, str(filepath))
            logger.info("Downloaded: %s", filename)
            print(f"  Downloaded: {filename}")
            results["downloaded"] += 1
        except Exception as exc:
            logger.error("Failed to download '%s' (key=%s): %s",
                         filename, item_key, exc)
            print(f"  FAILED: {filename} ({exc})")
            results["failed"] += 1

    return results