"""
CLI for managing the Gravity Spy vector database.

Commands:
    zotero-sync - Fetch PDFs from a Zotero group library
    wiki-sync   - Fetch Wiki.js pages to local disk
    pdf         - Ingest a single PDF file
    pdfs        - Ingest PDF files from a directory
    wiki        - Ingest wiki pages from local disk
    csv         - Ingest talk posts from a CSV file
    add-post    - Add a single talk post
    search      - Search the vector database
    status      - Show database status
    delete      - Delete a document from the database
    clear       - Clear all chunks of a specific source type
"""

import argparse
import sys
from pathlib import Path

from config import (
    PDF_DIR,
    CSV_DIR,
    CHROMA_DB_PATH,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BATCH_DELAY,
    ZOTERO_COLLECTION_KEY,
    WIKI_DIR,
)
from pdf_processor import process_pdfs, process_single_pdf
from talk_post_processor import ingest_talk_posts_from_csv, add_single_talk_post
from vector_store import retrieve_chunks, delete_by_source_type, delete_by_filename, count_chunks, get_all_documents
from zotero_sync import create_client as create_zotero_client, sync_pdfs
from wiki_processor import sync_wiki_pages, process_wiki_pages
from logging_config import get_logger

logger = get_logger(__name__)


def cmd_zotero_sync(args) -> int:
    """Handle Zotero sync command — fetch PDFs from Zotero group library."""
    try:
        client = create_zotero_client()
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    collection = getattr(args, "collection", None)

    if args.dry_run:
        print("Dry run: Checking Zotero for PDFs...")
    else:
        action = "Re-downloading" if args.force else "Syncing"
        target = f"collection '{collection}'" if collection else "entire group library"
        print(f"{action} PDFs from Zotero ({target})...")

    results = sync_pdfs(
        client=client,
        collection_key=collection,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        force=args.force,
    )

    print(f"\nTotal: {results['total']} PDFs in Zotero")
    print(f"  Downloaded: {results['downloaded']}")
    print(f"  Skipped: {results['skipped']}")
    if results["failed"]:
        print(f"  Failed: {results['failed']}")

    if not args.dry_run and results["downloaded"] > 0:
        print(f"\nRun 'python cli.py pdfs' to ingest the new PDFs into ChromaDB.")

    return 1 if results["failed"] else 0


def cmd_wiki_sync(args) -> int:
    """Handle wiki sync command — fetch Wiki.js pages to local disk."""
    print("Fetching wiki pages..." if not args.dry_run else "Dry run: Checking wiki...")

    try:
        results = sync_wiki_pages(
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            force=args.force,
        )
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1

    print(f"\nTotal: {results['total']} pages in wiki")
    print(f"  Downloaded: {results['downloaded']}")
    print(f"  Skipped: {results['skipped']}")
    if results["failed"]:
        print(f"  Failed: {results['failed']}")

    if not args.dry_run and results["downloaded"] > 0:
        print(f"\nRun 'python cli.py wiki' to ingest the new pages into ChromaDB.")

    return 1 if results["failed"] else 0


def cmd_wiki(args) -> int:
    """Handle wiki ingestion command."""
    input_path = Path(args.input_dir)

    if not input_path.is_dir():
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        print("Run 'python cli.py wiki-sync' first to fetch wiki pages.")
        return 1

    if args.dry_run:
        json_files = list(input_path.glob("*.json"))
        print(f"Dry run: Would process {len(json_files)} wiki page file(s)")
        return 0

    if args.replace:
        deleted = delete_by_source_type("wiki")
        print(f"Deleted {deleted} existing wiki chunks")

    count = process_wiki_pages(
        input_dir=args.input_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chunk_batch_size=args.batch_size,
        delay=args.delay,
    )
    print(f"\n✅ Processed {count} chunks from wiki pages")
    return 0


def cmd_pdf(args) -> int:
    """Handle single PDF ingestion command."""
    pdf_path = Path(args.file)
    
    if not pdf_path.is_file():
        print(f"Error: PDF file '{args.file}' does not exist")
        return 1
    
    if not pdf_path.suffix.lower() == '.pdf':
        print(f"Error: File '{args.file}' is not a PDF")
        return 1

    if args.dry_run:
        print(f"Dry run: Would process '{pdf_path.name}'")
        return 0

    count = process_single_pdf(
        pdf_path=str(pdf_path),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    print(f"\n✅ Processed {count} chunks from '{pdf_path.name}'")
    return 0


def cmd_pdfs(args) -> int:
    """Handle PDF ingestion command."""
    input_path = Path(args.input_dir)
    
    if not input_path.is_dir():
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return 1

    if args.dry_run:
        pdf_files = list(input_path.glob("*.pdf"))
        print(f"Dry run: Would process {len(pdf_files)} PDF file(s):")
        for pdf in pdf_files:
            print(f"  {pdf.name}")
        return 0

    if args.replace:
        deleted = delete_by_source_type("pdf")
        print(f"Deleted {deleted} existing PDF chunks")

    count = process_pdfs(
        input_dir=args.input_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chunk_batch_size=args.batch_size,
        delay=args.delay
    )
    print(f"\n✅ Processed {count} chunks from PDFs")
    return 0


def cmd_csv(args) -> int:
    """Handle CSV ingestion command."""
    csv_path = Path(args.csv_path)
    
    if not csv_path.is_file():
        print(f"Error: CSV file '{args.csv_path}' does not exist")
        return 1

    if args.dry_run:
        with open(csv_path, "r", encoding="utf-8") as f:
            row_count = sum(1 for _ in f) - 1  # Subtract header
        print(f"Dry run: Would process {row_count} rows from '{csv_path.name}'")
        return 0

    if args.replace:
        deleted = delete_by_source_type("talk_post")
        print(f"Deleted {deleted} existing talk post chunks")

    count = ingest_talk_posts_from_csv(
        csv_path=args.csv_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        delay=args.delay,
        on_duplicate=args.on_duplicate
    )
    print(f"\n✅ Processed {count} chunks from CSV")
    return 0


def cmd_add_post(args) -> int:
    """Handle single post addition command."""
    count = add_single_talk_post(
        post_id=args.post_id,
        title=args.title,
        content=args.content,
        author=args.author,
        date=args.date,
        url=args.url
    )
    print(f"\n✅ Added talk post with {count} chunks")
    return 0


def cmd_search(args) -> int:
    """Handle search command."""
    db_path = Path(CHROMA_DB_PATH)
    
    if not db_path.exists():
        print(f"Error: Vector DB not found at '{CHROMA_DB_PATH}'")
        print("Run 'python cli.py pdfs' or 'python cli.py csv' first")
        return 1

    print(f"Searching for: '{args.query}'")
    results = retrieve_chunks(query=args.query, top_k=args.top_k)

    print(f"\nFound {len(results)} results:\n")
    for i, chunk in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(f"Source: {chunk['metadata']['filename']}")
        pages = chunk['metadata'].get('page_numbers')
        if pages:
            print(f"Page(s): {pages}")
        if chunk['metadata'].get('title'):
            print(f"Title: {chunk['metadata']['title']}")
        print(f"Text: {chunk['text'][:500]}...")
        print()
    
    return 0


def cmd_status(args) -> int:
    """Show database status."""
    db_path = Path(CHROMA_DB_PATH)
    
    if not db_path.exists():
        print(f"Vector DB not found at '{CHROMA_DB_PATH}'")
        print("No documents ingested yet.")
        return 0

    total = count_chunks()
    docs = get_all_documents()
    
    pdf_count = sum(1 for f in docs if f.endswith('.pdf'))
    talk_count = sum(1 for f in docs if f.startswith('talk_post_'))
    wiki_count = sum(1 for f in docs if f.startswith('wiki_'))
    
    print(f"Database: {CHROMA_DB_PATH}")
    print(f"Total chunks: {total}")
    print(f"Total documents: {len(docs)}")
    print(f"  PDFs: {pdf_count}")
    print(f"  Talk posts: {talk_count}")
    print(f"  Wiki pages: {wiki_count}")
    
    if args.verbose and docs:
        print("\nDocuments:")
        for filename, info in sorted(docs.items()):
            pages = sorted(info['pages']) if info['pages'] else []
            page_info = f", pages {pages[0]}-{pages[-1]}" if pages else ""
            print(f"  {filename}: {info['chunk_count']} chunks{page_info}")
    
    return 0


def cmd_delete(args) -> int:
    """Handle delete command."""
    filename = args.filename
    
    # Check if document exists
    docs = get_all_documents()
    if filename not in docs:
        print(f"Error: Document '{filename}' not found in database")
        return 1
    
    chunk_count = docs[filename]['chunk_count']
    
    if args.dry_run:
        print(f"Dry run: Would delete '{filename}' ({chunk_count} chunks)")
        return 0
    
    if not args.force:
        confirm = input(f"Delete '{filename}' ({chunk_count} chunks)? [y/N] ")
        if confirm.lower() != 'y':
            print("Cancelled")
            return 0
    
    deleted = delete_by_filename(filename)
    print(f"✅ Deleted {deleted} chunks for '{filename}'")
    return 0


def cmd_clear(args) -> int:
    """Handle clear command - delete all chunks of a specific source type."""
    source_type = args.source_type
    
    # Validate source type
    valid_types = ["pdf", "talk_post", "wiki"]
    if source_type not in valid_types:
        print(f"Error: Invalid source type '{source_type}'")
        print(f"Valid types: {', '.join(valid_types)}")
        return 1
    
    if args.dry_run:
        # Count how many would be deleted
        docs = get_all_documents()
        if source_type == "pdf":
            count = sum(info['chunk_count'] for f, info in docs.items() if f.endswith('.pdf'))
        else:  # talk_post
            count = sum(info['chunk_count'] for f, info in docs.items() if f.startswith('talk_post_'))
        print(f"Dry run: Would delete {count} chunks of type '{source_type}'")
        return 0
    
    if not args.force:
        confirm = input(f"Delete ALL '{source_type}' chunks? This cannot be undone. [y/N] ")
        if confirm.lower() != 'y':
            print("Cancelled")
            return 0
    
    deleted = delete_by_source_type(source_type)
    print(f"✅ Deleted {deleted} chunks of type '{source_type}'")
    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Manage the Gravity Spy vector database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python cli.py zotero-sync
    python cli.py zotero-sync --collection RMXHTMEV
    python cli.py zotero-sync --dry-run
    python cli.py wiki-sync
    python cli.py wiki-sync --dry-run
    python cli.py wiki-sync --force
    python cli.py pdf --file ./data/pdfs/paper.pdf
    python cli.py pdfs --input-dir ./data/pdfs
    python cli.py pdfs --input-dir ./data/pdfs --dry-run
    python cli.py wiki
    python cli.py wiki --replace
    python cli.py csv --csv-path ./data/talk_posts.csv
    python cli.py csv --csv-path ./data/talk_posts.csv --on-duplicate replace
    python cli.py csv --csv-path ./data/talk_posts.csv --batch-size 50 --delay 2
    python cli.py add-post --post-id 123 --title "My Post" --content "..."
    python cli.py search --query "glitch classification"
    python cli.py status --verbose
    python cli.py delete --filename "paper.pdf"
    python cli.py clear --source-type talk_post
    python cli.py clear --source-type pdf --force
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Shared arguments for chunking
    chunk_parser = argparse.ArgumentParser(add_help=False)
    chunk_parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help=f"Maximum chunk size in characters (default: {DEFAULT_CHUNK_SIZE})"
    )
    chunk_parser.add_argument(
        "--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
        help=f"Overlap between chunks (default: {DEFAULT_CHUNK_OVERLAP})"
    )
    chunk_parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Number of items per batch (default: {DEFAULT_BATCH_SIZE})"
    )
    chunk_parser.add_argument(
        "--delay", type=float, default=DEFAULT_BATCH_DELAY,
        help=f"Seconds to wait between batches to avoid rate limiting (default: {DEFAULT_BATCH_DELAY})"
    )

    # Shared arguments for ingestion
    ingest_parser = argparse.ArgumentParser(add_help=False)
    ingest_parser.add_argument(
        "--replace", action="store_true",
        help="Delete existing chunks of this type before ingesting"
    )
    ingest_parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be processed without actually ingesting"
    )

    # Zotero sync
    zotero_parser = subparsers.add_parser(
        "zotero-sync",
        help="Fetch PDFs from a Zotero group library"
    )
    zotero_parser.add_argument(
        "--collection", type=str, default=ZOTERO_COLLECTION_KEY,
        help="Zotero collection key to sync (default: ZOT_COLLECTION_KEY from env)"
    )
    zotero_parser.add_argument(
        "--output-dir", type=str, default=str(PDF_DIR),
        help=f"Directory to save PDFs (default: {PDF_DIR})"
    )
    zotero_parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be downloaded without actually downloading"
    )
    zotero_parser.add_argument(
        "--force", "-f", action="store_true",
        help="Re-download PDFs even if they already exist locally"
    )
    zotero_parser.set_defaults(func=cmd_zotero_sync)

    # Wiki.js sync (fetch to disk)
    wiki_sync_parser = subparsers.add_parser(
        "wiki-sync",
        help="Fetch Wiki.js pages to local disk"
    )
    wiki_sync_parser.add_argument(
        "--output-dir", type=str, default=str(WIKI_DIR),
        help=f"Directory to save wiki pages (default: {WIKI_DIR})"
    )
    wiki_sync_parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be downloaded without actually downloading"
    )
    wiki_sync_parser.add_argument(
        "--force", "-f", action="store_true",
        help="Re-download pages even if they already exist locally"
    )
    wiki_sync_parser.set_defaults(func=cmd_wiki_sync)

    # Single PDF ingestion
    pdf_parser = subparsers.add_parser(
        "pdf",
        parents=[chunk_parser],
        help="Ingest a single PDF file"
    )
    pdf_parser.add_argument(
        "--file", type=str, required=True,
        help="Path to the PDF file"
    )
    pdf_parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be processed without actually ingesting"
    )
    pdf_parser.set_defaults(func=cmd_pdf)

    # PDF directory ingestion
    pdfs_parser = subparsers.add_parser(
        "pdfs", 
        parents=[chunk_parser, ingest_parser],
        help="Ingest PDFs from a directory"
    )
    pdfs_parser.add_argument(
        "--input-dir", type=str, default=str(PDF_DIR),
        help=f"Directory containing PDF files (default: {PDF_DIR})"
    )
    pdfs_parser.set_defaults(func=cmd_pdfs)

    # Wiki page ingestion (from local disk)
    wiki_ingest_parser = subparsers.add_parser(
        "wiki",
        parents=[chunk_parser, ingest_parser],
        help="Ingest wiki pages from local disk"
    )
    wiki_ingest_parser.add_argument(
        "--input-dir", type=str, default=str(WIKI_DIR),
        help=f"Directory containing wiki JSON files (default: {WIKI_DIR})"
    )
    wiki_ingest_parser.set_defaults(func=cmd_wiki)

    # CSV ingestion
    csv_parser = subparsers.add_parser(
        "csv", 
        parents=[chunk_parser, ingest_parser],
        help="Ingest talk posts from CSV"
    )
    csv_parser.add_argument(
        "--csv-path", type=str, default=str(CSV_DIR / "talk_posts.csv"),
        help=f"Path to CSV file (default: {CSV_DIR / 'talk_posts.csv'})"
    )
    csv_parser.add_argument(
        "--on-duplicate", type=str, default="skip",
        choices=["skip", "replace", "allow"],
        help="How to handle duplicate posts: skip (default), replace, or allow"
    )
    csv_parser.set_defaults(func=cmd_csv)

    # Single post addition
    add_parser = subparsers.add_parser("add-post", help="Add a single talk post")
    add_parser.add_argument("--post-id", type=str, required=True, help="Unique post ID")
    add_parser.add_argument("--title", type=str, required=True, help="Post title")
    add_parser.add_argument("--content", type=str, required=True, help="Post content")
    add_parser.add_argument("--author", type=str, default="", help="Post author")
    add_parser.add_argument("--date", type=str, default="", help="Post date")
    add_parser.add_argument("--url", type=str, default="", help="Post URL")
    add_parser.set_defaults(func=cmd_add_post)

    # Search
    search_parser = subparsers.add_parser("search", help="Search the vector database")
    search_parser.add_argument(
        "--query", type=str, required=True,
        help="Search query text"
    )
    search_parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K,
        help=f"Number of results to return (default: {DEFAULT_TOP_K})"
    )
    search_parser.set_defaults(func=cmd_search)

    # Status
    status_parser = subparsers.add_parser("status", help="Show database status")
    status_parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show detailed document list"
    )
    status_parser.set_defaults(func=cmd_status)

    # Delete single document
    delete_parser = subparsers.add_parser("delete", help="Delete a document from the database")
    delete_parser.add_argument(
        "--filename", type=str, required=True,
        help="Filename of the document to delete"
    )
    delete_parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    delete_parser.add_argument(
        "--force", "-f", action="store_true",
        help="Skip confirmation prompt"
    )
    delete_parser.set_defaults(func=cmd_delete)

    # Clear by source type
    clear_parser = subparsers.add_parser("clear", help="Clear all chunks of a specific source type")
    clear_parser.add_argument(
        "--source-type", type=str, required=True,
        choices=["pdf", "talk_post", "wiki"],
        help="Source type to clear: 'pdf', 'talk_post', or 'wiki'"
    )
    clear_parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    clear_parser.add_argument(
        "--force", "-f", action="store_true",
        help="Skip confirmation prompt"
    )
    clear_parser.set_defaults(func=cmd_clear)

    return parser


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())