"""
Upload data to S3 for the Gravity Spy chatbot.

Uploads:
  - PDFs (served directly from S3 via redirect)
  - ChromaDB index as a tar.gz archive (downloaded by the bot at startup)

Usage:
    python upload_to_s3.py                  # Upload everything
    python upload_to_s3.py --dry-run        # Show what would be uploaded
    python upload_to_s3.py --prefix mybot   # Custom S3 prefix
    python upload_to_s3.py --only pdfs      # Upload only PDFs
    python upload_to_s3.py --only index     # Upload only the ChromaDB archive

Requires a .env file (or shell environment) with:
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_ENDPOINT_URL
    AWS_BUCKET_NAME
"""

import argparse
import os
import sys
import tarfile
import tempfile
from pathlib import Path

import boto3
import dotenv

import config

dotenv.load_dotenv()

# =============================================================================
# Configuration
# =============================================================================

AWS_SERVICE = os.environ.get("AWS_SERVICE", "s3")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
AWS_ENDPOINT_URL = os.environ.get("AWS_ENDPOINT_URL", "")
AWS_BUCKET_NAME = os.environ.get("AWS_BUCKET_NAME", "")

DEFAULT_S3_PREFIX = "gravity-spy-wiki-bot"

PDF_DIR = config.PDF_DIR
CHROMA_DB_DIR = Path(config.CHROMA_DB_PATH)

CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".csv": "text/csv",
    ".gz": "application/gzip",
}


# =============================================================================
# S3 Operations
# =============================================================================

def validate_config():
    """Validate that required S3 configuration is present."""
    missing = []
    if not AWS_ACCESS_KEY_ID:
        missing.append("AWS_ACCESS_KEY_ID")
    if not AWS_SECRET_ACCESS_KEY:
        missing.append("AWS_SECRET_ACCESS_KEY")
    if not AWS_ENDPOINT_URL:
        missing.append("AWS_ENDPOINT_URL")
    if not AWS_BUCKET_NAME:
        missing.append("AWS_BUCKET_NAME")

    if missing:
        print(f"Error: Missing environment variables: {', '.join(missing)}")
        print("Check your .env file.")
        sys.exit(1)


def get_s3_client():
    """Create and return a configured S3 client."""
    return boto3.client(
        AWS_SERVICE,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        endpoint_url=AWS_ENDPOINT_URL,
    )


def get_content_type(filepath):
    """Determine content type from file extension."""
    suffix = Path(filepath).suffix.lower()
    return CONTENT_TYPES.get(suffix, "application/octet-stream")


def upload_file(s3_client, local_path, s3_key):
    """
    Upload a single file to S3.

    Returns:
        True if successful, False otherwise
    """
    try:
        s3_client.upload_file(
            str(local_path),
            AWS_BUCKET_NAME,
            s3_key,
            ExtraArgs={"ContentType": get_content_type(local_path)},
        )
        return True
    except Exception as e:
        print(f"  FAILED  {local_path}: {e}")
        return False


# =============================================================================
# PDF Upload
# =============================================================================

def upload_pdfs(s3_client, prefix, dry_run=False):
    """
    Upload all PDFs to S3.

    Returns:
        Dict with upload statistics.
    """
    if not PDF_DIR.is_dir():
        print(f"Warning: {PDF_DIR} does not exist, skipping PDFs")
        return {"total": 0, "success": 0, "failed": 0}

    files = [
        f for f in sorted(PDF_DIR.iterdir())
        if f.is_file() and not f.name.startswith(".")
    ]

    stats = {"total": len(files), "success": 0, "failed": 0}
    print(f"Uploading {len(files)} PDF(s)...")

    for filepath in files:
        s3_key = f"{prefix}/pdfs/{filepath.name}"
        size_mb = filepath.stat().st_size / (1024 * 1024)

        if dry_run:
            print(f"  [dry-run] {filepath.name} ({size_mb:.1f} MB)")
            continue

        if upload_file(s3_client, filepath, s3_key):
            print(f"  Uploaded {filepath.name} ({size_mb:.1f} MB)")
            stats["success"] += 1
        else:
            stats["failed"] += 1

    return stats


# =============================================================================
# ChromaDB Index Archive
# =============================================================================

def upload_index(s3_client, prefix, dry_run=False):
    """
    Tar the ChromaDB directory and upload as a single archive.

    The archive is created with chroma_db/ as the top-level directory,
    so extracting into data/ produces data/chroma_db/.

    Returns:
        True if successful, False otherwise
    """
    if not CHROMA_DB_DIR.is_dir():
        print(f"Error: {CHROMA_DB_DIR} does not exist")
        return False

    # Check that the directory has content
    contents = list(CHROMA_DB_DIR.iterdir())
    if not contents:
        print(f"Error: {CHROMA_DB_DIR} is empty")
        return False

    s3_key = f"{prefix}/chroma_db.tar.gz"

    if dry_run:
        print(f"  [dry-run] {CHROMA_DB_DIR} -> s3://{AWS_BUCKET_NAME}/{s3_key}")
        return True

    print(f"Creating archive of {CHROMA_DB_DIR}...")

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with tarfile.open(tmp_path, "w:gz") as tar:
            tar.add(str(CHROMA_DB_DIR), arcname="chroma_db")

        size_mb = Path(tmp_path).stat().st_size / (1024 * 1024)
        print(f"  Archive size: {size_mb:.1f} MB")

        print(f"  Uploading to {s3_key}...")
        if upload_file(s3_client, tmp_path, s3_key):
            print(f"  Uploaded chroma_db.tar.gz ({size_mb:.1f} MB)")
            return True
        return False
    finally:
        os.unlink(tmp_path)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Upload data to S3 for the Gravity Spy chatbot"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without uploading",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=DEFAULT_S3_PREFIX,
        help=f"S3 key prefix (default: {DEFAULT_S3_PREFIX})",
    )
    parser.add_argument(
        "--only",
        type=str,
        choices=["pdfs", "index"],
        default=None,
        help="Upload only PDFs or only the ChromaDB index",
    )
    args = parser.parse_args()

    print(f"Bucket:   {AWS_BUCKET_NAME}")
    print(f"Prefix:   {args.prefix}")
    print(f"Endpoint: {AWS_ENDPOINT_URL}")
    print()

    if not args.dry_run:
        validate_config()
        s3_client = get_s3_client()
    else:
        s3_client = None

    success = True

    # Upload PDFs
    if args.only is None or args.only == "pdfs":
        pdf_stats = upload_pdfs(s3_client, args.prefix, dry_run=args.dry_run)
        if pdf_stats["failed"] > 0:
            success = False
        print()

    # Upload ChromaDB index
    if args.only is None or args.only == "index":
        if not upload_index(s3_client, args.prefix, dry_run=args.dry_run):
            success = False
        print()

    if args.dry_run:
        print("Dry run complete.")
    elif success:
        print("Upload complete.")
    else:
        print("Upload completed with errors.")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())