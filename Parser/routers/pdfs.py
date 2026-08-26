"""
PDF management endpoints (read-only).

Destructive operations (upload, delete) are handled via CLI.
See: python cli.py --help
"""

import os
from urllib.parse import quote

from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse

from models import DocumentInfo
from vector_store import get_all_documents
from routers.utils import validate_filename

router = APIRouter(prefix="/api/pdfs", tags=["pdfs"])

# S3 public base URL for PDF downloads
S3_SEED_DATA_BASE = os.environ.get(
    "S3_SEED_DATA_BASE", "https://s3.gswiki.ischool.syr.edu/images"
)
S3_SEED_DATA_PREFIX = os.environ.get(
    "S3_SEED_DATA_PREFIX", "gravity-spy-wiki-bot"
)


@router.get("", response_model=list[DocumentInfo])
async def list_pdfs():
    """Get list of all PDFs in the database."""
    docs = get_all_documents()

    # Filter to only PDF files
    pdf_docs = {
        filename: info
        for filename, info in docs.items()
        if filename.endswith('.pdf')
    }

    return [
        DocumentInfo(
            filename=filename,
            chunk_count=info['chunk_count'],
            pages=sorted(list(info['pages']))
        )
        for filename, info in pdf_docs.items()
    ]


@router.get("/{filename}", response_model=dict)
async def get_pdf_info(filename: str):
    """Get detailed information about a specific PDF."""
    docs = get_all_documents()

    if filename not in docs:
        raise HTTPException(status_code=404, detail=f"PDF '{filename}' not found")

    info = docs[filename]
    pages = sorted(list(info['pages']))

    return {
        "filename": filename,
        "chunk_count": info['chunk_count'],
        "pages": pages,
        "total_pages": len(pages)
    }


@router.get("/{filename}/download")
async def download_pdf(filename: str):
    """
    Redirect to the PDF on S3 for download.

    Only allows files with .pdf extension to prevent open redirect abuse.
    """
    filename = validate_filename(filename, allowed_extension=".pdf")

    # Verify the PDF exists in the index
    docs = get_all_documents()
    if filename not in docs:
        raise HTTPException(status_code=404, detail=f"PDF '{filename}' not found")

    # URL-encode the filename for the S3 path
    encoded_filename = quote(filename, safe="")
    s3_url = f"{S3_SEED_DATA_BASE}/{S3_SEED_DATA_PREFIX}/pdfs/{encoded_filename}"

    return RedirectResponse(url=s3_url)