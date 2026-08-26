"""
PDF processing library.

Extracts text from PDFs, chunks, generates embeddings via Azure OpenAI,
and stores in ChromaDB vector database.
"""

import uuid
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)
from processor import split_into_chunks, embed_and_store_batched, embed_chunks
from vector_store import store_chunks
from logging_config import get_logger

logger = get_logger(__name__)


def clean_text(text: str) -> str:
    """
    Clean extracted text from PDFs.

    Args:
        text: Raw text extracted from PDF

    Returns:
        Cleaned text
    """
    text = text.replace("-\n", "")
    text = text.replace("\n", " ")
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip()


def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """
    Extract text from a PDF file, preserving page numbers.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of dictionaries with 'page' and 'text' keys
    """
    pages_data = []

    try:
        reader = PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                pages_data.append({
                    "page": page_num,
                    "text": text
                })
    except PdfReadError as e:
        logger.error(f"Failed to read PDF {pdf_path}: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error reading {pdf_path}: {e}")
        return []

    return pages_data


def extract_title_from_text(text: str, max_length: int = 100) -> str | None:
    """
    Attempt to extract a title from text.
    
    Looks for the first line that appears to be a title:
    - Between 10 and max_length characters
    - Not all uppercase (unless short, suggesting an acronym or short title)

    Args:
        text: Text to extract title from
        max_length: Maximum length for title

    Returns:
        Title string or None
    """
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line or len(line) <= 10 or len(line) >= max_length:
            continue
        
        # Skip all-caps lines (likely headers/footers) unless short
        is_all_caps = line.isupper()
        is_short = len(line.split()) <= 10
        if is_all_caps and not is_short:
            continue
            
        return line[:max_length]

    # Fallback: try first sentence
    sentences = text.split('.')
    if sentences:
        first_sentence = sentences[0].strip()
        if 10 < len(first_sentence) < max_length:
            return first_sentence[:max_length]

    return None


def _extract_chunks_from_pdf(
    pdf_path: Path,
    chunk_size: int,
    chunk_overlap: int
) -> list[dict]:
    """
    Extract and chunk text from a single PDF.
    
    Args:
        pdf_path: Path to the PDF file
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of chunk dicts (without embeddings)
    """
    pages_data = extract_text_from_pdf(pdf_path)
    
    if not pages_data:
        logger.warning(f"No text extracted from {pdf_path.name}")
        return []
    
    title = extract_title_from_text(pages_data[0]["text"]) if pages_data else None
    chunks = []
    
    for page_data in pages_data:
        page_num = page_data["page"]
        cleaned_text = clean_text(page_data["text"])
        
        if not cleaned_text:
            continue
        
        text_chunks = split_into_chunks(cleaned_text, chunk_size, chunk_overlap)
        
        for chunk_index, chunk_text in enumerate(text_chunks):
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": chunk_text,
                "metadata": {
                    "filename": pdf_path.name,
                    "page_numbers": str([page_num]),
                    "title": title or "",
                    "chunk_index": str(chunk_index),
                    "source_type": "pdf",
                }
            })
    
    return chunks


def process_pdfs(
    input_dir: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    chunk_batch_size: int = 500,
    delay: float = 0.0
) -> int:
    """
    Process PDFs: extract text, chunk, embed via Azure OpenAI, store in ChromaDB.

    Processes chunks in batches to manage memory. Each batch:
    1. Collects chunks until batch size reached
    2. Generates embeddings in a single batched call
    3. Stores in ChromaDB

    Args:
        input_dir: Directory containing PDF files
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Overlap between chunks in characters
        chunk_batch_size: Number of chunks to process per embedding batch
        delay: Seconds to wait between batches (helps avoid rate limiting)

    Returns:
        Total number of chunks stored
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        logger.warning(f"Directory not found: {input_dir}")
        return 0
    
    pdf_files = list(input_path.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {input_dir}")
        return 0

    logger.info(f"Found {len(pdf_files)} PDF file(s)")

    # Extract and chunk all PDFs
    all_chunks: list[dict] = []
    
    for i, pdf_path in enumerate(pdf_files, start=1):
        logger.info(f"Processing PDF {i}/{len(pdf_files)}: {pdf_path.name}")
        
        chunks = _extract_chunks_from_pdf(pdf_path, chunk_size, chunk_overlap)
        
        if chunks:
            logger.info(f"  Extracted {len(chunks)} chunks from {pdf_path.name}")
            all_chunks.extend(chunks)

    if not all_chunks:
        logger.warning("No chunks extracted from any PDFs")
        return 0

    logger.info(f"Total chunks to process: {len(all_chunks)}")

    # Embed and store in batches
    total_stored = embed_and_store_batched(all_chunks, batch_size=chunk_batch_size, delay=delay)

    logger.info(f"Complete: {len(pdf_files)} PDFs processed, {total_stored} chunks stored")
    
    return total_stored


def process_single_pdf(
    pdf_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> int:
    """
    Process a single PDF file.

    Args:
        pdf_path: Path to the PDF file
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Overlap between chunks in characters

    Returns:
        Number of chunks stored
    """
    path = Path(pdf_path)
    
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    logger.info(f"Processing: {path.name}")
    
    chunks = _extract_chunks_from_pdf(path, chunk_size, chunk_overlap)
    
    if not chunks:
        return 0
    
    logger.info(f"Generating embeddings for {len(chunks)} chunks...")
    embed_chunks(chunks)
    
    store_chunks(chunks)
    logger.info(f"Stored {len(chunks)} chunks from {path.name}")
    
    return len(chunks)