"""
PDF Ingestion Script with Embeddings and Retrieval

Takes all PDFs in a folder, extracts their text, splits into chunks,
generates embeddings, and saves chunks + embeddings + metadata to JSONL.
Includes semantic search/retrieval functionality.
"""

import argparse
import json
import os
import uuid
import ast
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from numpy.linalg import norm

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


# Global embedding model (loaded once for efficiency)
_embedding_model = None

# Global ChromaDB client
_chroma_client = None


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Get or initialize the embedding model (singleton pattern).
    
    Args:
        model_name: Name of the sentence-transformers model to use
        
    Returns:
        SentenceTransformer model instance
    """
    global _embedding_model
    if _embedding_model is None:
        print(f"Loading embedding model: {model_name}")
        _embedding_model = SentenceTransformer(model_name)
        print("Model loaded successfully")
    return _embedding_model


def get_chroma_client(persist_directory: str = "chroma_db"):
    """
    Get or initialize ChromaDB client.
    
    Args:
        persist_directory: Directory to persist ChromaDB data
        
    Returns:
        ChromaDB client instance
    """
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
    return _chroma_client


def get_or_create_collection(collection_name: str = "pdf_chunks"):
    """
    Get or create a ChromaDB collection.
    
    Args:
        collection_name: Name of the collection
        
    Returns:
        ChromaDB collection
    """
    client = get_chroma_client()
    try:
        collection = client.get_collection(name=collection_name)
        print(f"Using existing collection: {collection_name}")
    except:
        collection = client.create_collection(name=collection_name)
        print(f"Created new collection: {collection_name}")
    return collection


def clean_text(text: str) -> str:
    """
    Clean extracted text from PDFs.
    
    - Replace hyphenated line breaks (-\n) with empty string
    - Replace newlines with spaces
    - Collapse multiple spaces to single space
    
    Args:
        text: Raw text extracted from PDF
        
    Returns:
        Cleaned text
    """
    # Fix hyphenation: replace "-\n" with ""
    text = text.replace("-\n", "")
    
    # Replace newlines with spaces
    text = text.replace("\n", " ")
    
    # Collapse multiple spaces to single space
    while "  " in text:
        text = text.replace("  ", " ")
    
    return text.strip()


def extract_text_from_pdf(pdf_path: Path) -> List[Dict[str, any]]:
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
            if text.strip():  # Only add non-empty pages
                pages_data.append({
                    "page": page_num,
                    "text": text
                })
    
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return []
    
    return pages_data


def extract_title_from_text(text: str, max_length: int = 100) -> Optional[str]:
    """
    Attempt to extract a title from text (first non-empty line or first sentence).
    
    Args:
        text: Text to extract title from
        max_length: Maximum length for title
        
    Returns:
        Title string or None
    """
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if line and len(line) > 10 and len(line) < max_length:
            # Check if it looks like a title (not all caps, has some structure)
            if not line.isupper() or len(line.split()) <= 10:
                return line[:max_length]
    
    # Fallback: first sentence
    sentences = text.split('.')
    if sentences:
        first_sentence = sentences[0].strip()
        if len(first_sentence) > 10 and len(first_sentence) < max_length:
            return first_sentence[:max_length]
    
    return None


def chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int
) -> List[str]:
    """
    Split text into overlapping chunks using LangChain's RecursiveCharacterTextSplitter.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum size of each chunk
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_text(text)
    return chunks


def generate_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Generate embeddings for a list of texts.
    
    Args:
        texts: List of text strings to embed
        model_name: Name of the embedding model
        
    Returns:
        numpy array of embeddings (shape: [num_texts, embedding_dim])
    """
    model = get_embedding_model(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings


def process_pdfs(
    input_dir: str,
    output_path: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str = "all-MiniLM-L6-v2"
) -> None:
    """
    Main processing function: read PDFs, extract text, chunk, embed, and save to JSONL.
    
    Args:
        input_dir: Directory containing PDF files
        output_path: Path to output JSONL file
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Overlap between chunks in characters
        embedding_model: Name of the sentence-transformers model to use
    """
    input_path = Path(input_dir)
    output_file = Path(output_path)
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Find all PDF files
    pdf_files = list(input_path.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s)")
    
    # Store chunks in the desired format
    processed_chunks = []
    
    # Track page numbers per chunk for grouping
    chunk_page_map: Dict[str, List[int]] = {}  # chunk_id -> list of page numbers
    
    # Process each PDF
    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        
        # Extract text with page numbers
        pages_data = extract_text_from_pdf(pdf_path)
        
        if not pages_data:
            print(f"  No text extracted from {pdf_path.name}")
            continue
        
        # Extract title from first page
        title = None
        if pages_data:
            title = extract_title_from_text(pages_data[0]["text"])
        
        pdf_chunk_count = 0
        
        # Process each page
        for page_data in pages_data:
            page_num = page_data["page"]
            raw_text = page_data["text"]
            
            # Clean the text
            cleaned_text = clean_text(raw_text)
            
            if not cleaned_text:
                continue
            
            # Chunk the text
            chunks = chunk_text(cleaned_text, chunk_size, chunk_overlap)
            
            # Create chunk records with metadata
            for chunk_index, chunk_text_content in enumerate(chunks):
                chunk_id = str(uuid.uuid4())
                
                # Store chunk in desired format
                chunk_record = {
                    "text": chunk_text_content,
                    "metadata": {
                        "filename": pdf_path.name,
                        "page_numbers": [page_num],  # Single page per chunk initially
                        "title": title,
                    },
                    "id": chunk_id,  # Keep ID for reference
                    "chunk_index": chunk_index,
                }
                
                processed_chunks.append(chunk_record)
                chunk_page_map[chunk_id] = [page_num]
                pdf_chunk_count += 1
        
        print(f"  Extracted {pdf_chunk_count} chunks")
    
    # Generate embeddings for all chunks
    print(f"\nGenerating embeddings for {len(processed_chunks)} chunks...")
    chunk_texts = [chunk["text"] for chunk in processed_chunks]
    embeddings = generate_embeddings(chunk_texts, embedding_model)
    
    # Add embeddings to chunks
    for i, chunk in enumerate(processed_chunks):
        chunk["embedding"] = embeddings[i].tolist()  # Convert numpy array to list
    
    # Write all chunks to JSONL file (for backup/compatibility)
    print(f"Writing {len(processed_chunks)} chunks with embeddings to {output_path}")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in processed_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    
    # Store chunks in ChromaDB vector database
    print(f"\nStoring {len(processed_chunks)} chunks in vector database...")
    collection = get_or_create_collection()
    
    # Prepare data for ChromaDB
    ids = [chunk["id"] for chunk in processed_chunks]
    texts = [chunk["text"] for chunk in processed_chunks]
    embeddings_list = [chunk["embedding"] for chunk in processed_chunks]
    metadatas = [
        {
            "filename": chunk["metadata"]["filename"],
            "page_numbers": str(chunk["metadata"]["page_numbers"]),  # ChromaDB needs strings
            "title": chunk["metadata"]["title"] or "",
            "chunk_index": str(chunk["chunk_index"])
        }
        for chunk in processed_chunks
    ]
    
    # Add to ChromaDB (in batches for large datasets)
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i+batch_size]
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = embeddings_list[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]
        
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_texts,
            metadatas=batch_metadatas
        )
        print(f"  Stored batch {i//batch_size + 1}/{(len(ids)-1)//batch_size + 1}")
    
    print(f"✅ All chunks stored in vector database!")
    print("Done!")


def load_chunks_from_jsonl(jsonl_path: str) -> List[Dict]:
    """
    Load chunks with embeddings from JSONL file.
    
    Args:
        jsonl_path: Path to JSONL file
        
    Returns:
        List of chunk dictionaries
    """
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score (0-1)
    """
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))


def retrieve_chunks(
    query: str,
    chunks: List[Dict] = None,
    top_k: int = 5,
    model_name: str = "all-MiniLM-L6-v2",
    use_vector_db: bool = True
) -> List[Dict]:
    """
    Retrieve top-k most similar chunks for a query using semantic search.
    
    Args:
        query: Search query text
        chunks: List of chunk dictionaries with embeddings (optional if use_vector_db=True)
        top_k: Number of top results to return
        model_name: Name of the embedding model (must match the one used for chunks)
        use_vector_db: If True, use ChromaDB; if False, use provided chunks list
        
    Returns:
        List of top-k chunks sorted by similarity (most similar first)
    """
    # Generate query embedding
    model = get_embedding_model(model_name)
    query_embedding = model.encode(query, convert_to_numpy=True).tolist()
    
    if use_vector_db:
        # Use ChromaDB for fast search
        collection = get_or_create_collection()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Convert ChromaDB results to our format
        retrieved_chunks = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                chunk = {
                    "id": results['ids'][0][i],
                    "text": results['documents'][0][i],
                    "metadata": {
                        "filename": results['metadatas'][0][i].get("filename", ""),
                        "page_numbers": ast.literal_eval(results['metadatas'][0][i].get("page_numbers", "[]")),  # Convert back to list
                        "title": results['metadatas'][0][i].get("title") or None,
                    },
                    "chunk_index": int(results['metadatas'][0][i].get("chunk_index", 0)),
                    "distance": results['distances'][0][i] if 'distances' in results else None
                }
                retrieved_chunks.append(chunk)
        
        return retrieved_chunks
    else:
        # Fallback to original method using chunks list
        if chunks is None:
            raise ValueError("chunks must be provided if use_vector_db=False")
        
        # Calculate similarities
        similarities = []
        for chunk in chunks:
            chunk_embedding = np.array(chunk["embedding"])
            similarity = cosine_similarity(np.array(query_embedding), chunk_embedding)
            similarities.append((similarity, chunk))
        
        # Sort by similarity (descending) and return top-k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in similarities[:top_k]]


def search_chunks(
    jsonl_path: str = None,
    query: str = None,
    top_k: int = 5,
    model_name: str = "all-MiniLM-L6-v2",
    use_vector_db: bool = True
) -> List[Dict]:
    """
    Search chunks using semantic search (from vector DB or JSONL).
    
    Args:
        jsonl_path: Path to JSONL file (only used if use_vector_db=False)
        query: Search query text
        top_k: Number of top results to return
        model_name: Name of the embedding model
        use_vector_db: If True, use ChromaDB; if False, load from JSONL
        
    Returns:
        List of top-k chunks sorted by similarity
    """
    if use_vector_db:
        return retrieve_chunks(query, top_k=top_k, model_name=model_name, use_vector_db=True)
    else:
        if jsonl_path is None:
            raise ValueError("jsonl_path required when use_vector_db=False")
        chunks = load_chunks_from_jsonl(jsonl_path)
        return retrieve_chunks(query, chunks, top_k, model_name, use_vector_db=False)


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Extract text from PDFs, chunk it, embed it, and save to JSONL. "
                    "Also supports semantic search/retrieval."
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Process PDFs and generate embeddings")
    ingest_parser.add_argument(
        "--input_dir",
        type=str,
        default="data/pdfs",
        help="Directory containing PDF files (default: data/pdfs)"
    )
    ingest_parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/chunks.jsonl",
        help="Output JSONL file path (default: outputs/chunks.jsonl)"
    )
    ingest_parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Maximum chunk size in characters (default: 1000)"
    )
    ingest_parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=200,
        help="Overlap between chunks in characters (default: 200)"
    )
    ingest_parser.add_argument(
        "--embedding_model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model name (default: all-MiniLM-L6-v2)"
    )
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search chunks using semantic search")
    search_parser.add_argument(
        "--jsonl_path",
        type=str,
        default="outputs/chunks.jsonl",
        help="Path to JSONL file with chunks (default: outputs/chunks.jsonl)"
    )
    search_parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Search query text"
    )
    search_parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top results to return (default: 5)"
    )
    search_parser.add_argument(
        "--embedding_model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model name (default: all-MiniLM-L6-v2)"
    )
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        # Validate input directory exists
        if not os.path.isdir(args.input_dir):
            print(f"Error: Input directory '{args.input_dir}' does not exist")
            return
        
        # Process PDFs
        process_pdfs(
            input_dir=args.input_dir,
            output_path=args.output_path,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            embedding_model=args.embedding_model
        )
    
    elif args.command == "search":
        # Try vector DB first, fallback to JSONL
        use_vector_db = True
        if not os.path.exists("chroma_db"):
            print("Vector database not found, using JSONL file...")
            use_vector_db = False
            if not os.path.exists(args.jsonl_path):
                print(f"Error: Neither vector DB nor JSONL file '{args.jsonl_path}' exists")
                print("Run 'python ingest_pdfs.py ingest' first to process PDFs")
                return
        
        print(f"Searching for: '{args.query}'")
        results = search_chunks(
            jsonl_path=args.jsonl_path,
            query=args.query,
            top_k=args.top_k,
            model_name=args.embedding_model,
            use_vector_db=use_vector_db
        )
        
        print(f"\nFound {len(results)} results:\n")
        for i, chunk in enumerate(results, 1):
            print(f"--- Result {i} ---")
            print(f"Source: {chunk['metadata']['filename']}")
            print(f"Page(s): {chunk['metadata']['page_numbers']}")
            if chunk['metadata']['title']:
                print(f"Title: {chunk['metadata']['title']}")
            print(f"Text: {chunk['text'][:1000]}...")
            print()
    
    else:
        # Default to ingest if no command specified (backward compatibility)
        if not os.path.isdir("data/pdfs"):
            parser.print_help()
            return
        
        process_pdfs(
            input_dir="data/pdfs",
            output_path="outputs/chunks.jsonl",
            chunk_size=1000,
            chunk_overlap=200
        )


if __name__ == "__main__":
    main()
