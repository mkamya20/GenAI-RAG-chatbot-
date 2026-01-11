"""
Script to ingest talk posts from CSV into ChromaDB vector database.
Supports both batch ingestion from CSV and individual post addition.
"""

import csv
import uuid
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from ingest_pdfs import get_embedding_model, get_chroma_client, get_or_create_collection


def process_talk_post(
    post_id: str,
    title: str,
    content: str,
    author: Optional[str] = None,
    date: Optional[str] = None,
    url: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_model_name: str = "all-MiniLM-L6-v2"
) -> List[Dict]:
    """
    Process a single talk post: chunk it, generate embeddings, and prepare for storage.
    
    Args:
        post_id: Unique identifier for the post
        title: Post title
        content: Post content/text
        author: Post author (optional)
        date: Post date (optional)
        url: Post URL (optional)
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Overlap between chunks
        embedding_model_name: Embedding model to use
        
    Returns:
        List of chunk dictionaries ready for storage
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    # Combine title and content for better context
    full_text = f"{title}\n\n{content}" if title else content
    
    # Chunk the text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_text(full_text)
    
    # Generate embeddings
    model = get_embedding_model(embedding_model_name)
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    
    # Prepare chunks for storage
    processed_chunks = []
    for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_id = f"talk_post_{post_id}_{i}"
        processed_chunks.append({
            "id": chunk_id,
            "text": chunk_text,
            "embedding": embedding.tolist(),
            "metadata": {
                "source_type": "talk_post",
                "post_id": post_id,
                "title": title or "",
                "author": author or "",
                "date": date or "",
                "url": url or "",
                "chunk_index": str(i),
                "filename": f"talk_post_{post_id}"  # For compatibility with existing code
            }
        })
    
    return processed_chunks


def ingest_talk_posts_from_csv(
    csv_path: str,
    collection_name: str = "pdf_chunks",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_model_name: str = "all-MiniLM-L6-v2"
) -> int:
    """
    Ingest talk posts from a CSV file into ChromaDB.
    
    Expected CSV columns:
    - id (or post_id): Unique identifier
    - title: Post title
    - content (or text, body): Post content
    - author: Post author (optional)
    - date (or created_at, posted_at): Post date (optional)
    - url: Post URL (optional)
    
    Args:
        csv_path: Path to CSV file
        collection_name: ChromaDB collection name
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap between chunks
        embedding_model_name: Embedding model name
        
    Returns:
        Number of posts processed
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Read CSV
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Map CSV columns to our standard format
    # Handle Gravity Spy talk posts CSV structure
    if 'comment_id' in df.columns:
        # This is a Gravity Spy talk posts CSV
        df['id'] = df['comment_id'].astype(str)
        df['content'] = df['comment_body'].fillna('')
        df['title'] = df['discussion_title'].fillna('')
        df['author'] = df['comment_user_login'].fillna('')
        df['date'] = df['comment_created_at'].fillna('')
        # Create a URL-like identifier from discussion_id
        df['url'] = 'discussion_' + df['discussion_id'].astype(str)
    else:
        # Normalize column names (handle different possible column names)
        column_mapping = {
            'id': 'id',
            'post_id': 'id',
            'title': 'title',
            'content': 'content',
            'text': 'content',
            'body': 'content',
            'author': 'author',
            'date': 'date',
            'created_at': 'date',
            'posted_at': 'date',
            'url': 'url',
            'link': 'url'
        }
        
        # Rename columns to standard names
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and new_name not in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)
        
        # Check required columns
        if 'id' not in df.columns:
            # Generate IDs if not present
            df['id'] = [str(uuid.uuid4()) for _ in range(len(df))]
        
        if 'content' not in df.columns:
            raise ValueError("CSV must have a 'content', 'text', 'body', or 'comment_body' column")
        
        if 'title' not in df.columns:
            df['title'] = ""  # Use empty string if no title
        
        # Fill missing optional columns
        for col in ['author', 'date', 'url']:
            if col not in df.columns:
                df[col] = ""
    
    print(f"Found {len(df)} talk posts in CSV")
    
    # Get ChromaDB collection
    collection = get_or_create_collection(collection_name)
    
    # Process each post
    all_chunks = []
    for idx, row in df.iterrows():
        post_id = str(row['id'])
        title = str(row['title']) if pd.notna(row['title']) else ""
        content = str(row['content']) if pd.notna(row['content']) else ""
        author = str(row['author']) if 'author' in df.columns and pd.notna(row['author']) else ""
        date = str(row['date']) if 'date' in df.columns and pd.notna(row['date']) else ""
        url = str(row['url']) if 'url' in df.columns and pd.notna(row['url']) else ""
        
        if not content or content.strip() == "":
            print(f"  Skipping post {post_id}: empty content")
            continue
        
        print(f"  Processing post {idx+1}/{len(df)}: {title[:50]}...")
        chunks = process_talk_post(
            post_id=post_id,
            title=title,
            content=content,
            author=author,
            date=date,
            url=url,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model_name=embedding_model_name
        )
        all_chunks.extend(chunks)
    
    # Store all chunks in ChromaDB
    if all_chunks:
        print(f"\nStoring {len(all_chunks)} chunks in vector database...")
        
        # Prepare data for ChromaDB
        ids = [chunk["id"] for chunk in all_chunks]
        texts = [chunk["text"] for chunk in all_chunks]
        embeddings_list = [chunk["embedding"] for chunk in all_chunks]
        metadatas = [
            {
                "source_type": chunk["metadata"]["source_type"],
                "post_id": chunk["metadata"]["post_id"],
                "title": chunk["metadata"]["title"],
                "author": chunk["metadata"]["author"],
                "date": chunk["metadata"]["date"],
                "url": chunk["metadata"]["url"],
                "chunk_index": chunk["metadata"]["chunk_index"],
                "filename": chunk["metadata"]["filename"],
                "page_numbers": "[]"  # For compatibility
            }
            for chunk in all_chunks
        ]
        
        # Add to ChromaDB in batches
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
    
    return len(df)


def add_single_talk_post(
    post_id: str,
    title: str,
    content: str,
    author: Optional[str] = None,
    date: Optional[str] = None,
    url: Optional[str] = None,
    collection_name: str = "pdf_chunks",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_model_name: str = "all-MiniLM-L6-v2"
) -> int:
    """
    Add a single talk post to ChromaDB.
    
    Args:
        post_id: Unique identifier for the post
        title: Post title
        content: Post content
        author: Post author (optional)
        date: Post date (optional)
        url: Post URL (optional)
        collection_name: ChromaDB collection name
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap between chunks
        embedding_model_name: Embedding model name
        
    Returns:
        Number of chunks created
    """
    # Process the post
    chunks = process_talk_post(
        post_id=post_id,
        title=title,
        content=content,
        author=author or "",
        date=date or "",
        url=url or "",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model_name=embedding_model_name
    )
    
    if not chunks:
        return 0
    
    # Get collection and store
    collection = get_or_create_collection(collection_name)
    
    ids = [chunk["id"] for chunk in chunks]
    texts = [chunk["text"] for chunk in chunks]
    embeddings_list = [chunk["embedding"] for chunk in chunks]
    metadatas = [
        {
            "source_type": chunk["metadata"]["source_type"],
            "post_id": chunk["metadata"]["post_id"],
            "title": chunk["metadata"]["title"],
            "author": chunk["metadata"]["author"],
            "date": chunk["metadata"]["date"],
            "url": chunk["metadata"]["url"],
            "chunk_index": chunk["metadata"]["chunk_index"],
            "filename": chunk["metadata"]["filename"],
            "page_numbers": "[]"
        }
        for chunk in chunks
    ]
    
    collection.add(
        ids=ids,
        embeddings=embeddings_list,
        documents=texts,
        metadatas=metadatas
    )
    
    print(f"✅ Added talk post '{title}' with {len(chunks)} chunks")
    return len(chunks)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest talk posts from CSV into ChromaDB vector database"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # CSV ingestion command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest posts from CSV file")
    ingest_parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to CSV file with talk posts"
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
    
    # Single post command
    add_parser = subparsers.add_parser("add", help="Add a single talk post")
    add_parser.add_argument("--post_id", type=str, required=True, help="Unique post ID")
    add_parser.add_argument("--title", type=str, required=True, help="Post title")
    add_parser.add_argument("--content", type=str, required=True, help="Post content")
    add_parser.add_argument("--author", type=str, default="", help="Post author")
    add_parser.add_argument("--date", type=str, default="", help="Post date")
    add_parser.add_argument("--url", type=str, default="", help="Post URL")
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        count = ingest_talk_posts_from_csv(
            csv_path=args.csv_path,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            embedding_model_name=args.embedding_model
        )
        print(f"\n✅ Processed {count} talk posts")
    elif args.command == "add":
        count = add_single_talk_post(
            post_id=args.post_id,
            title=args.title,
            content=args.content,
            author=args.author,
            date=args.date,
            url=args.url
        )
        print(f"\n✅ Added talk post with {count} chunks")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
