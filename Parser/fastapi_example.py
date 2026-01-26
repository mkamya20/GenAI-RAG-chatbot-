"""
Complete FastAPI application for Gravity Spy PDF Search & Chatbot.
Connects ChromaDB vector database with Azure OpenAI for RAG-based Q&A.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import AzureOpenAI
from openai import APIError
import os
import uuid
from typing import List, Optional, Dict
from pathlib import Path
from dotenv import load_dotenv
from ingest_pdfs import (
    retrieve_chunks, 
    get_embedding_model,
    get_or_create_collection,
    process_pdfs
)
from ingest_talk_posts import add_single_talk_post, ingest_talk_posts_from_csv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Gravity Spy PDF Search & Chatbot API",
    description="RAG-based document search and Q&A system",
    version="1.0.0"
)

# CORS middleware for web integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Azure OpenAI client
try:
    azure_client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    AZURE_OPENAI_AVAILABLE = True
except Exception as e:
    print(f"Warning: Azure OpenAI not configured: {e}")
    azure_client = None
    AZURE_OPENAI_AVAILABLE = False

# Request/Response models
class ChatRequest(BaseModel):
    query: str
    top_k: int = 5
    use_rag: bool = True  # If False, just return chunks without OpenAI

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict]
    chunks_used: int

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    filter_filename: Optional[str] = None

class SearchResponse(BaseModel):
    results: List[Dict]
    total: int

class DocumentInfo(BaseModel):
    filename: str
    chunk_count: int
    pages: List[int]

class HealthResponse(BaseModel):
    status: str
    chromadb_available: bool
    azure_openai_available: bool
    total_chunks: int

# Helper function to get document list from ChromaDB
def get_documents_info() -> List[DocumentInfo]:
    """Get list of all documents and their metadata."""
    try:
        collection = get_or_create_collection()
        results = collection.get()
        
        # Group chunks by filename
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
            # Parse page numbers
            page_str = metadata.get('page_numbers', '[]')
            try:
                import ast
                pages = ast.literal_eval(page_str)
                if isinstance(pages, list):
                    docs[filename]['pages'].update(pages)
            except:
                pass
        
        return [
            DocumentInfo(
                filename=filename,
                chunk_count=info['chunk_count'],
                pages=sorted(list(info['pages']))
            )
            for filename, info in docs.items()
        ]
    except Exception as e:
        print(f"Error getting documents info: {e}")
        return []

@app.get("/")
async def root():
    """Serve the frontend HTML file."""
    return FileResponse("frontend.html")

@app.get("/api", response_model=Dict)
async def api_info():
    """API information endpoint."""
    return {
        "name": "Gravity Spy PDF Search & Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/api/chat",
            "search": "/api/search",
            "documents": "/api/documents",
            "health": "/api/health",
            "docs": "/docs"
        }
    }

@app.get("/api/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    try:
        collection = get_or_create_collection()
        total_chunks = collection.count()
    except:
        total_chunks = 0
    
    return HealthResponse(
        status="ok",
        chromadb_available=total_chunks > 0,
        azure_openai_available=AZURE_OPENAI_AVAILABLE,
        total_chunks=total_chunks
    )

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint with RAG (Retrieval-Augmented Generation).
    
    1. Searches ChromaDB for relevant chunks
    2. Sends chunks to Azure OpenAI for answer generation
    3. Returns answer with sources
    """
    try:
        # Initialize answer variable
        answer = None
        
        # Step 1: Search ChromaDB for relevant chunks
        relevant_chunks = retrieve_chunks(
            query=request.query,
            top_k=request.top_k,
            use_vector_db=True
        )
        
        if not relevant_chunks:
            raise HTTPException(
                status_code=404,
                detail="No relevant chunks found. Make sure documents are processed."
            )
        
        # Step 2: Format chunks as context
        context_parts = []
        for chunk in relevant_chunks:
            filename = chunk['metadata'].get('filename', 'Unknown')
            pages = chunk['metadata'].get('page_numbers', [])
            text = chunk['text']
            context_parts.append(
                f"[Document: {filename}, Page(s): {pages}]\n{text}"
            )
        context = "\n\n".join(context_parts)
        
        # Step 3: Generate answer with Azure OpenAI (if enabled)
        if request.use_rag and AZURE_OPENAI_AVAILABLE:
            try:
                response = azure_client.chat.completions.create(
                    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4"),
                    messages=[
                        {
                            "role": "system",
                            "content": """You are a helpful assistant for the Gravity Spy project, 
                            a citizen science initiative for gravitational wave detection. 
                            Answer questions based ONLY on the provided context from scientific documents.
                            If the context doesn't contain enough information, say so.
                            Always cite which document(s) you're referencing."""
                        },
                        {
                            "role": "user",
                            "content": f"""Context from documents:
{context}

Question: {request.query}

Please provide a clear answer based on the context above, and mention which document(s) you referenced."""
                        }
                    ],
                    max_completion_tokens=800
                )
                answer = response.choices[0].message.content
                
               
            except (APIError, Exception) as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Azure OpenAI error: {str(e)}")
        else:
            # Fallback: just return the first chunk's text
            if not AZURE_OPENAI_AVAILABLE:
                answer = f"Azure OpenAI not configured. Here are the relevant document excerpts:\n\n{context[:500]}..."
            else:
                answer = context[:500] + "..."
        
       
        
        # Step 5: Prepare sources
        sources = [
            {
                "filename": chunk['metadata'].get('filename', 'Unknown'),
                "page_numbers": chunk['metadata'].get('page_numbers', []),
                "title": chunk['metadata'].get('title'),
                "text_preview": chunk['text'][:300] + "..." if len(chunk['text']) > 300 else chunk['text']
            }
            for chunk in relevant_chunks
        ]
        
        
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            chunks_used=len(relevant_chunks)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/api/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Direct semantic search endpoint (returns chunks without OpenAI).
    Useful for getting raw search results.
    """
    try:
        # Search ChromaDB
        results = retrieve_chunks(
            query=request.query,
            top_k=request.top_k,
            use_vector_db=True
        )
        
        # Filter by filename if specified
        if request.filter_filename:
            results = [
                chunk for chunk in results
                if chunk['metadata'].get('filename', '').lower() == request.filter_filename.lower()
            ]
        
        # Format results
        formatted_results = [
            {
                "id": chunk.get('id', ''),
                "text": chunk['text'],
                "metadata": {
                    "filename": chunk['metadata'].get('filename', 'Unknown'),
                    "page_numbers": chunk['metadata'].get('page_numbers', []),
                    "title": chunk['metadata'].get('title'),
                    "chunk_index": chunk.get('chunk_index', 0)
                },
                "similarity_distance": chunk.get('distance')
            }
            for chunk in results
        ]
        
        return SearchResponse(
            results=formatted_results,
            total=len(formatted_results)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search error: {str(e)}"
        )

@app.get("/api/documents", response_model=List[DocumentInfo])
async def get_documents():
    """Get list of all documents in the database."""
    try:
        documents = get_documents_info()
        return documents
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving documents: {str(e)}"
        )

@app.get("/api/documents/{filename}", response_model=Dict)
async def get_document_info(filename: str):
    """Get detailed information about a specific document."""
    try:
        documents = get_documents_info()
        doc = next((d for d in documents if d.filename == filename), None)
        
        if not doc:
            raise HTTPException(
                status_code=404,
                detail=f"Document '{filename}' not found"
            )
        
        return {
            "filename": doc.filename,
            "chunk_count": doc.chunk_count,
            "pages": doc.pages,
            "total_pages": len(doc.pages) if doc.pages else 0
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving document info: {str(e)}"
        )

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and process a PDF file.
    Note: This saves the file and processes it. In production, you might want
    to add authentication and file validation.
    """
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported"
            )
        
        # Save uploaded file
        upload_dir = Path("data/pdfs")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process the PDF
        process_pdfs(
            input_dir=str(upload_dir),
            output_path="outputs/chunks.jsonl",
            chunk_size=1000,
            chunk_overlap=200
        )
        
        return {
            "message": f"File '{file.filename}' uploaded and processed successfully",
            "filename": file.filename,
            "status": "processed"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )

@app.delete("/api/documents/{filename}")
async def delete_document(filename: str):
    """
    Delete a document and its chunks from the database.
    Note: This removes chunks from ChromaDB. The original PDF file is not deleted.
    """
    try:
        collection = get_or_create_collection()
        
        # Get all chunks for this filename
        results = collection.get()
        chunk_ids_to_delete = []
        
        for i, chunk_id in enumerate(results['ids']):
            metadata = results['metadatas'][i] if results['metadatas'] else {}
            if metadata.get('filename', '') == filename:
                chunk_ids_to_delete.append(chunk_id)
        
        if not chunk_ids_to_delete:
            raise HTTPException(
                status_code=404,
                detail=f"Document '{filename}' not found in database"
            )
        
        # Delete chunks
        collection.delete(ids=chunk_ids_to_delete)
        
        return {
            "message": f"Deleted {len(chunk_ids_to_delete)} chunks for document '{filename}'",
            "filename": filename,
            "chunks_deleted": len(chunk_ids_to_delete)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting document: {str(e)}"
        )

# Talk Posts Endpoints
class TalkPostRequest(BaseModel):
    post_id: str
    title: str
    content: str
    author: Optional[str] = None
    date: Optional[str] = None
    url: Optional[str] = None

class TalkPostResponse(BaseModel):
    message: str
    post_id: str
    chunks_created: int

@app.post("/api/talk-posts", response_model=TalkPostResponse)
async def add_talk_post(post: TalkPostRequest):
    """
    Add a single talk post to the vector database dynamically.
    This endpoint allows you to add new talk posts as they're posted.
    """
    try:
        from ingest_talk_posts import add_single_talk_post
        chunks_created = add_single_talk_post(
            post_id=post.post_id,
            title=post.title,
            content=post.content,
            author=post.author or "",
            date=post.date or "",
            url=post.url or ""
        )
        
        return TalkPostResponse(
            message=f"Talk post '{post.title}' added successfully",
            post_id=post.post_id,
            chunks_created=chunks_created
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error adding talk post: {str(e)}"
        )

@app.post("/api/talk-posts/upload-csv")
async def upload_talk_posts_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file containing talk posts and ingest them into the database.
    
    Expected CSV columns:
    - id (or post_id): Unique identifier
    - title: Post title
    - content (or text, body): Post content
    - author: Post author (optional)
    - date (or created_at, posted_at): Post date (optional)
    - url: Post URL (optional)
    """
    try:
        from ingest_talk_posts import ingest_talk_posts_from_csv
        
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=400,
                detail="Only CSV files are supported"
            )
        
        # Save uploaded file temporarily
        upload_dir = Path("data/temp")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Ingest the CSV
        posts_count = ingest_talk_posts_from_csv(str(file_path))
        
        # Clean up temp file
        file_path.unlink()
        
        return {
            "message": f"Successfully ingested {posts_count} talk posts from CSV",
            "filename": file.filename,
            "posts_processed": posts_count,
            "status": "processed"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing CSV file: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
