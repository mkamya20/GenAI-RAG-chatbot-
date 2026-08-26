"""
Chat and search endpoints.
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException
from openai import APIError, APIConnectionError, RateLimitError

from models import ChatRequest, ChatResponse, SearchRequest, SearchResponse
from vector_store import retrieve_chunks
import azure_client
import chat_logger
import config
import email_notifier

router = APIRouter(prefix="/api", tags=["chat"])

# Response truncation limits
TEXT_PREVIEW_LENGTH = 300
CONTEXT_PREVIEW_LENGTH = 500


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    Chat endpoint with RAG (Retrieval-Augmented Generation).

    1. Searches ChromaDB for relevant chunks
    2. Sends chunks to Azure OpenAI for answer generation
    3. Returns answer with sources
    4. Sends email notification in background (if enabled)
    """
    relevant_chunks = retrieve_chunks(query=request.query, top_k=request.top_k)

    if not relevant_chunks:
        raise HTTPException(
            status_code=404,
            detail="No relevant chunks found. Make sure documents are processed."
        )

    context_parts = []
    for chunk in relevant_chunks:
        filename = chunk['metadata'].get('filename', 'Unknown')
        pages = chunk['metadata'].get('page_numbers', [])
        text = chunk['text']
        context_parts.append(f"[Document: {filename}, Page(s): {pages}]\n{text}")
    context = "\n\n".join(context_parts)

    if request.use_rag and azure_client.is_available():
        answer = _generate_rag_answer(request.query, context)
    else:
        if not azure_client.is_available():
            answer = (
                f"Azure OpenAI not configured. "
                f"Here are the relevant document excerpts:\n\n"
                f"{context[:CONTEXT_PREVIEW_LENGTH]}..."
            )
        else:
            answer = context[:CONTEXT_PREVIEW_LENGTH] + "..."

    sources = [
        {
            "filename": chunk['metadata'].get('filename', 'Unknown'),
            "page_numbers": chunk['metadata'].get('page_numbers', []),
            "title": chunk['metadata'].get('title'),
            "url": chunk['metadata'].get('url'),
            "text_preview": _truncate(chunk['text'], TEXT_PREVIEW_LENGTH)
        }
        for chunk in relevant_chunks
    ]

    if config.NOTIFY_MODE == "digest":
        background_tasks.add_task(
            chat_logger.log_chat_round,
            request.query,
            answer,
            sources,
        )
    else:
        background_tasks.add_task(
            email_notifier.send_chat_notification,
            request.query,
            answer,
            sources,
        )

    return ChatResponse(
        answer=answer,
        sources=sources,
        chunks_used=len(relevant_chunks)
    )


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Direct semantic search endpoint (returns chunks without OpenAI).
    """
    results = retrieve_chunks(query=request.query, top_k=request.top_k)

    if request.filter_filename:
        results = [
            chunk for chunk in results
            if chunk['metadata'].get('filename', '').lower() == request.filter_filename.lower()
        ]

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

    return SearchResponse(results=formatted_results, total=len(formatted_results))


def _generate_rag_answer(query: str, context: str) -> str:
    """
    Generate an answer using Azure OpenAI with the provided context.
    
    Args:
        query: User's question
        context: Retrieved document context
        
    Returns:
        Generated answer text
        
    Raises:
        HTTPException: On Azure API errors
    """
    try:
        return azure_client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": """You are a helpful assistant for the Gravity Spy project,
                    a citizen science initiative for gravitational wave detection.
                    Answer questions based ONLY on the provided context from scientific documents.
                    If the context doesn't contain enough information, say so.
                    Do not list or cite document names in your answer; source links are provided separately."""
                },
                {
                    "role": "user",
                    "content": f"""Context from documents:
{context}

Question: {query}

Please provide a clear answer based on the context above."""
                }
            ],
            max_tokens=800
        )
    except RateLimitError as e:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded: {e}")
    except APIConnectionError as e:
        raise HTTPException(status_code=503, detail=f"Azure OpenAI connection error: {e}")
    except APIError as e:
        raise HTTPException(status_code=502, detail=f"Azure OpenAI error: {e}")


def _truncate(text: str, max_length: int) -> str:
    """Truncate text with ellipsis if needed."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."