"""
Health and info endpoints.
"""

import pathlib

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from models import HealthResponse
from vector_store import count_chunks
import azure_client

router = APIRouter(tags=["health"])


@router.get("/api", response_model=dict)
async def api_info():
    """API information endpoint."""
    return {
        "name": "Gravity Spy PDF Search & Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/api/chat",
            "search": "/api/search",
            "pdfs": "/api/pdfs",
            "health": "/api/health",
            "docs": "/docs"
        }
    }


@router.get("/api/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    try:
        total_chunks = count_chunks()
    except Exception:
        total_chunks = 0

    return HealthResponse(
        status="ok",
        chromadb_available=total_chunks > 0,
        azure_openai_available=azure_client.is_available(),
        total_chunks=total_chunks
    )


@router.get("/api/init-log", response_class=PlainTextResponse)
async def get_init_log():
    """
    Return the container initialization log.

    Useful for debugging startup issues in environments where
    the filesystem is not directly accessible (e.g., Kubernetes).
    """
    log_path = pathlib.Path("data/init.log")

    if not log_path.exists():
        return PlainTextResponse("No init log found.", status_code=404)

    return PlainTextResponse(log_path.read_text())