"""
Pydantic models for API request/response schemas.
"""

from typing import Optional

from pydantic import BaseModel

from config import DEFAULT_TOP_K


# =============================================================================
# Chat Models
# =============================================================================

class ChatRequest(BaseModel):
    query: str
    top_k: int = DEFAULT_TOP_K
    use_rag: bool = True


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    chunks_used: int


# =============================================================================
# Search Models
# =============================================================================

class SearchRequest(BaseModel):
    query: str
    top_k: int = DEFAULT_TOP_K
    filter_filename: Optional[str] = None


class SearchResponse(BaseModel):
    results: list[dict]
    total: int


# =============================================================================
# Document Models
# =============================================================================

class DocumentInfo(BaseModel):
    filename: str
    chunk_count: int
    pages: list[int]


# =============================================================================
# Health Models
# =============================================================================

class HealthResponse(BaseModel):
    status: str
    chromadb_available: bool
    azure_openai_available: bool
    total_chunks: int


# =============================================================================
# Talk Post Models
# =============================================================================

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