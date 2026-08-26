"""
API routers package.
"""

from routers.health import router as health_router
from routers.chat import router as chat_router
from routers.pdfs import router as pdfs_router

__all__ = [
    "health_router",
    "chat_router",
    "pdfs_router",
]