"""
FastAPI application for Gravity Spy PDF Search & Chatbot.
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import config
import chat_logger
import email_notifier
from routers import (
    health_router,
    chat_router,
    pdfs_router,
)
from logging_config import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler. Starts the digest loop if configured."""
    digest_task = None

    if (
        config.NOTIFY_ENABLED
        and config.NOTIFY_MODE == "digest"
        and email_notifier.is_enabled()
    ):
        digest_task = asyncio.create_task(_digest_loop())
        logger.info(
            f"Digest email loop started "
            f"(interval: {config.NOTIFY_DIGEST_INTERVAL} minutes)"
        )

    yield

    # Shutdown: cancel the digest loop and flush any remaining entries
    if digest_task is not None:
        digest_task.cancel()
        try:
            await digest_task
        except asyncio.CancelledError:
            pass

        # Send any remaining entries on shutdown
        entries = chat_logger.read_and_rotate()
        if entries:
            email_notifier.send_digest(entries)
            logger.info(f"Flushed {len(entries)} entries on shutdown")


async def _digest_loop():
    """Periodically read the chat log and send a digest email."""
    interval_seconds = config.NOTIFY_DIGEST_INTERVAL * 60

    while True:
        await asyncio.sleep(interval_seconds)

        try:
            entries = chat_logger.read_and_rotate()
            if entries:
                email_notifier.send_digest(entries)
        except Exception as e:
            logger.error(f"Digest loop error: {e}")

app = FastAPI(
    title="Gravity Spy PDF Search & Chatbot API",
    description="RAG-based document search and Q&A system",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Mount routers
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(pdfs_router)


@app.get("/embed")
async def embed():
    """Serve the embeddable chat widget."""
    return FileResponse("templates/embed.html")


@app.get("/chat")
async def chat_page():
    """Serve the original full chat interface."""
    return FileResponse("templates/chat.html")


@app.get("/")
async def root():
    """Serve the embed demo page."""
    return FileResponse("templates/demo.html")


logger.info("FastAPI application initialized")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)