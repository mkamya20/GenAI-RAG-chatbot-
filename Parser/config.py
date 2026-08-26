"""
Application configuration.

Loads environment variables and provides centralized access to all settings.
"""

import os
import pathlib

import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()


# =============================================================================
# Path Configuration
# =============================================================================
PDF_DIR = pathlib.Path(os.getenv("PDF_DIR", "data/pdfs"))
WIKI_DIR = pathlib.Path(os.getenv("WIKI_DIR", "data/wiki"))
CSV_DIR = pathlib.Path(os.getenv("CSV_DIR", "data/csvs"))
TEMP_DIR = pathlib.Path(os.getenv("TEMP_DIR", "data/temp"))
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "data/chroma_db")


# =============================================================================
# Azure OpenAI Configuration
# =============================================================================
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "GravitySpy-gpt-4o-mini")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "GravitySpy-text-embedding-3-small")


# =============================================================================
# ChromaDB Configuration
# =============================================================================
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "pdf_chunks")


# =============================================================================
# Processing Defaults
# =============================================================================
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 5

# Batch sizes for processing and API calls
# Lower values reduce rate limiting but increase processing time
DEFAULT_BATCH_SIZE = int(os.getenv("DEFAULT_BATCH_SIZE", "100"))
DEFAULT_BATCH_DELAY = float(os.getenv("DEFAULT_BATCH_DELAY", "0.0"))
EMBEDDING_BATCH_SIZE = DEFAULT_BATCH_SIZE  # Alias for backward compatibility


# =============================================================================
# Email Notification Configuration
# =============================================================================
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
NOTIFY_FROM = os.getenv("NOTIFY_FROM", "")
NOTIFY_TO = os.getenv("NOTIFY_TO", "")  # Comma-separated list of recipients
NOTIFY_ENABLED = os.getenv("NOTIFY_ENABLED", "false").lower() == "true"
NOTIFY_MODE = os.getenv("NOTIFY_MODE", "round")  # "round" or "digest"
NOTIFY_DIGEST_INTERVAL = int(os.getenv("NOTIFY_DIGEST_INTERVAL", "60"))  # minutes
CHAT_LOG_PATH = pathlib.Path(os.getenv("CHAT_LOG_PATH", "data/chat_log.jsonl"))


# =============================================================================
# ZOTERO document Library
# =============================================================================
ZOTERO_API_KEY = os.getenv("ZOT_API_KEY")
ZOTERO_GROUP_ID = os.getenv("ZOT_GROUP_ID")
ZOTERO_COLLECTION_KEY = os.getenv("ZOT_COLLECTION_KEY")

# =============================================================================
# GSWIKI
# =============================================================================
WIKI_GRAPHQL_URL = os.getenv("WIKI_GRAPHQL_URL", "")
WIKI_API_TOKEN = os.getenv("WIKI_API_TOKEN", "")
WIKI_PUBLIC_URL = os.getenv("WIKI_PUBLIC_URL", "")
