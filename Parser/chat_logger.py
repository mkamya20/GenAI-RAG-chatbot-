"""
Chat interaction logger.

Appends each chat round as a JSON line to a log file.
Used in digest mode to accumulate rounds for periodic email summaries.
"""

import json
import fcntl
from datetime import datetime, timezone

import config
from logging_config import get_logger

logger = get_logger(__name__)


def log_chat_round(query: str, answer: str, sources: list[dict]) -> None:
    """
    Append a chat round to the log file as a JSON line.

    Uses file locking to handle concurrent writes safely.

    Args:
        query: The user's question
        answer: The generated answer
        sources: List of source dicts
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "answer": answer,
        "sources": [
            {
                "filename": s.get("filename", "Unknown"),
                "page_numbers": s.get("page_numbers", []),
            }
            for s in sources
        ],
    }

    try:
        config.CHAT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        with open(config.CHAT_LOG_PATH, "a", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps(entry) + "\n")
            fcntl.flock(f, fcntl.LOCK_UN)

        logger.debug("Chat round logged")

    except Exception as e:
        logger.error(f"Failed to log chat round: {e}")


def read_and_rotate() -> list[dict]:
    """
    Read all entries from the chat log and clear it.

    Returns:
        List of log entry dicts. Empty list if no log or no entries.
    """
    if not config.CHAT_LOG_PATH.exists():
        return []

    entries = []

    try:
        with open(config.CHAT_LOG_PATH, "r+", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)

            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping malformed log line: {line[:80]}")

            # Truncate the file
            f.seek(0)
            f.truncate()

            fcntl.flock(f, fcntl.LOCK_UN)

        if entries:
            logger.info(f"Read {len(entries)} entries from chat log")

    except Exception as e:
        logger.error(f"Failed to read/rotate chat log: {e}")

    return entries