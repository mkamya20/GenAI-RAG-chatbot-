"""
Email notification for chat interactions.

Supports two modes:
  - "round": Sends an email per chat round (immediate)
  - "digest": Accumulates rounds in a log, sends periodic digest emails

Designed to be called from FastAPI BackgroundTasks (round mode)
or a lifespan-managed async loop (digest mode).
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone

import config
from logging_config import get_logger

logger = get_logger(__name__)


def is_enabled() -> bool:
    """Check if email notifications are configured and enabled."""
    return (
        config.NOTIFY_ENABLED
        and bool(config.SMTP_HOST)
        and bool(config.NOTIFY_FROM)
        and bool(config.NOTIFY_TO)
    )


def send_chat_notification(query: str, answer: str, sources: list[dict]) -> None:
    """
    Send an email notification for a single chat round.

    Used in "round" mode.

    Args:
        query: The user's question
        answer: The generated answer
        sources: List of source dicts with 'filename', 'page_numbers', etc.
    """
    if not is_enabled():
        return

    try:
        recipients = _get_recipients()
        if not recipients:
            return

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        subject = f"Gravity Spy Chat: {_truncate(query, 60)}"

        text_body = _format_round_text(query, answer, sources, timestamp)
        html_body = _format_round_html(query, answer, sources, timestamp)

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = config.NOTIFY_FROM
        msg["To"] = ", ".join(recipients)

        msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        _send(msg, recipients)
        logger.debug(f"Chat notification sent to {recipients}")

    except Exception as e:
        logger.error(f"Failed to send chat notification: {e}")


def send_digest(entries: list[dict]) -> None:
    """
    Send a digest email summarizing multiple chat rounds.

    Used in "digest" mode. Called by the lifespan digest loop.

    Args:
        entries: List of log entry dicts from chat_logger.read_and_rotate()
    """
    if not is_enabled() or not entries:
        return

    try:
        recipients = _get_recipients()
        if not recipients:
            return

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        count = len(entries)
        subject = f"Gravity Spy Chat Digest: {count} interaction{'s' if count != 1 else ''}"

        text_body = _format_digest_text(entries, timestamp)
        html_body = _format_digest_html(entries, timestamp)

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = config.NOTIFY_FROM
        msg["To"] = ", ".join(recipients)

        msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        _send(msg, recipients)
        logger.info(f"Digest email sent: {count} interactions to {recipients}")

    except Exception as e:
        logger.error(f"Failed to send digest email: {e}")


# =============================================================================
# Private Helpers
# =============================================================================

def _get_recipients() -> list[str]:
    """Parse the comma-separated recipient list."""
    return [addr.strip() for addr in config.NOTIFY_TO.split(",") if addr.strip()]


def _send(msg: MIMEMultipart, recipients: list[str]) -> None:
    """Send the email via SMTP."""
    with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT) as server:
        if config.SMTP_USE_TLS:
            server.starttls()
        if config.SMTP_USERNAME and config.SMTP_PASSWORD:
            server.login(config.SMTP_USERNAME, config.SMTP_PASSWORD)
        server.sendmail(config.NOTIFY_FROM, recipients, msg.as_string())


def _truncate(text: str, max_length: int) -> str:
    """Truncate text for use in email subject."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("\n", "<br>")
    )


def _format_sources_text(sources: list[dict]) -> str:
    """Format sources as plain text lines."""
    if not sources:
        return ""
    lines = ["Sources:"]
    for source in sources:
        filename = source.get("filename", "Unknown")
        pages = source.get("page_numbers", [])
        page_info = f" (pages {', '.join(str(p) for p in pages)})" if pages else ""
        lines.append(f"  - {filename}{page_info}")
    return "\n".join(lines)


def _format_sources_html(sources: list[dict]) -> str:
    """Format sources as an HTML list."""
    if not sources:
        return ""
    items = []
    for source in sources:
        filename = source.get("filename", "Unknown")
        pages = source.get("page_numbers", [])
        page_info = f" (pages {', '.join(str(p) for p in pages)})" if pages else ""
        items.append(f"<li>{_escape_html(filename)}{page_info}</li>")
    return "<strong>Sources:</strong><ul>" + "".join(items) + "</ul>"


# =============================================================================
# Round Formatting (single Q&A)
# =============================================================================

def _format_round_text(query: str, answer: str, sources: list[dict], timestamp: str) -> str:
    """Format a single round as plain text."""
    lines = [
        "Gravity Spy Chat Notification",
        f"Time: {timestamp}",
        "",
        "Question:",
        query,
        "",
        "Answer:",
        answer,
    ]
    sources_text = _format_sources_text(sources)
    if sources_text:
        lines.extend(["", sources_text])
    return "\n".join(lines)


def _format_round_html(query: str, answer: str, sources: list[dict], timestamp: str) -> str:
    """Format a single round as HTML."""
    return f"""
    <div style="font-family: -apple-system, sans-serif; max-width: 600px; margin: 0 auto;">
        <h3 style="color: #667eea; margin-bottom: 4px;">Gravity Spy Chat</h3>
        <p style="color: #999; font-size: 0.85em; margin-top: 0;">{timestamp}</p>

        <div style="margin-bottom: 16px;">
            <strong>Question:</strong>
            <div style="background: #f0f0f0; padding: 12px; border-radius: 8px; margin-top: 4px;">
                {_escape_html(query)}
            </div>
        </div>

        <div style="margin-bottom: 16px;">
            <strong>Answer:</strong>
            <div style="background: #f9f9f9; padding: 12px; border-radius: 8px; margin-top: 4px; border: 1px solid #e0e0e0;">
                {_escape_html(answer)}
            </div>
        </div>

        {f'<div>{_format_sources_html(sources)}</div>' if sources else ''}
    </div>
    """


# =============================================================================
# Digest Formatting (multiple rounds)
# =============================================================================

def _format_digest_text(entries: list[dict], timestamp: str) -> str:
    """Format multiple rounds as a plain text digest."""
    count = len(entries)
    lines = [
        f"Gravity Spy Chat Digest",
        f"Generated: {timestamp}",
        f"Interactions: {count}",
        "",
    ]

    for i, entry in enumerate(entries, 1):
        entry_time = entry.get("timestamp", "unknown time")
        lines.append(f"--- Interaction {i} ({entry_time}) ---")
        lines.append(f"Q: {entry.get('query', '')}")
        lines.append(f"A: {entry.get('answer', '')}")
        sources_text = _format_sources_text(entry.get("sources", []))
        if sources_text:
            lines.append(sources_text)
        lines.append("")

    return "\n".join(lines)


def _format_digest_html(entries: list[dict], timestamp: str) -> str:
    """Format multiple rounds as an HTML digest."""
    count = len(entries)

    rounds_html = ""
    for i, entry in enumerate(entries, 1):
        entry_time = entry.get("timestamp", "unknown time")
        query = entry.get("query", "")
        answer = entry.get("answer", "")
        sources = entry.get("sources", [])

        rounds_html += f"""
        <div style="margin-bottom: 24px; padding-bottom: 24px; border-bottom: 1px solid #eee;">
            <p style="color: #999; font-size: 0.8em; margin: 0 0 8px 0;">
                Interaction {i} &middot; {_escape_html(entry_time)}
            </p>
            <div style="margin-bottom: 8px;">
                <strong>Q:</strong>
                <div style="background: #f0f0f0; padding: 10px; border-radius: 6px; margin-top: 4px;">
                    {_escape_html(query)}
                </div>
            </div>
            <div style="margin-bottom: 8px;">
                <strong>A:</strong>
                <div style="background: #f9f9f9; padding: 10px; border-radius: 6px; margin-top: 4px; border: 1px solid #e0e0e0;">
                    {_escape_html(answer)}
                </div>
            </div>
            {f'<div style="font-size: 0.85em;">{_format_sources_html(sources)}</div>' if sources else ''}
        </div>
        """

    return f"""
    <div style="font-family: -apple-system, sans-serif; max-width: 600px; margin: 0 auto;">
        <h3 style="color: #667eea; margin-bottom: 4px;">Gravity Spy Chat Digest</h3>
        <p style="color: #999; font-size: 0.85em; margin-top: 0;">
            {timestamp} &middot; {count} interaction{'s' if count != 1 else ''}
        </p>
        {rounds_html}
    </div>
    """