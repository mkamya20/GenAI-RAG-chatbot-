"""
Tests for pdf_processor module and processor module chunking.

Tests pure functions without mocking, and process_pdfs with mocked embeddings.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from pdf_processor import clean_text, extract_title_from_text
from processor import split_into_chunks


class TestCleanText:
    """Tests for clean_text function."""

    def test_removes_hyphenated_line_breaks(self):
        text = "gravi-\ntational waves"
        assert clean_text(text) == "gravitational waves"

    def test_replaces_newlines_with_spaces(self):
        text = "line one\nline two\nline three"
        assert clean_text(text) == "line one line two line three"

    def test_collapses_multiple_spaces(self):
        text = "too    many     spaces"
        assert clean_text(text) == "too many spaces"

    def test_strips_whitespace(self):
        text = "   padded text   "
        assert clean_text(text) == "padded text"

    def test_handles_empty_string(self):
        assert clean_text("") == ""

    def test_combined_cleaning(self):
        text = "  gravi-\ntational   waves\nare    cool  "
        assert clean_text(text) == "gravitational waves are cool"


class TestSplitIntoChunks:
    """Tests for split_into_chunks function."""

    def test_short_text_single_chunk(self):
        text = "Short text."
        chunks = split_into_chunks(text, chunk_size=100, chunk_overlap=20)
        assert len(chunks) == 1
        assert chunks[0] == "Short text."

    def test_long_text_multiple_chunks(self):
        text = "Word " * 100  # 500 characters
        chunks = split_into_chunks(text, chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1

    def test_chunks_respect_max_size(self):
        text = "Word " * 100
        chunk_size = 100
        chunks = split_into_chunks(text, chunk_size=chunk_size, chunk_overlap=20)
        for chunk in chunks:
            assert len(chunk) <= chunk_size + 50  # Allow some flexibility for word boundaries

    def test_overlap_creates_redundancy(self):
        text = "The quick brown fox jumps over the lazy dog. " * 10
        chunks = split_into_chunks(text, chunk_size=100, chunk_overlap=30)
        
        if len(chunks) > 1:
            # Check that consecutive chunks share some content
            for i in range(len(chunks) - 1):
                # Last part of chunk i should appear in chunk i+1
                chunk_end = chunks[i][-30:]
                assert any(
                    word in chunks[i + 1] 
                    for word in chunk_end.split() 
                    if len(word) > 3
                )

    def test_empty_text(self):
        chunks = split_into_chunks("", chunk_size=100, chunk_overlap=20)
        assert chunks == []

    def test_whitespace_only_text(self):
        chunks = split_into_chunks("   \n\t  ", chunk_size=100, chunk_overlap=20)
        assert chunks == []

    def test_custom_separators(self):
        text = "Section A|Section B|Section C"
        chunks = split_into_chunks(
            text, 
            chunk_size=15, 
            chunk_overlap=0,
            separators=["|", " "]
        )
        assert len(chunks) >= 2


class TestExtractTitleFromText:
    """Tests for extract_title_from_text function."""

    def test_extracts_first_suitable_line(self):
        text = "Introduction to LIGO\n\nThis document describes..."
        title = extract_title_from_text(text)
        assert title == "Introduction to LIGO"

    def test_respects_max_length(self):
        text = "A" * 200 + "\n\nBody text here"
        title = extract_title_from_text(text, max_length=50)
        # Should skip the too-long first line
        assert title is None or len(title) <= 50

    def test_skips_short_lines(self):
        text = "Hi\n\nThis is the actual title line\n\nBody text"
        title = extract_title_from_text(text)
        assert title == "This is the actual title line"

    def test_returns_none_for_unsuitable_text(self):
        text = "A\nB\nC"  # All too short
        title = extract_title_from_text(text)
        assert title is None

    def test_uses_first_sentence_fallback(self):
        text = "This is a reasonable title sentence. And more content follows."
        title = extract_title_from_text(text)
        assert "title sentence" in title


class TestProcessPdfs:
    """Tests for process_pdfs function (requires mocking)."""

    def test_returns_zero_for_empty_directory(self, tmp_path, mock_embeddings):
        from pdf_processor import process_pdfs
        
        count = process_pdfs(input_dir=str(tmp_path))
        assert count == 0

    def test_returns_zero_for_nonexistent_directory(self, tmp_path, mock_embeddings):
        from pdf_processor import process_pdfs
        
        nonexistent = tmp_path / "does_not_exist"
        count = process_pdfs(input_dir=str(nonexistent))
        assert count == 0