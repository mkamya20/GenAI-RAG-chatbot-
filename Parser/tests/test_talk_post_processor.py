"""
Tests for talk_post_processor module.

Tests pure functions without mocking, and ingestion functions with mocked embeddings.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from talk_post_processor import chunk_talk_post, _normalize_row


class TestChunkTalkPost:
    """Tests for chunk_talk_post function."""

    def test_creates_single_chunk_for_short_content(self):
        chunks = chunk_talk_post(
            post_id="123",
            title="Test Title",
            content="Short content.",
            chunk_size=1000,
            chunk_overlap=100
        )
        
        assert len(chunks) == 1
        assert chunks[0]["id"] == "talk_post_123_0"
        assert "Test Title" in chunks[0]["text"]
        assert "Short content" in chunks[0]["text"]

    def test_creates_multiple_chunks_for_long_content(self):
        long_content = "This is a sentence about gravitational waves. " * 50
        chunks = chunk_talk_post(
            post_id="456",
            title="Long Post",
            content=long_content,
            chunk_size=200,
            chunk_overlap=50
        )
        
        assert len(chunks) > 1
        for i, chunk in enumerate(chunks):
            assert chunk["id"] == f"talk_post_456_{i}"

    def test_metadata_is_populated(self):
        chunks = chunk_talk_post(
            post_id="789",
            title="Metadata Test",
            content="Testing metadata fields.",
            author="scientist",
            date="2024-01-15",
            url="https://example.com/post/789"
        )
        
        assert len(chunks) == 1
        metadata = chunks[0]["metadata"]
        
        assert metadata["source_type"] == "talk_post"
        assert metadata["post_id"] == "789"
        assert metadata["title"] == "Metadata Test"
        assert metadata["author"] == "scientist"
        assert metadata["date"] == "2024-01-15"
        assert metadata["url"] == "https://example.com/post/789"
        assert metadata["chunk_index"] == "0"
        assert metadata["filename"] == "talk_post_789"
        assert metadata["page_numbers"] == "[]"

    def test_handles_empty_title(self):
        chunks = chunk_talk_post(
            post_id="111",
            title="",
            content="Content without title."
        )
        
        assert len(chunks) == 1
        assert chunks[0]["text"] == "Content without title."

    def test_handles_empty_content(self):
        chunks = chunk_talk_post(
            post_id="222",
            title="Title Only",
            content=""
        )
        
        # Should still create a chunk from the title
        assert len(chunks) == 1
        assert "Title Only" in chunks[0]["text"]

    def test_default_optional_fields(self):
        chunks = chunk_talk_post(
            post_id="333",
            title="Minimal",
            content="Minimal content."
        )
        
        metadata = chunks[0]["metadata"]
        assert metadata["author"] == ""
        assert metadata["date"] == ""
        assert metadata["url"] == ""


class TestNormalizeRow:
    """Tests for _normalize_row function."""

    def test_gravity_spy_format(self):
        row = {
            "comment_id": "12345",
            "comment_body": "This is a comment about glitches.",
            "discussion_title": "Interesting Glitch",
            "comment_user_login": "researcher1",
            "comment_created_at": "2024-01-15T10:30:00Z",
            "board_id": "100",
            "discussion_id": "200"
        }
        
        result = _normalize_row(row)
        
        assert result is not None
        assert result["id"] == "12345"
        assert result["content"] == "This is a comment about glitches."
        assert result["title"] == "Interesting Glitch"
        assert result["author"] == "researcher1"
        assert result["date"] == "2024-01-15T10:30:00Z"
        assert result["url"] == "https://www.zooniverse.org/projects/zooniverse/gravity-spy/talk/100/200"

    def test_gravity_spy_format_without_board_id(self):
        row = {
            "comment_id": "12345",
            "comment_body": "Content here.",
            "discussion_title": "Title",
            "comment_user_login": "user",
            "comment_created_at": "2024-01-15",
        }
        
        result = _normalize_row(row)
        
        assert result is not None
        assert result["url"] == ""

    def test_gravity_spy_format_empty_content_returns_none(self):
        row = {
            "comment_id": "12345",
            "comment_body": "",
            "discussion_title": "Empty Post"
        }
        
        result = _normalize_row(row)
        assert result is None

    def test_gravity_spy_format_whitespace_content_returns_none(self):
        row = {
            "comment_id": "12345",
            "comment_body": "   \n\t  ",
            "discussion_title": "Whitespace Post"
        }
        
        result = _normalize_row(row)
        assert result is None

    def test_generic_format_with_content(self):
        row = {
            "id": "abc",
            "title": "Generic Title",
            "content": "Generic content here.",
            "author": "someone",
            "date": "2024-02-01",
            "url": "https://example.com"
        }
        
        result = _normalize_row(row)
        
        assert result is not None
        assert result["id"] == "abc"
        assert result["content"] == "Generic content here."
        assert result["title"] == "Generic Title"

    def test_generic_format_with_text_field(self):
        row = {
            "id": "xyz",
            "text": "Using text field instead."
        }
        
        result = _normalize_row(row)
        
        assert result is not None
        assert result["content"] == "Using text field instead."

    def test_generic_format_with_body_field(self):
        row = {
            "id": "xyz",
            "body": "Using body field instead."
        }
        
        result = _normalize_row(row)
        
        assert result is not None
        assert result["content"] == "Using body field instead."

    def test_generic_format_uses_post_id_fallback(self):
        row = {
            "post_id": "fallback_id",
            "content": "Some content."
        }
        
        result = _normalize_row(row)
        
        assert result is not None
        assert result["id"] == "fallback_id"

    def test_generic_format_generates_uuid_when_no_id(self):
        row = {
            "content": "Content without id."
        }
        
        result = _normalize_row(row)
        
        assert result is not None
        assert result["id"]  # Should have generated a UUID
        assert len(result["id"]) > 0

    def test_generic_format_date_fallbacks(self):
        row1 = {"content": "Test", "created_at": "2024-01-01"}
        row2 = {"content": "Test", "posted_at": "2024-02-02"}
        
        assert _normalize_row(row1)["date"] == "2024-01-01"
        assert _normalize_row(row2)["date"] == "2024-02-02"

    def test_generic_format_url_fallback(self):
        row = {"content": "Test", "link": "https://example.com/link"}
        
        result = _normalize_row(row)
        assert result["url"] == "https://example.com/link"

    def test_generic_format_empty_content_returns_none(self):
        row = {
            "id": "123",
            "title": "Has title but no content",
            "content": ""
        }
        
        result = _normalize_row(row)
        assert result is None

    def test_generic_format_missing_optional_fields(self):
        row = {
            "content": "Just content, nothing else."
        }
        
        result = _normalize_row(row)
        
        assert result is not None
        assert result["title"] == ""
        assert result["author"] == ""
        assert result["date"] == ""
        assert result["url"] == ""


class TestAddSingleTalkPost:
    """Tests for add_single_talk_post function."""

    def test_adds_post_successfully(self, mock_embeddings, cleanup_test_collection):
        from talk_post_processor import add_single_talk_post
        
        count = add_single_talk_post(
            post_id="test_001",
            title="Test Post",
            content="This is test content for the post.",
            collection_name=cleanup_test_collection
        )
        
        assert count >= 1

    def test_returns_zero_for_empty_content(self, mock_embeddings, cleanup_test_collection):
        from talk_post_processor import add_single_talk_post
        
        count = add_single_talk_post(
            post_id="test_002",
            title="",
            content="",
            collection_name=cleanup_test_collection
        )
        
        # Empty title + empty content = empty text = no chunks
        assert count == 0


class TestIngestTalkPostsFromCsv:
    """Tests for ingest_talk_posts_from_csv function."""

    def test_ingests_generic_csv(self, tmp_path, mock_embeddings, cleanup_test_collection):
        from talk_post_processor import ingest_talk_posts_from_csv
        
        csv_content = """id,title,content,author
1,First Post,This is the first post content.,alice
2,Second Post,This is the second post content.,bob
3,Third Post,This is the third post content.,charlie
"""
        csv_file = tmp_path / "posts.csv"
        csv_file.write_text(csv_content)
        
        count = ingest_talk_posts_from_csv(
            csv_path=str(csv_file),
            collection_name=cleanup_test_collection
        )
        
        assert count >= 3  # At least one chunk per post

    def test_ingests_gravity_spy_csv(self, tmp_path, mock_embeddings, cleanup_test_collection):
        from talk_post_processor import ingest_talk_posts_from_csv
        
        csv_content = """comment_id,comment_body,discussion_title,comment_user_login,board_id,discussion_id
101,Interesting glitch pattern here.,Glitch Discussion,scientist1,10,20
102,I agree this looks unusual.,Glitch Discussion,scientist2,10,20
"""
        csv_file = tmp_path / "gravity_spy.csv"
        csv_file.write_text(csv_content)
        
        count = ingest_talk_posts_from_csv(
            csv_path=str(csv_file),
            collection_name=cleanup_test_collection
        )
        
        assert count >= 2

    def test_skips_empty_content_rows(self, tmp_path, mock_embeddings, cleanup_test_collection):
        from talk_post_processor import ingest_talk_posts_from_csv
        
        csv_content = """id,title,content
1,Has Content,This row has content.
2,Empty Content,
3,Also Has Content,This row also has content.
"""
        csv_file = tmp_path / "mixed.csv"
        csv_file.write_text(csv_content)
        
        count = ingest_talk_posts_from_csv(
            csv_path=str(csv_file),
            collection_name=cleanup_test_collection
        )
        
        # Should process 2 posts (skipping the empty one)
        assert count >= 2

    def test_raises_for_missing_file(self, cleanup_test_collection):
        from talk_post_processor import ingest_talk_posts_from_csv
        
        with pytest.raises(FileNotFoundError):
            ingest_talk_posts_from_csv(
                csv_path="/nonexistent/path/file.csv",
                collection_name=cleanup_test_collection
            )

    def test_respects_batch_size(self, tmp_path, mock_embeddings, cleanup_test_collection):
        from talk_post_processor import ingest_talk_posts_from_csv
        
        # Create CSV with enough rows to trigger multiple batches
        rows = ["id,title,content"]
        for i in range(10):
            rows.append(f"{i},Post {i},Content for post {i} with enough text.")
        
        csv_file = tmp_path / "batched.csv"
        csv_file.write_text("\n".join(rows))
        
        count = ingest_talk_posts_from_csv(
            csv_path=str(csv_file),
            collection_name=cleanup_test_collection,
            batch_size=3  # Small batch size to test batching
        )
        
        assert count >= 10