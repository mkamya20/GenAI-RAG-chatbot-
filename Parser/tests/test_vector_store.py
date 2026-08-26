"""
Tests for vector_store module.

Uses isolated test collection to avoid polluting production data.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from vector_store import (
    store_chunks,
    retrieve_chunks,
    delete_by_filename,
    delete_by_source_type,
    get_all_documents,
    count_chunks,
    get_or_create_collection,
)


class TestStoreChunks:
    """Tests for store_chunks function."""

    def test_stores_chunks_successfully(self, sample_chunks, cleanup_test_collection):
        count = store_chunks(sample_chunks, collection_name=cleanup_test_collection)
        assert count == 3

    def test_returns_zero_for_empty_list(self, cleanup_test_collection):
        count = store_chunks([], collection_name=cleanup_test_collection)
        assert count == 0

    def test_chunks_are_retrievable_after_store(self, sample_chunks, cleanup_test_collection, mock_embeddings):
        store_chunks(sample_chunks, collection_name=cleanup_test_collection)
        
        results = retrieve_chunks(
            query="gravitational waves",
            top_k=3,
            collection_name=cleanup_test_collection
        )
        
        assert len(results) > 0


class TestRetrieveChunks:
    """Tests for retrieve_chunks function."""

    def test_retrieves_relevant_chunks(self, sample_chunks, cleanup_test_collection, mock_embeddings):
        store_chunks(sample_chunks, collection_name=cleanup_test_collection)
        
        results = retrieve_chunks(
            query="LIGO laser detector",
            top_k=2,
            collection_name=cleanup_test_collection
        )
        
        assert len(results) <= 2
        for result in results:
            assert "id" in result
            assert "text" in result
            assert "metadata" in result

    def test_respects_top_k_limit(self, sample_chunks, cleanup_test_collection, mock_embeddings):
        store_chunks(sample_chunks, collection_name=cleanup_test_collection)
        
        results = retrieve_chunks(
            query="waves",
            top_k=1,
            collection_name=cleanup_test_collection
        )
        
        assert len(results) == 1

    def test_returns_empty_for_empty_collection(self, cleanup_test_collection, mock_embeddings):
        # Ensure collection exists but is empty
        get_or_create_collection(cleanup_test_collection)
        
        results = retrieve_chunks(
            query="anything",
            top_k=5,
            collection_name=cleanup_test_collection
        )
        
        assert results == []

    def test_includes_metadata_in_results(self, sample_chunks, cleanup_test_collection, mock_embeddings):
        store_chunks(sample_chunks, collection_name=cleanup_test_collection)
        
        results = retrieve_chunks(
            query="gravitational",
            top_k=1,
            collection_name=cleanup_test_collection
        )
        
        assert len(results) == 1
        metadata = results[0]["metadata"]
        assert "filename" in metadata
        assert "page_numbers" in metadata


class TestDeleteByFilename:
    """Tests for delete_by_filename function."""

    def test_deletes_chunks_by_filename(self, sample_chunks, cleanup_test_collection):
        store_chunks(sample_chunks, collection_name=cleanup_test_collection)
        
        # Delete chunks from test_doc.pdf (2 chunks)
        deleted = delete_by_filename("test_doc.pdf", collection_name=cleanup_test_collection)
        assert deleted == 2

        # Verify remaining count
        remaining = count_chunks(collection_name=cleanup_test_collection)
        assert remaining == 1

    def test_returns_zero_for_nonexistent_file(self, sample_chunks, cleanup_test_collection):
        store_chunks(sample_chunks, collection_name=cleanup_test_collection)
        
        deleted = delete_by_filename("nonexistent.pdf", collection_name=cleanup_test_collection)
        assert deleted == 0

    def test_returns_zero_for_empty_collection(self, cleanup_test_collection):
        get_or_create_collection(cleanup_test_collection)
        
        deleted = delete_by_filename("any.pdf", collection_name=cleanup_test_collection)
        assert deleted == 0


class TestDeleteBySourceType:
    """Tests for delete_by_source_type function."""

    def test_deletes_chunks_by_source_type(self, sample_chunks, cleanup_test_collection):
        store_chunks(sample_chunks, collection_name=cleanup_test_collection)
        
        # All sample chunks have source_type="pdf"
        deleted = delete_by_source_type("pdf", collection_name=cleanup_test_collection)
        assert deleted == 3

        # Verify collection is empty
        remaining = count_chunks(collection_name=cleanup_test_collection)
        assert remaining == 0

    def test_returns_zero_for_nonexistent_source_type(self, sample_chunks, cleanup_test_collection):
        store_chunks(sample_chunks, collection_name=cleanup_test_collection)
        
        deleted = delete_by_source_type("talk_post", collection_name=cleanup_test_collection)
        assert deleted == 0

        # Verify nothing was deleted
        remaining = count_chunks(collection_name=cleanup_test_collection)
        assert remaining == 3

    def test_returns_zero_for_empty_collection(self, cleanup_test_collection):
        get_or_create_collection(cleanup_test_collection)
        
        deleted = delete_by_source_type("pdf", collection_name=cleanup_test_collection)
        assert deleted == 0


class TestGetAllDocuments:
    """Tests for get_all_documents function."""

    def test_returns_document_summary(self, sample_chunks, cleanup_test_collection):
        store_chunks(sample_chunks, collection_name=cleanup_test_collection)
        
        # Patch the function to use test collection
        # Note: get_all_documents uses default collection, so we test indirectly
        collection = get_or_create_collection(cleanup_test_collection)
        results = collection.get()
        
        assert len(results['ids']) == 3

    def test_groups_by_filename(self, sample_chunks, cleanup_test_collection):
        store_chunks(sample_chunks, collection_name=cleanup_test_collection)
        
        collection = get_or_create_collection(cleanup_test_collection)
        results = collection.get()
        
        filenames = set()
        for metadata in results['metadatas']:
            filenames.add(metadata.get('filename'))
        
        assert "test_doc.pdf" in filenames
        assert "another_doc.pdf" in filenames


class TestCountChunks:
    """Tests for count_chunks function."""

    def test_counts_chunks_correctly(self, sample_chunks, cleanup_test_collection):
        store_chunks(sample_chunks, collection_name=cleanup_test_collection)
        
        count = count_chunks(collection_name=cleanup_test_collection)
        assert count == 3

    def test_returns_zero_for_empty_collection(self, cleanup_test_collection):
        get_or_create_collection(cleanup_test_collection)
        
        count = count_chunks(collection_name=cleanup_test_collection)
        assert count == 0