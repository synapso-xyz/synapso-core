from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from synapso_core.ingestor.document_ingestor import ingest_file


class TestIngestFile:
    """Test cases for the ingest_file function."""

    @patch("synapso_core.ingestor.document_ingestor.get_config")
    @patch("synapso_core.ingestor.document_ingestor.ChunkerFactory")
    @patch("synapso_core.ingestor.document_ingestor.VectorizerFactory")
    @patch("synapso_core.ingestor.document_ingestor.DataStoreFactory")
    def test_ingest_file_success(
        self,
        mock_data_store_factory,
        mock_vectorizer_factory,
        mock_chunker_factory,
        mock_get_config,
    ):
        """Test successful file ingestion."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.chunker.chunker_type = "test_chunker"
        mock_config.vectorizer.vectorizer_type = "test_vectorizer"
        mock_config.vector_store.vector_db_type = "test_vector_store"
        mock_config.private_store.private_db_type = "test_private_store"
        mock_get_config.return_value = mock_config

        # Create mock chunks with text attribute
        mock_chunk1 = MagicMock()
        mock_chunk1.text = "chunk1"
        mock_chunk2 = MagicMock()
        mock_chunk2.text = "chunk2"
        chunks = [mock_chunk1, mock_chunk2]

        mock_chunker = MagicMock()
        mock_chunker.chunk_file.return_value = chunks
        mock_chunker_factory.create_chunker.return_value = mock_chunker

        mock_vectorizer = MagicMock()
        mock_vectorizer.vectorize_batch.return_value = ["vector1", "vector2"]
        mock_vectorizer_factory.create_vectorizer.return_value = mock_vectorizer

        mock_vector_store = MagicMock()
        mock_private_store = MagicMock()
        mock_data_store_factory.get_vector_store.return_value = mock_vector_store
        mock_data_store_factory.get_private_store.return_value = mock_private_store

        # Test
        file_path = Path("/test/file.md")
        success, error_context = ingest_file(file_path)

        # Assertions
        assert success is True
        assert error_context is None

        mock_chunker_factory.create_chunker.assert_called_once_with("test_chunker")
        mock_vectorizer_factory.create_vectorizer.assert_called_once_with(
            "test_vectorizer"
        )
        mock_data_store_factory.get_vector_store.assert_called_once_with(
            "test_vector_store"
        )
        mock_data_store_factory.get_private_store.assert_called_once_with(
            "test_private_store"
        )

        mock_chunker.chunk_file.assert_called_once_with(str(file_path))
        mock_vectorizer.vectorize_batch.assert_called_once_with(chunks)

        # Verify private store insert was called for each chunk
        assert mock_private_store.insert.call_count == 2
        mock_private_store.insert.assert_any_call("chunk1")
        mock_private_store.insert.assert_any_call("chunk2")

        # Verify vector store insert was called for each vector
        assert mock_vector_store.insert.call_count == 2
        mock_vector_store.insert.assert_any_call("vector1")
        mock_vector_store.insert.assert_any_call("vector2")

    @patch("synapso_core.ingestor.document_ingestor.get_config")
    @patch("synapso_core.ingestor.document_ingestor.ChunkerFactory")
    @patch("synapso_core.ingestor.document_ingestor.VectorizerFactory")
    @patch("synapso_core.ingestor.document_ingestor.DataStoreFactory")
    def test_ingest_file_empty_chunks(
        self,
        mock_data_store_factory,
        mock_vectorizer_factory,
        mock_chunker_factory,
        mock_get_config,
    ):
        """Test file ingestion with empty chunks."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.chunker.chunker_type = "test_chunker"
        mock_config.vectorizer.vectorizer_type = "test_vectorizer"
        mock_config.vector_store.vector_db_type = "test_vector_store"
        mock_config.private_store.private_db_type = "test_private_store"
        mock_get_config.return_value = mock_config

        mock_chunker = MagicMock()
        mock_chunker.chunk_file.return_value = []  # Empty chunks
        mock_chunker_factory.create_chunker.return_value = mock_chunker

        mock_vectorizer = MagicMock()
        mock_vectorizer.vectorize_batch.return_value = []  # Empty vectors
        mock_vectorizer_factory.create_vectorizer.return_value = mock_vectorizer

        mock_vector_store = MagicMock()
        mock_private_store = MagicMock()
        mock_data_store_factory.get_vector_store.return_value = mock_vector_store
        mock_data_store_factory.get_private_store.return_value = mock_private_store

        # Test
        file_path = Path("/test/empty.md")
        success, error_context = ingest_file(file_path)

        # Assertions
        assert success is True
        assert error_context is None

        mock_chunker.chunk_file.assert_called_once_with(str(file_path))
        mock_vectorizer.vectorize_batch.assert_called_once_with([])
        mock_private_store.insert.assert_not_called()
        mock_vector_store.insert.assert_not_called()

    @patch("synapso_core.ingestor.document_ingestor.get_config")
    @patch("synapso_core.ingestor.document_ingestor.ChunkerFactory")
    @patch("synapso_core.ingestor.document_ingestor.VectorizerFactory")
    @patch("synapso_core.ingestor.document_ingestor.DataStoreFactory")
    def test_ingest_file_chunker_error(
        self,
        mock_data_store_factory,
        mock_vectorizer_factory,
        mock_chunker_factory,
        mock_get_config,
    ):
        """Test file ingestion when chunker fails."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.chunker.chunker_type = "test_chunker"
        mock_config.vectorizer.vectorizer_type = "test_vectorizer"
        mock_config.vector_store.vector_db_type = "test_vector_store"
        mock_config.private_store.private_db_type = "test_private_store"
        mock_get_config.return_value = mock_config

        mock_chunker = MagicMock()
        mock_chunker.chunk_file.side_effect = Exception("Chunker failed")
        mock_chunker_factory.create_chunker.return_value = mock_chunker

        mock_vectorizer = MagicMock()
        mock_vectorizer_factory.create_vectorizer.return_value = mock_vectorizer

        mock_vector_store = MagicMock()
        mock_private_store = MagicMock()
        mock_data_store_factory.get_vector_store.return_value = mock_vector_store
        mock_data_store_factory.get_private_store.return_value = mock_private_store

        # Test
        file_path = Path("/test/file.md")
        success, error_context = ingest_file(file_path)

        # Assertions
        assert success is False
        assert error_context is not None
        assert "error_type" in error_context
        assert "traceback" in error_context
        assert error_context["error_type"] == "Chunker failed"

    @patch("synapso_core.ingestor.document_ingestor.get_config")
    @patch("synapso_core.ingestor.document_ingestor.ChunkerFactory")
    @patch("synapso_core.ingestor.document_ingestor.VectorizerFactory")
    @patch("synapso_core.ingestor.document_ingestor.DataStoreFactory")
    def test_ingest_file_vectorizer_error(
        self,
        mock_data_store_factory,
        mock_vectorizer_factory,
        mock_chunker_factory,
        mock_get_config,
    ):
        """Test file ingestion when vectorizer fails."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.chunker.chunker_type = "test_chunker"
        mock_config.vectorizer.vectorizer_type = "test_vectorizer"
        mock_config.vector_store.vector_db_type = "test_vector_store"
        mock_config.private_store.private_db_type = "test_private_store"
        mock_get_config.return_value = mock_config

        mock_chunk = MagicMock()
        mock_chunk.text = "chunk1"
        chunks = [mock_chunk]

        mock_chunker = MagicMock()
        mock_chunker.chunk_file.return_value = chunks
        mock_chunker_factory.create_chunker.return_value = mock_chunker

        mock_vectorizer = MagicMock()
        mock_vectorizer.vectorize_batch.side_effect = Exception("Vectorizer failed")
        mock_vectorizer_factory.create_vectorizer.return_value = mock_vectorizer

        mock_vector_store = MagicMock()
        mock_private_store = MagicMock()
        mock_data_store_factory.get_vector_store.return_value = mock_vector_store
        mock_data_store_factory.get_private_store.return_value = mock_private_store

        # Test
        file_path = Path("/test/file.md")
        success, error_context = ingest_file(file_path)

        # Assertions
        assert success is False
        assert error_context is not None
        assert error_context["error_type"] == "Vectorizer failed"

    @patch("synapso_core.ingestor.document_ingestor.get_config")
    @patch("synapso_core.ingestor.document_ingestor.ChunkerFactory")
    @patch("synapso_core.ingestor.document_ingestor.VectorizerFactory")
    @patch("synapso_core.ingestor.document_ingestor.DataStoreFactory")
    def test_ingest_file_vector_store_error(
        self,
        mock_data_store_factory,
        mock_vectorizer_factory,
        mock_chunker_factory,
        mock_get_config,
    ):
        """Test file ingestion when vector store fails."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.chunker.chunker_type = "test_chunker"
        mock_config.vectorizer.vectorizer_type = "test_vectorizer"
        mock_config.vector_store.vector_db_type = "test_vector_store"
        mock_config.private_store.private_db_type = "test_private_store"
        mock_get_config.return_value = mock_config

        mock_chunk = MagicMock()
        mock_chunk.text = "chunk1"
        chunks = [mock_chunk]

        mock_chunker = MagicMock()
        mock_chunker.chunk_file.return_value = chunks
        mock_chunker_factory.create_chunker.return_value = mock_chunker

        mock_vectorizer = MagicMock()
        mock_vectorizer.vectorize_batch.return_value = ["vector1"]
        mock_vectorizer_factory.create_vectorizer.return_value = mock_vectorizer

        mock_vector_store = MagicMock()
        mock_vector_store.insert.side_effect = Exception("Vector store failed")
        mock_private_store = MagicMock()
        mock_data_store_factory.get_vector_store.return_value = mock_vector_store
        mock_data_store_factory.get_private_store.return_value = mock_private_store

        # Test
        file_path = Path("/test/file.md")
        success, error_context = ingest_file(file_path)

        # Assertions
        assert success is False
        assert error_context is not None
        assert error_context["error_type"] == "Vector store failed"

    @patch("synapso_core.ingestor.document_ingestor.get_config")
    def test_ingest_file_config_error(self, mock_get_config):
        """Test file ingestion when config loading fails."""
        # Setup mocks
        mock_get_config.side_effect = Exception("Config failed")

        # Test
        file_path = Path("/test/file.md")
        success, error_context = ingest_file(file_path)

        # Assertions
        assert success is False
        assert error_context is not None
        assert error_context["error_type"] == "Config failed"

    @patch("synapso_core.ingestor.document_ingestor.get_config")
    @patch("synapso_core.ingestor.document_ingestor.ChunkerFactory")
    @patch("synapso_core.ingestor.document_ingestor.VectorizerFactory")
    @patch("synapso_core.ingestor.document_ingestor.DataStoreFactory")
    def test_ingest_file_mismatched_chunks_vectors(
        self,
        mock_data_store_factory,
        mock_vectorizer_factory,
        mock_chunker_factory,
        mock_get_config,
    ):
        """Test file ingestion with mismatched chunks and vectors."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.chunker.chunker_type = "test_chunker"
        mock_config.vectorizer.vectorizer_type = "test_vectorizer"
        mock_config.vector_store.vector_db_type = "test_vector_store"
        mock_config.private_store.private_db_type = "test_private_store"
        mock_get_config.return_value = mock_config

        # Create mock chunks with text attribute
        mock_chunk1 = MagicMock()
        mock_chunk1.text = "chunk1"
        mock_chunk2 = MagicMock()
        mock_chunk2.text = "chunk2"
        mock_chunk3 = MagicMock()
        mock_chunk3.text = "chunk3"
        chunks = [mock_chunk1, mock_chunk2, mock_chunk3]

        mock_chunker = MagicMock()
        mock_chunker.chunk_file.return_value = chunks
        mock_chunker_factory.create_chunker.return_value = mock_chunker

        mock_vectorizer = MagicMock()
        mock_vectorizer.vectorize_batch.return_value = [
            "vector1",
            "vector2",
        ]  # Fewer vectors than chunks
        mock_vectorizer_factory.create_vectorizer.return_value = mock_vectorizer

        mock_vector_store = MagicMock()
        mock_private_store = MagicMock()
        mock_data_store_factory.get_vector_store.return_value = mock_vector_store
        mock_data_store_factory.get_private_store.return_value = mock_private_store

        # Test
        file_path = Path("/test/file.md")
        success, error_context = ingest_file(file_path)

        # Assertions
        assert success is True
        assert error_context is None

        # Should insert all chunks to private store
        assert mock_private_store.insert.call_count == 3
        mock_private_store.insert.assert_any_call("chunk1")
        mock_private_store.insert.assert_any_call("chunk2")
        mock_private_store.insert.assert_any_call("chunk3")

        # Should only insert the vectors that were created
        assert mock_vector_store.insert.call_count == 2
        mock_vector_store.insert.assert_any_call("vector1")
        mock_vector_store.insert.assert_any_call("vector2")

    @patch("synapso_core.ingestor.document_ingestor.get_config")
    @patch("synapso_core.ingestor.document_ingestor.ChunkerFactory")
    @patch("synapso_core.ingestor.document_ingestor.VectorizerFactory")
    @patch("synapso_core.ingestor.document_ingestor.DataStoreFactory")
    def test_ingest_file_with_different_file_types(
        self,
        mock_data_store_factory,
        mock_vectorizer_factory,
        mock_chunker_factory,
        mock_get_config,
    ):
        """Test file ingestion with different file path types."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.chunker.chunker_type = "test_chunker"
        mock_config.vectorizer.vectorizer_type = "test_vectorizer"
        mock_config.vector_store.vector_db_type = "test_vector_store"
        mock_config.private_store.private_db_type = "test_private_store"
        mock_get_config.return_value = mock_config

        mock_chunk = MagicMock()
        mock_chunk.text = "chunk1"
        chunks = [mock_chunk]

        mock_chunker = MagicMock()
        mock_chunker.chunk_file.return_value = chunks
        mock_chunker_factory.create_chunker.return_value = mock_chunker

        mock_vectorizer = MagicMock()
        mock_vectorizer.vectorize_batch.return_value = ["vector1"]
        mock_vectorizer_factory.create_vectorizer.return_value = mock_vectorizer

        mock_vector_store = MagicMock()
        mock_private_store = MagicMock()
        mock_data_store_factory.get_vector_store.return_value = mock_vector_store
        mock_data_store_factory.get_private_store.return_value = mock_private_store

        # Test with string path
        string_path = "/test/file.md"
        success, error_context = ingest_file(Path(string_path))
        assert success is True
        assert error_context is None
        mock_chunker.chunk_file.assert_called_with(string_path)

        # Reset mocks
        mock_chunker.chunk_file.reset_mock()
        mock_vector_store.insert.reset_mock()
        mock_private_store.insert.reset_mock()

        # Test with Path object
        path_obj = Path("/test/file.md")
        success, error_context = ingest_file(path_obj)
        assert success is True
        assert error_context is None
        mock_chunker.chunk_file.assert_called_with(str(path_obj))

    @patch("synapso_core.ingestor.document_ingestor.get_config")
    @patch("synapso_core.ingestor.document_ingestor.ChunkerFactory")
    @patch("synapso_core.ingestor.document_ingestor.VectorizerFactory")
    @patch("synapso_core.ingestor.document_ingestor.DataStoreFactory")
    def test_ingest_file_traceback_in_error_context(
        self,
        mock_data_store_factory,
        mock_vectorizer_factory,
        mock_chunker_factory,
        mock_get_config,
    ):
        """Test that error context includes proper traceback information."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.chunker.chunker_type = "test_chunker"
        mock_config.vectorizer.vectorizer_type = "test_vectorizer"
        mock_config.vector_store.vector_db_type = "test_vector_store"
        mock_config.private_store.private_db_type = "test_private_store"
        mock_get_config.return_value = mock_config

        mock_chunker = MagicMock()
        mock_chunker.chunk_file.side_effect = ValueError("Test error")
        mock_chunker_factory.create_chunker.return_value = mock_chunker

        mock_vectorizer = MagicMock()
        mock_vectorizer_factory.create_vectorizer.return_value = mock_vectorizer

        mock_vector_store = MagicMock()
        mock_private_store = MagicMock()
        mock_data_store_factory.get_vector_store.return_value = mock_vector_store
        mock_data_store_factory.get_private_store.return_value = mock_private_store

        # Test
        file_path = Path("/test/file.md")
        success, error_context = ingest_file(file_path)

        # Assertions
        assert success is False
        assert error_context is not None
        assert "error_type" in error_context
        assert "traceback" in error_context
        assert error_context["error_type"] == "Test error"
        assert isinstance(error_context["traceback"], str)

    @patch("synapso_core.ingestor.document_ingestor.get_config")
    @patch("synapso_core.ingestor.document_ingestor.ChunkerFactory")
    @patch("synapso_core.ingestor.document_ingestor.VectorizerFactory")
    @patch("synapso_core.ingestor.document_ingestor.DataStoreFactory")
    def test_ingest_file_large_number_of_chunks(
        self,
        mock_data_store_factory,
        mock_vectorizer_factory,
        mock_chunker_factory,
        mock_get_config,
    ):
        """Test file ingestion with a large number of chunks."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.chunker.chunker_type = "test_chunker"
        mock_config.vectorizer.vectorizer_type = "test_vectorizer"
        mock_config.vector_store.vector_db_type = "test_vector_store"
        mock_config.private_store.private_db_type = "test_private_store"
        mock_get_config.return_value = mock_config

        # Create many chunks
        chunks = []
        for i in range(100):
            mock_chunk = MagicMock()
            mock_chunk.text = f"chunk_{i}"
            chunks.append(mock_chunk)

        mock_chunker = MagicMock()
        mock_chunker.chunk_file.return_value = chunks
        mock_chunker_factory.create_chunker.return_value = mock_chunker

        vectors = [f"vector_{i}" for i in range(100)]
        mock_vectorizer = MagicMock()
        mock_vectorizer.vectorize_batch.return_value = vectors
        mock_vectorizer_factory.create_vectorizer.return_value = mock_vectorizer

        mock_vector_store = MagicMock()
        mock_private_store = MagicMock()
        mock_data_store_factory.get_vector_store.return_value = mock_vector_store
        mock_data_store_factory.get_private_store.return_value = mock_private_store

        # Test
        file_path = Path("/test/large_file.md")
        success, error_context = ingest_file(file_path)

        # Assertions
        assert success is True
        assert error_context is None

        # Verify all chunks were inserted to private store
        assert mock_private_store.insert.call_count == 100
        for i in range(100):
            mock_private_store.insert.assert_any_call(f"chunk_{i}")

        # Verify all vectors were inserted to vector store
        assert mock_vector_store.insert.call_count == 100
        for i in range(100):
            mock_vector_store.insert.assert_any_call(f"vector_{i}")

    @patch("synapso_core.ingestor.document_ingestor.get_config")
    @patch("synapso_core.ingestor.document_ingestor.ChunkerFactory")
    @patch("synapso_core.ingestor.document_ingestor.VectorizerFactory")
    @patch("synapso_core.ingestor.document_ingestor.DataStoreFactory")
    def test_ingest_file_partial_vector_store_failure(
        self,
        mock_data_store_factory,
        mock_vectorizer_factory,
        mock_chunker_factory,
        mock_get_config,
    ):
        """Test file ingestion when vector store fails on some inserts but not others."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.chunker.chunker_type = "test_chunker"
        mock_config.vectorizer.vectorizer_type = "test_vectorizer"
        mock_config.vector_store.vector_db_type = "test_vector_store"
        mock_config.private_store.private_db_type = "test_private_store"
        mock_get_config.return_value = mock_config

        # Create mock chunks with text attribute
        mock_chunk1 = MagicMock()
        mock_chunk1.text = "chunk1"
        mock_chunk2 = MagicMock()
        mock_chunk2.text = "chunk2"
        mock_chunk3 = MagicMock()
        mock_chunk3.text = "chunk3"
        chunks = [mock_chunk1, mock_chunk2, mock_chunk3]

        mock_chunker = MagicMock()
        mock_chunker.chunk_file.return_value = chunks
        mock_chunker_factory.create_chunker.return_value = mock_chunker

        mock_vectorizer = MagicMock()
        mock_vectorizer.vectorize_batch.return_value = ["vector1", "vector2", "vector3"]
        mock_vectorizer_factory.create_vectorizer.return_value = mock_vectorizer

        mock_vector_store = MagicMock()
        # Make insert fail on the second call
        mock_vector_store.insert.side_effect = [None, Exception("Insert failed"), None]
        mock_private_store = MagicMock()
        mock_data_store_factory.get_vector_store.return_value = mock_vector_store
        mock_data_store_factory.get_private_store.return_value = mock_private_store

        # Test
        file_path = Path("/test/file.md")
        success, error_context = ingest_file(file_path)

        # Assertions
        assert success is False
        assert error_context is not None
        assert error_context["error_type"] == "Insert failed"

        # Verify all chunks were inserted to private store
        assert mock_private_store.insert.call_count == 3
        mock_private_store.insert.assert_any_call("chunk1")
        mock_private_store.insert.assert_any_call("chunk2")
        mock_private_store.insert.assert_any_call("chunk3")

        # Verify first vector insert succeeded, second failed
        assert mock_vector_store.insert.call_count == 2  # Stops at first failure


# Integration tests
class TestDocumentIngestorIntegration:
    """Integration tests for document ingestion."""

    @pytest.fixture
    def sample_markdown_file(self, tmp_path):
        """Create a sample markdown file for testing."""
        file_path = tmp_path / "sample.md"
        content = """# Test Document

This is a test document with multiple paragraphs.

## Section 1

This is the first section with some content.

## Section 2

This is the second section with different content.

- List item 1
- List item 2
- List item 3

### Subsection

More content in a subsection.
"""
        file_path.write_text(content)
        return file_path

    @patch("synapso_core.ingestor.document_ingestor.get_config")
    @patch("synapso_core.ingestor.document_ingestor.ChunkerFactory")
    @patch("synapso_core.ingestor.document_ingestor.VectorizerFactory")
    @patch("synapso_core.ingestor.document_ingestor.DataStoreFactory")
    def test_ingest_real_markdown_file(
        self,
        mock_data_store_factory,
        mock_vectorizer_factory,
        mock_chunker_factory,
        mock_get_config,
        sample_markdown_file,
    ):
        """Test ingestion of a real markdown file."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.chunker.chunker_type = "test_chunker"
        mock_config.vectorizer.vectorizer_type = "test_vectorizer"
        mock_config.vector_store.vector_db_type = "test_vector_store"
        mock_config.private_store.private_db_type = "test_private_store"
        mock_get_config.return_value = mock_config

        # Simulate realistic chunking
        expected_chunks = [
            "# Test Document\n\nThis is a test document with multiple paragraphs.",
            "## Section 1\n\nThis is the first section with some content.",
            "## Section 2\n\nThis is the second section with different content.",
            "- List item 1\n- List item 2\n- List item 3\n\n### Subsection\n\nMore content in a subsection.",
        ]

        # Create mock chunks with text attribute
        mock_chunks = []
        for chunk_text in expected_chunks:
            mock_chunk = MagicMock()
            mock_chunk.text = chunk_text
            mock_chunks.append(mock_chunk)

        mock_chunker = MagicMock()
        mock_chunker.chunk_file.return_value = mock_chunks
        mock_chunker_factory.create_chunker.return_value = mock_chunker

        # Simulate realistic vectorization
        expected_vectors = [f"vector_{i}" for i in range(len(expected_chunks))]
        mock_vectorizer = MagicMock()
        mock_vectorizer.vectorize_batch.return_value = expected_vectors
        mock_vectorizer_factory.create_vectorizer.return_value = mock_vectorizer

        mock_vector_store = MagicMock()
        mock_private_store = MagicMock()
        mock_data_store_factory.get_vector_store.return_value = mock_vector_store
        mock_data_store_factory.get_private_store.return_value = mock_private_store

        # Test
        success, error_context = ingest_file(sample_markdown_file)

        # Assertions
        assert success is True
        assert error_context is None

        # Verify chunker was called with the correct file path
        mock_chunker.chunk_file.assert_called_once_with(str(sample_markdown_file))

        # Verify vectorizer was called with the chunks
        mock_vectorizer.vectorize_batch.assert_called_once_with(mock_chunks)

        # Verify all chunks were inserted to private store
        assert mock_private_store.insert.call_count == len(expected_chunks)
        for chunk_text in expected_chunks:
            mock_private_store.insert.assert_any_call(chunk_text)

        # Verify all vectors were inserted to vector store
        assert mock_vector_store.insert.call_count == len(expected_vectors)
        for vector in expected_vectors:
            mock_vector_store.insert.assert_any_call(vector)
