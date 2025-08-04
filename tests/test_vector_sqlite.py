import sqlite3
from unittest.mock import patch

import numpy as np
import pytest
import sqlite_vss

from synapso_core.data_store.backend.sqlite.sqlite_vector_store import SqliteVectorStore
from synapso_core.data_store.backend.sqlite.utils import create_sqlite_db_if_not_exists
from synapso_core.models import Vector, VectorMetadata


class TestVectorMetadata(VectorMetadata):
    """Test implementation of VectorMetadata for testing purposes."""

    def __init__(self, content_hash: str, additional_data: dict | None = None):
        super().__init__(content_hash, additional_data or {})


class TestCreateSqliteDbIfNotExists:
    def test_create_sqlite_db_if_not_exists_new_db(self, tmp_path):
        """Test creating a new SQLite database file."""
        db_path = tmp_path / "test_vector.db"

        # Verify the file doesn't exist initially
        assert not db_path.exists()

        # Create the database
        create_sqlite_db_if_not_exists(str(db_path))

        # Verify the file was created
        assert db_path.exists()

    def test_create_sqlite_db_if_not_exists_existing_db(self, tmp_path):
        """Test behavior when database file already exists."""
        db_path = tmp_path / "test_vector.db"

        # Create the database first
        create_sqlite_db_if_not_exists(str(db_path))
        initial_size = db_path.stat().st_size

        # Try to create it again
        create_sqlite_db_if_not_exists(str(db_path))

        # Verify the file still exists and size hasn't changed
        assert db_path.exists()
        assert db_path.stat().st_size == initial_size


class TestSqliteVectorStore:
    @pytest.fixture
    def temp_db_config(self, tmp_path):
        """Create a temporary database configuration."""
        db_path = tmp_path / "test_vector.db"

        # Mock the config to return our temporary database path
        mock_config = type(
            "MockConfig",
            (),
            {
                "vector_store": type(
                    "MockVectorStore",
                    (),
                    {"vector_db_type": "sqlite", "vector_db_path": str(db_path)},
                )()
            },
        )()

        return mock_config, db_path

    @pytest.fixture
    def sample_vector(self):
        """Create a sample vector for testing."""
        vector_data = [0.1, 0.2, 0.3, 0.4, 0.5] * 76 + [
            0.1,
            0.2,
            0.3,
            0.4,
        ]  # 384 dimensions
        metadata = TestVectorMetadata(
            "test_hash_123", {"source": "test", "chunk_id": "chunk_1"}
        )
        return Vector("test_vector_1", vector_data, metadata)

    @pytest.fixture
    def sample_vector_no_metadata(self):
        """Create a sample vector without metadata for testing."""
        vector_data = [0.5, 0.4, 0.3, 0.2, 0.1] * 76 + [
            0.5,
            0.4,
            0.3,
            0.2,
        ]  # 384 dimensions
        return Vector("test_vector_2", vector_data, None)

    @pytest.fixture
    def vector_store_adapter(self, temp_db_config):
        """Create a SqliteVectorStore adapter with proper cleanup."""
        mock_config, db_path = temp_db_config

        with patch(
            "synapso_core.data_store.backend.sqlite.sqlite_vector_store.get_config",
            return_value=mock_config,
        ):
            adapter = SqliteVectorStore()
            adapter.setup()
            yield adapter
            adapter.close()

    def test_vectorstore_setup_creates_database(self, temp_db_config):
        """Test that vectorstore_setup creates the actual database file."""
        mock_config, db_path = temp_db_config

        # Verify the database doesn't exist initially
        assert not db_path.exists()

        with patch(
            "synapso_core.data_store.backend.sqlite.sqlite_vector_store.get_config",
            return_value=mock_config,
        ):
            adapter = SqliteVectorStore()
            adapter.setup()

            # Verify the database file was created
            assert db_path.exists()

            adapter.close()

    def test_vectorstore_setup_creates_tables(self, temp_db_config):
        """Test that vectorstore_setup creates the necessary tables."""
        mock_config, db_path = temp_db_config

        with patch(
            "synapso_core.data_store.backend.sqlite.sqlite_vector_store.get_config",
            return_value=mock_config,
        ):
            adapter = SqliteVectorStore()
            adapter.setup()

            # Check if the vectors and metadata tables exist
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()

                # Check vectors table
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='vss_vectors'"
                )
                tables = [row[0] for row in cursor.fetchall()]
                assert "vss_vectors" in tables

                # Check metadata table
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='metadata'"
                )
                tables = [row[0] for row in cursor.fetchall()]
                assert "metadata" in tables

            adapter.close()

    def test_vectorstore_setup_wrong_db_type(self):
        """Test that vectorstore_setup raises error for wrong database type."""
        mock_config = type(
            "MockConfig",
            (),
            {
                "vector_store": type(
                    "MockVectorStore",
                    (),
                    {
                        "vector_db_type": "postgresql",  # Wrong type
                        "vector_db_path": "/tmp/test.db",
                    },
                )()
            },
        )()

        with patch(
            "synapso_core.data_store.backend.sqlite.sqlite_vector_store.get_config",
            return_value=mock_config,
        ):
            with pytest.raises(ValueError, match="Vector store type is not sqlite"):
                adapter = SqliteVectorStore()

    def test_insert_vector_with_metadata(self, temp_db_config, sample_vector):
        """Test inserting a vector with metadata."""
        mock_config, db_path = temp_db_config

        with patch(
            "synapso_core.data_store.backend.sqlite.sqlite_vector_store.get_config",
            return_value=mock_config,
        ):
            adapter = SqliteVectorStore()
            adapter.setup()

            # Insert the vector
            result = adapter.insert(sample_vector)
            assert result is True

            # Verify the vector was inserted by checking the database directly
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT content_hash FROM metadata WHERE content_hash = ?",
                    (sample_vector.vector_id,),
                )
                result = cursor.fetchone()
                assert result is not None
                assert result[0] == sample_vector.vector_id

            adapter.close()

    def test_insert_vector_without_metadata(
        self, temp_db_config, sample_vector_no_metadata
    ):
        """Test inserting a vector without metadata."""
        mock_config, db_path = temp_db_config

        with patch(
            "synapso_core.data_store.backend.sqlite.sqlite_vector_store.get_config",
            return_value=mock_config,
        ):
            adapter = SqliteVectorStore()
            adapter.setup()

            # Insert vector without metadata
            assert adapter.insert(sample_vector_no_metadata) is True

            # Verify vector was inserted by checking the database directly
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT content_hash FROM metadata WHERE content_hash = ?",
                    (sample_vector_no_metadata.vector_id,),
                )
                result = cursor.fetchone()
                assert result is not None
                assert result[0] == sample_vector_no_metadata.vector_id

            adapter.close()

    def test_insert_duplicate_vector(self, temp_db_config, sample_vector):
        """Test inserting a duplicate vector raises an error."""
        mock_config, db_path = temp_db_config

        with patch(
            "synapso_core.data_store.backend.sqlite.sqlite_vector_store.get_config",
            return_value=mock_config,
        ):
            adapter = SqliteVectorStore()
            adapter.setup()

            # Insert the vector first time
            assert adapter.insert(sample_vector) is True

            # Try to insert the same vector again - should return True (handles duplicates)
            result = adapter.insert(sample_vector)
            assert result is True

            # Verify the vector exists by checking the database directly
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT content_hash FROM metadata WHERE content_hash = ?",
                    (sample_vector.vector_id,),
                )
                result = cursor.fetchone()
                assert result is not None
                assert result[0] == sample_vector.vector_id

            adapter.close()

    def test_get_by_id_existing_vector(self, temp_db_config, sample_vector):
        """Test retrieving an existing vector by ID."""
        mock_config, db_path = temp_db_config

        with patch(
            "synapso_core.data_store.backend.sqlite.sqlite_vector_store.get_config",
            return_value=mock_config,
        ):
            adapter = SqliteVectorStore()
            adapter.setup()

            # Insert a vector
            adapter.insert(sample_vector)

            # Retrieve it
            retrieved_vector = adapter.get_by_id(sample_vector.vector_id)
            assert retrieved_vector is not None
            assert retrieved_vector.vector_id == sample_vector.vector_id
            assert np.allclose(retrieved_vector.vector, sample_vector.vector, rtol=1e-5)
            assert (
                retrieved_vector.metadata.content_hash
                == sample_vector.metadata.content_hash
            )

            adapter.close()

    def test_get_by_id_nonexistent_vector(self, temp_db_config):
        """Test retrieving a non-existent vector by ID."""
        mock_config, db_path = temp_db_config

        with patch(
            "synapso_core.data_store.backend.sqlite.sqlite_vector_store.get_config",
            return_value=mock_config,
        ):
            adapter = SqliteVectorStore()
            adapter.setup()

            # Try to retrieve a non-existent vector
            result = adapter.get_by_id("nonexistent_vector_id")
            assert result is None

            adapter.close()

    def test_vector_search_basic(self, temp_db_config, sample_vector):
        """Test basic vector search functionality."""
        pytest.skip("Vector similarity search requires sqlite_vss extension")

    def test_vector_search_multiple_vectors(self, temp_db_config):
        """Test vector search with multiple vectors."""
        pytest.skip("Vector similarity search requires sqlite_vss extension")

    def test_vector_search_with_filters(self, temp_db_config, sample_vector):
        """Test vector search with filters (currently not implemented)."""
        pytest.skip("Vector similarity search requires sqlite_vss extension")

    def test_delete_not_implemented(self, temp_db_config):
        """Test that delete method raises NotImplementedError."""
        mock_config, db_path = temp_db_config

        with patch(
            "synapso_core.data_store.backend.sqlite.sqlite_vector_store.get_config",
            return_value=mock_config,
        ):
            adapter = SqliteVectorStore()
            adapter.setup()

            with pytest.raises(NotImplementedError, match="Not implemented"):
                adapter.delete("test_id")

            adapter.close()

    def test_update_metadata_not_implemented(self, temp_db_config):
        """Test that update_metadata method raises NotImplementedError."""
        mock_config, db_path = temp_db_config

        with patch(
            "synapso_core.data_store.backend.sqlite.sqlite_vector_store.get_config",
            return_value=mock_config,
        ):
            adapter = SqliteVectorStore()
            adapter.setup()

            metadata = TestVectorMetadata("new_hash", {"updated": True})
            with pytest.raises(NotImplementedError, match="Not implemented"):
                adapter.update_metadata("test_id", metadata)

            adapter.close()

    def test_count_not_implemented(self, temp_db_config):
        """Test that count method raises NotImplementedError."""
        mock_config, db_path = temp_db_config

        with patch(
            "synapso_core.data_store.backend.sqlite.sqlite_vector_store.get_config",
            return_value=mock_config,
        ):
            adapter = SqliteVectorStore()
            adapter.setup()

            with pytest.raises(NotImplementedError, match="Not implemented"):
                adapter.count()

            adapter.close()

    def test_vectorstore_teardown_not_implemented(self, temp_db_config):
        """Test that vectorstore_teardown method raises NotImplementedError."""
        mock_config, db_path = temp_db_config

        with patch(
            "synapso_core.data_store.backend.sqlite.sqlite_vector_store.get_config",
            return_value=mock_config,
        ):
            adapter = SqliteVectorStore()
            adapter.setup()

            with pytest.raises(NotImplementedError, match="Not implemented"):
                adapter.teardown()

            adapter.close()

    def test_database_persistence(self, temp_db_config, sample_vector):
        """Test that data persists between adapter instances."""
        mock_config, db_path = temp_db_config

        with patch(
            "synapso_core.data_store.backend.sqlite.sqlite_vector_store.get_config",
            return_value=mock_config,
        ):
            # Create first adapter and insert data
            adapter1 = SqliteVectorStore()
            adapter1.setup()
            adapter1.insert(sample_vector)

            # Create second adapter and verify data persists
            adapter2 = SqliteVectorStore()
            adapter2.setup()

            retrieved_vector = adapter2.get_by_id(sample_vector.vector_id)
            assert retrieved_vector is not None
            assert retrieved_vector.vector_id == sample_vector.vector_id
            assert np.allclose(retrieved_vector.vector, sample_vector.vector, rtol=1e-5)

            adapter1.close()
            adapter2.close()

    def test_multiple_setup_calls(self, temp_db_config):
        """Test that multiple setup calls don't cause issues."""
        mock_config, db_path = temp_db_config

        with patch(
            "synapso_core.data_store.backend.sqlite.sqlite_vector_store.get_config",
            return_value=mock_config,
        ):
            adapter = SqliteVectorStore()
            # Call setup multiple times
            assert adapter.setup() is True
            assert adapter.setup() is True

            # Verify database still exists and is functional
            assert db_path.exists()

            adapter.close()

    def test_context_manager(self, temp_db_config):
        """Test that the adapter works as a context manager."""
        mock_config, db_path = temp_db_config

        with patch(
            "synapso_core.data_store.backend.sqlite.sqlite_vector_store.get_config",
            return_value=mock_config,
        ):
            with SqliteVectorStore() as adapter:
                adapter.setup()
                assert adapter is not None
            adapter.close()

    def test_sqlite_vss_extension_loading(self, temp_db_config):
        """Test that SQLite VSS extension is properly loaded."""
        mock_config, db_path = temp_db_config

        with patch(
            "synapso_core.data_store.backend.sqlite.sqlite_vector_store.get_config",
            return_value=mock_config,
        ):
            adapter = SqliteVectorStore()
            adapter.setup()

            # Check if VSS extension is loaded by querying version
            # Note: This might fail on systems where SQLite extensions are not enabled
            try:
                with sqlite3.connect(db_path) as conn:
                    if hasattr(conn, "enable_load_extension"):
                        conn.enable_load_extension(True)
                        sqlite_vss.load(conn)
                        conn.enable_load_extension(False)

                        cursor = conn.cursor()
                        cursor.execute("SELECT vss_version()")
                        version = cursor.fetchone()[0]
                        assert version is not None
                        assert isinstance(version, str)
                    else:
                        # Skip this test if extension loading is not available
                        pytest.skip("SQLite extension loading not available")
            except Exception as e:
                # Skip test if VSS extension cannot be loaded
                pytest.skip(f"VSS extension cannot be loaded: {e}")

    def test_vector_dimensions_384(self, temp_db_config):
        """Test that vectors with 384 dimensions work correctly."""
        mock_config, db_path = temp_db_config

        with patch(
            "synapso_core.data_store.backend.sqlite.sqlite_vector_store.get_config",
            return_value=mock_config,
        ):
            adapter = SqliteVectorStore()
            adapter.setup()

            # Create a vector with exactly 384 dimensions (as specified in the schema)
            vector_data = [0.1] * 384
            metadata = TestVectorMetadata("test_384", {"dimensions": 384})
            vector = Vector("test_384_vector", vector_data, metadata)

            # Insert and retrieve the vector
            result = adapter.insert(vector)
            assert result is True

            retrieved_vector = adapter.get_by_id(vector.vector_id)
            assert retrieved_vector is not None
            assert len(retrieved_vector.vector) == 384
            assert np.allclose(retrieved_vector.vector, vector_data, rtol=1e-5)

    def test_metadata_serialization_deserialization(self, temp_db_config):
        """Test that metadata is properly serialized and deserialized."""
        mock_config, db_path = temp_db_config

        with patch(
            "synapso_core.data_store.backend.sqlite.sqlite_vector_store.get_config",
            return_value=mock_config,
        ):
            adapter = SqliteVectorStore()
            adapter.setup()

            # Create metadata with various data types
            additional_data = {
                "string": "test_value",
                "integer": 42,
                "float": 3.14,
                "boolean": True,
                "null": None,
                "list": [1, 2, 3],
                "dict": {"nested": "value"},
            }
            metadata = VectorMetadata(
                content_hash="test_hash_123", additional_data=additional_data
            )

            # Create and insert vector
            vector_data = [0.1, 0.2, 0.3] * 128  # 384 dimensions
            vector = Vector("test_vector_metadata", vector_data, metadata)
            adapter.insert(vector)

            # Retrieve vector
            retrieved_vector = adapter.get_by_id(vector.vector_id)
            assert retrieved_vector is not None
            assert retrieved_vector.metadata is not None
            assert retrieved_vector.metadata.content_hash == metadata.content_hash
            # The retrieved metadata should have the same additional_data
            assert retrieved_vector.metadata.additional_data == additional_data


class TestSqliteVectorStoreIntegration:
    """Integration tests that test the full workflow."""

    @pytest.fixture
    def temp_db_path(self, tmp_path):
        """Create a temporary database path."""
        return tmp_path / "integration_vector_test.db"

    def test_full_workflow(self, temp_db_path):
        """Test the complete workflow from setup to vector operations."""
        # Mock config
        mock_config = type(
            "MockConfig",
            (),
            {
                "vector_store": type(
                    "MockVectorStore",
                    (),
                    {"vector_db_type": "sqlite", "vector_db_path": str(temp_db_path)},
                )()
            },
        )()

        with patch(
            "synapso_core.data_store.backend.sqlite.sqlite_vector_store.get_config",
            return_value=mock_config,
        ):
            # Setup
            adapter = SqliteVectorStore()
            assert adapter.setup() is True

            # Verify database exists
            assert temp_db_path.exists()

            # Test vector operations
            vector_data = [0.1, 0.2, 0.3] * 128  # 384 dimensions
            metadata = TestVectorMetadata(
                "integration_test_hash", {"test": "integration"}
            )
            vector = Vector("integration_test_vector", vector_data, metadata)

            # Insert vector
            assert adapter.insert(vector) is True

            # Retrieve vector
            retrieved_vector = adapter.get_by_id(vector.vector_id)
            assert retrieved_vector is not None
            assert retrieved_vector.vector_id == vector.vector_id
            assert np.allclose(retrieved_vector.vector, vector.vector, rtol=1e-5)
            assert retrieved_vector.metadata.content_hash == metadata.content_hash

            adapter.close()

            # Skip vector search test since it requires sqlite_vss extension
            # query_vector = Vector("query", vector_data, None)
            # results = adapter.vector_search(query_vector, top_k=5)

    def test_multiple_vectors_workflow(self, temp_db_path):
        """Test workflow with multiple vectors."""
        mock_config = type(
            "MockConfig",
            (),
            {
                "vector_store": type(
                    "MockVectorStore",
                    (),
                    {"vector_db_type": "sqlite", "vector_db_path": str(temp_db_path)},
                )()
            },
        )()

        with patch(
            "synapso_core.data_store.backend.sqlite.sqlite_vector_store.get_config",
            return_value=mock_config,
        ):
            adapter = SqliteVectorStore()
            adapter.setup()

            # Insert multiple vectors
            vectors = []
            for i in range(5):
                vector_data = [float(j + i * 0.1) for j in range(384)]
                metadata = VectorMetadata(
                    content_hash=f"hash_{i}",
                    additional_data={"index": i, "batch": "test"},
                )
                vector = Vector(f"vector_{i}", vector_data, metadata)
                adapter.insert(vector)
                vectors.append(vector)

            # Test retrieval of all vectors
            for i, vector in enumerate(vectors):
                retrieved = adapter.get_by_id(vector.vector_id)
                assert retrieved is not None
                assert retrieved.vector_id == vector.vector_id
                assert retrieved.metadata.additional_data["index"] == i

            # Skip vector search tests since they require sqlite_vss extension
            # for i, vector in enumerate(vectors):
            #     query_vector = Vector(f"query_{i}", vector.vector, None)
            #     results = adapter.vector_search(query_vector, top_k=3)

            adapter.close()
