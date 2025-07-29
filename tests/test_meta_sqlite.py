from unittest.mock import patch

import pytest
from sqlalchemy import text

from src.synapso_core.persistence.implementations.meta_store.meta_sqlite import MetaSqliteAdapter
from src.synapso_core.sqlite_utils import create_sqlite_db_if_not_exists


class TestCreateSqliteDbIfNotExists:
    def test_create_sqlite_db_if_not_exists_new_db(self, tmp_path):
        """Test creating a new SQLite database file."""
        db_path = tmp_path / "test.db"

        # Verify the file doesn't exist initially
        assert not db_path.exists()

        # Create the database
        create_sqlite_db_if_not_exists(str(db_path))

        # Verify the file was created
        assert db_path.exists()
        # SQLite files can be empty initially, so we just check existence

    def test_create_sqlite_db_if_not_exists_existing_db(self, tmp_path):
        """Test behavior when database file already exists."""
        db_path = tmp_path / "test.db"

        # Create the database first
        create_sqlite_db_if_not_exists(str(db_path))
        initial_size = db_path.stat().st_size

        # Try to create it again
        create_sqlite_db_if_not_exists(str(db_path))

        # Verify the file still exists and size hasn't changed
        assert db_path.exists()
        assert db_path.stat().st_size == initial_size


class TestMetaSqliteAdapter:
    @pytest.fixture
    def temp_db_config(self, tmp_path):
        """Create a temporary database configuration."""
        db_path = tmp_path / "test_meta.db"

        # Mock the config to return our temporary database path
        mock_config = type(
            "MockConfig",
            (),
            {
                "meta_store": type(
                    "MockMetaStore",
                    (),
                    {"meta_db_type": "sqlite", "meta_db_path": str(db_path)},
                )()
            },
        )()

        return mock_config, db_path

    def test_metastore_setup_creates_database(self, temp_db_config):
        """Test that metastore_setup creates the actual database file."""
        mock_config, db_path = temp_db_config

        # Verify the database doesn't exist initially
        assert not db_path.exists()

        with patch(
            "src.synapso_core.persistence.implementations.meta_store.meta_sqlite.get_config",
            return_value=mock_config,
        ):
            adapter = MetaSqliteAdapter()
            result = adapter.metastore_setup()

            # Verify setup was successful
            assert result is True

            # Verify the database file was created
            assert db_path.exists()

    def test_metastore_setup_creates_tables(self, temp_db_config):
        """Test that metastore_setup creates the necessary tables."""
        mock_config, db_path = temp_db_config

        with patch(
            "src.synapso_core.persistence.implementations.meta_store.meta_sqlite.get_config",
            return_value=mock_config,
        ):
            with MetaSqliteAdapter() as adapter:
                adapter.metastore_setup()

                # Verify tables were created by checking if we can query them
                engine = adapter.get_sync_engine()
                with engine.connect() as conn:
                    # Check if tables exist by querying them
                    result = conn.execute(
                        text("SELECT name FROM sqlite_master WHERE type='table'")
                    )
                    table_names = [row[0] for row in result]
                    assert "cortex" in table_names

    def test_metastore_setup_wrong_db_type(self):
        """Test that metastore_setup raises error for wrong database type."""
        mock_config = type(
            "MockConfig",
            (),
            {
                "meta_store": type(
                    "MockMetaStore",
                    (),
                    {
                        "meta_db_type": "postgresql",  # Wrong type
                        "meta_db_path": "/tmp/test.db",
                    },
                )()
            },
        )()

        with patch(
            "src.synapso_core.persistence.implementations.meta_store.meta_sqlite.get_config",
            return_value=mock_config,
        ):
            with pytest.raises(ValueError, match="Meta store type is not sqlite"):
                adapter = MetaSqliteAdapter()

    def test_get_sync_engine(self, temp_db_config):
        """Test getting a sync engine."""
        mock_config, db_path = temp_db_config

        with patch(
            "src.synapso_core.persistence.implementations.meta_store.meta_sqlite.get_config",
            return_value=mock_config,
        ):
            adapter = MetaSqliteAdapter()
            adapter.metastore_setup()

            engine = adapter.get_sync_engine()
            assert engine is not None

            # Test that the engine can connect
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                assert result.scalar() == 1

    def test_get_async_engine(self, temp_db_config):
        """Test getting an async engine."""
        mock_config, db_path = temp_db_config

        with patch(
            "src.synapso_core.persistence.implementations.meta_store.meta_sqlite.get_config",
            return_value=mock_config,
        ):
            adapter = MetaSqliteAdapter()
            adapter.metastore_setup()

            async_engine = adapter.get_async_engine()
            assert async_engine is not None

    def test_database_persistence(self, temp_db_config):
        """Test that data persists between adapter instances."""
        mock_config, db_path = temp_db_config

        # Create first adapter and insert data
        with patch(
            "src.synapso_core.persistence.implementations.meta_store.meta_sqlite.get_config",
            return_value=mock_config,
        ):
            with MetaSqliteAdapter() as adapter1:
                adapter1.metastore_setup()

                # Insert some test data
                engine = adapter1.get_sync_engine()
                with engine.connect() as conn:
                    from datetime import datetime

                    now = datetime.now()
                    conn.execute(
                        text(
                            "INSERT INTO cortex (cortex_id, cortex_name, path, created_at, updated_at) VALUES (:cortex_id, :cortex_name, :path, :created_at, :updated_at)"
                        ),
                        {
                            "cortex_id": "test_id",
                            "cortex_name": "test_cortex",
                            "path": "/test/path",
                            "created_at": now,
                            "updated_at": now,
                        },
                    )
                    conn.commit()

            # Create second adapter and verify data persists
            with MetaSqliteAdapter() as adapter2:
                adapter2.metastore_setup()

                # Verify data still exists
                engine = adapter2.get_sync_engine()
                with engine.connect() as conn:
                    result = conn.execute(
                        text("SELECT * FROM cortex WHERE cortex_id = :cortex_id"),
                        {"cortex_id": "test_id"},
                    )
                    row = result.fetchone()
                    assert row is not None
                    assert row[1] == "test_cortex"  # cortex_name column
                    assert row[2] == "/test/path"  # path column

    def test_multiple_setup_calls(self, temp_db_config):
        """Test that multiple setup calls don't cause issues."""
        mock_config, db_path = temp_db_config

        with patch(
            "src.synapso_core.persistence.implementations.meta_store.meta_sqlite.get_config",
            return_value=mock_config,
        ):
            adapter = MetaSqliteAdapter()

            # Call setup multiple times
            assert adapter.metastore_setup() is True
            assert adapter.metastore_setup() is True
            assert adapter.metastore_setup() is True

            # Verify database still exists and is functional
            assert db_path.exists()
            engine = adapter.get_sync_engine()
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                assert result.scalar() == 1


class TestMetaSqliteAdapterIntegration:
    """Integration tests that test the full workflow."""

    @pytest.fixture
    def temp_db_path(self, tmp_path):
        """Create a temporary database path."""
        return tmp_path / "integration_test.db"

    def test_full_workflow(self, temp_db_path):
        """Test the complete workflow from setup to teardown."""
        # Mock config
        mock_config = type(
            "MockConfig",
            (),
            {
                "meta_store": type(
                    "MockMetaStore",
                    (),
                    {"meta_db_type": "sqlite", "meta_db_path": str(temp_db_path)},
                )()
            },
        )()

        with patch(
            "src.synapso_core.persistence.implementations.meta_store.meta_sqlite.get_config",
            return_value=mock_config,
        ):
            # Setup
            adapter = MetaSqliteAdapter()
            assert adapter.metastore_setup() is True

            # Verify database exists
            assert temp_db_path.exists()

            # Test sync operations
            sync_engine = adapter.get_sync_engine()
            with sync_engine.connect() as conn:
                # Create a test table
                conn.execute(
                    text(
                        "CREATE TABLE test_integration (id INTEGER PRIMARY KEY, data TEXT)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO test_integration (id, data) VALUES (1, 'integration_test')"
                    )
                )
                conn.commit()

                # Verify data
                result = conn.execute(
                    text("SELECT data FROM test_integration WHERE id = 1")
                )
                assert result.scalar() == "integration_test"

            # Test async engine creation (without using it)
            async_engine = adapter.get_async_engine()
            assert async_engine is not None
