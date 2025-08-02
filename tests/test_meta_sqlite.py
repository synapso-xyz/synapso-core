from unittest.mock import patch

import pytest
from sqlalchemy import text

from synapso_core.data_store.backend.sqlite.sqlite_meta_store import SqliteMetaStore
from synapso_core.data_store.backend.sqlite.utils import create_sqlite_db_if_not_exists


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


class TestSqliteMetaStore:
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

    def test_setup_creates_database(self, temp_db_config):
        """Test that setup creates the actual database file."""
        mock_config, db_path = temp_db_config

        # Verify the database doesn't exist initially
        assert not db_path.exists()

        with patch(
            "synapso_core.data_store.backend.sqlite.sqlite_meta_store.get_config",
            return_value=mock_config,
        ):
            adapter = SqliteMetaStore()
            result = adapter.setup()

            # Verify setup was successful
            assert result is True

            # Verify the database file was created
            assert db_path.exists()

    def test_setup_creates_tables(self, temp_db_config):
        """Test that setup creates the necessary tables."""
        mock_config, db_path = temp_db_config

        with patch(
            "synapso_core.data_store.backend.sqlite.sqlite_meta_store.get_config",
            return_value=mock_config,
        ):
            adapter = SqliteMetaStore()
            adapter.setup()

            # Verify tables were created by checking if we can query them
            engine = adapter.get_sync_engine()
            with engine.connect() as conn:
                # Check if tables exist by querying them
                result = conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table'")
                )
                table_names = [row[0] for row in result]
                assert "cortex" in table_names

    def test_setup_wrong_db_type(self):
        """Test that setup raises error for wrong database type."""
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
            "synapso_core.data_store.backend.sqlite.sqlite_meta_store.get_config",
            return_value=mock_config,
        ):
            with pytest.raises(ValueError, match="Meta store type is not sqlite"):
                adapter = SqliteMetaStore()

    def test_get_sync_engine(self, temp_db_config):
        """Test getting a sync engine."""
        mock_config, db_path = temp_db_config

        with patch(
            "synapso_core.data_store.backend.sqlite.sqlite_meta_store.get_config",
            return_value=mock_config,
        ):
            adapter = SqliteMetaStore()
            adapter.setup()

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
            "synapso_core.data_store.backend.sqlite.sqlite_meta_store.get_config",
            return_value=mock_config,
        ):
            adapter = SqliteMetaStore()
            adapter.setup()

            async_engine = adapter.get_async_engine()
            assert async_engine is not None

    def test_create_cortex(self, temp_db_config):
        """Test creating a cortex."""
        mock_config, db_path = temp_db_config

        with patch(
            "synapso_core.data_store.backend.sqlite.sqlite_meta_store.get_config",
            return_value=mock_config,
        ):
            adapter = SqliteMetaStore()
            adapter.setup()

            # Create a cortex
            cortex_id = adapter.create_cortex("test_cortex", "/test/path")
            assert cortex_id is not None

            # Verify the cortex was created in the database
            engine = adapter.get_sync_engine()
            with engine.connect() as conn:
                result = conn.execute(
                    text("SELECT * FROM cortex WHERE cortex_id = :cortex_id"),
                    {"cortex_id": cortex_id},
                )
                row = result.fetchone()
                assert row is not None
                assert row[1] == "test_cortex"  # cortex_name column
                assert row[2] == "/test/path"  # path column

    def test_get_cortex_by_id(self, temp_db_config):
        """Test getting a cortex by ID."""
        mock_config, db_path = temp_db_config

        with patch(
            "synapso_core.data_store.backend.sqlite.sqlite_meta_store.get_config",
            return_value=mock_config,
        ):
            adapter = SqliteMetaStore()
            adapter.setup()

            # Create a cortex first
            cortex_id = adapter.create_cortex("test_cortex", "/test/path")

            # Get the cortex by ID
            cortex = adapter.get_cortex_by_id(cortex_id)
            assert cortex is not None
            assert cortex.cortex_name == "test_cortex"
            assert cortex.path == "/test/path"

    def test_get_cortex_by_id_not_found(self, temp_db_config):
        """Test getting a cortex by ID when it doesn't exist."""
        mock_config, db_path = temp_db_config

        with patch(
            "synapso_core.data_store.backend.sqlite.sqlite_meta_store.get_config",
            return_value=mock_config,
        ):
            adapter = SqliteMetaStore()
            adapter.setup()

            # Try to get a non-existent cortex
            cortex = adapter.get_cortex_by_id("non_existent_id")
            assert cortex is None

    def test_get_cortex_by_name(self, temp_db_config):
        """Test getting a cortex by name."""
        mock_config, db_path = temp_db_config

        with patch(
            "synapso_core.data_store.backend.sqlite.sqlite_meta_store.get_config",
            return_value=mock_config,
        ):
            adapter = SqliteMetaStore()
            adapter.setup()

            # Create a cortex first
            cortex_id = adapter.create_cortex("test_cortex", "/test/path")

            # Get the cortex by name
            found_cortex_id = adapter.get_cortex_by_name("test_cortex")
            assert found_cortex_id == cortex_id

    def test_get_cortex_by_name_not_found(self, temp_db_config):
        """Test getting a cortex by name when it doesn't exist."""
        mock_config, db_path = temp_db_config

        with patch(
            "synapso_core.data_store.backend.sqlite.sqlite_meta_store.get_config",
            return_value=mock_config,
        ):
            adapter = SqliteMetaStore()
            adapter.setup()

            # Try to get a non-existent cortex
            cortex_id = adapter.get_cortex_by_name("non_existent_cortex")
            assert cortex_id is None

    def test_list_cortices(self, temp_db_config):
        """Test listing all cortices."""
        mock_config, db_path = temp_db_config

        with patch(
            "synapso_core.data_store.backend.sqlite.sqlite_meta_store.get_config",
            return_value=mock_config,
        ):
            adapter = SqliteMetaStore()
            adapter.setup()

            # Create multiple cortices
            cortex_id1 = adapter.create_cortex("cortex1", "/path1")
            cortex_id2 = adapter.create_cortex("cortex2", "/path2")

            # List all cortices
            cortices = adapter.list_cortices()
            assert len(cortices) == 2
            cortex_names = [c.cortex_name for c in cortices]
            assert "cortex1" in cortex_names
            assert "cortex2" in cortex_names

    def test_update_cortex(self, temp_db_config):
        """Test updating a cortex."""
        mock_config, db_path = temp_db_config

        with patch(
            "synapso_core.data_store.backend.sqlite.sqlite_meta_store.get_config",
            return_value=mock_config,
        ):
            adapter = SqliteMetaStore()
            adapter.setup()

            # Create a cortex first
            cortex_id = adapter.create_cortex("test_cortex", "/test/path")

            # Get the cortex and update it
            cortex = adapter.get_cortex_by_id(cortex_id)
            cortex.cortex_name = "updated_cortex"
            cortex.path = "/updated/path"

            # Update the cortex
            result = adapter.update_cortex(cortex)
            assert result is True

            # Verify the update
            updated_cortex = adapter.get_cortex_by_id(cortex_id)
            assert updated_cortex.cortex_name == "updated_cortex"
            assert updated_cortex.path == "/updated/path"

    def test_database_persistence(self, temp_db_config):
        """Test that data persists between adapter instances."""
        mock_config, db_path = temp_db_config

        # Create first adapter and insert data
        with patch(
            "synapso_core.data_store.backend.sqlite.sqlite_meta_store.get_config",
            return_value=mock_config,
        ):
            adapter1 = SqliteMetaStore()
            adapter1.setup()

            # Create a cortex
            cortex_id = adapter1.create_cortex("test_cortex", "/test/path")

            # Create second adapter and verify data persists
            adapter2 = SqliteMetaStore()
            adapter2.setup()

            # Verify data still exists
            cortex = adapter2.get_cortex_by_id(cortex_id)
            assert cortex is not None
            assert cortex.cortex_name == "test_cortex"
            assert cortex.path == "/test/path"

    def test_multiple_setup_calls(self, temp_db_config):
        """Test that multiple setup calls don't cause issues."""
        mock_config, db_path = temp_db_config

        with patch(
            "synapso_core.data_store.backend.sqlite.sqlite_meta_store.get_config",
            return_value=mock_config,
        ):
            adapter = SqliteMetaStore()

            # Call setup multiple times
            assert adapter.setup() is True
            assert adapter.setup() is True
            assert adapter.setup() is True

            # Verify database still exists and is functional
            assert db_path.exists()
            engine = adapter.get_sync_engine()
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                assert result.scalar() == 1


class TestSqliteMetaStoreIntegration:
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
            "synapso_core.data_store.backend.sqlite.sqlite_meta_store.get_config",
            return_value=mock_config,
        ):
            # Setup
            adapter = SqliteMetaStore()
            assert adapter.setup() is True

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

            # Test cortex operations
            cortex_id = adapter.create_cortex("integration_cortex", "/integration/path")
            assert cortex_id is not None

            cortex = adapter.get_cortex_by_id(cortex_id)
            assert cortex.cortex_name == "integration_cortex"
            assert cortex.path == "/integration/path"

            # Test listing cortices
            cortices = adapter.list_cortices()
            assert len(cortices) == 1
            assert cortices[0].cortex_name == "integration_cortex"
