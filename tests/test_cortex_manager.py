import dataclasses
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.synapso.cortex_manager import (
    SUPPORTED_FORMATS,
    Cortex,
    FileRecord,
    FileState,
    _classify,
    _file_walk,
    _get_file_list_path,
    _get_ingestion_errors_path,
    _validate_cortex_path,
    create_cortex,
    delete_cortex,
    get_async_engine,
    get_cortex_by_id,
    index_cortex,
    initialize_cortex,
    purge_cortex,
)


class TestFileState:
    def test_file_state_enum_values(self):
        """Test that FileState enum has correct values."""
        assert FileState.ELIGIBLE.value == 0
        assert FileState.HIDDEN_FILE.value == 1
        assert FileState.HIDDEN_DIRECTORY.value == 2
        assert FileState.UNSUPPORTED_FORMAT.value == 3


class TestFileRecord:
    def test_file_record_creation(self):
        """Test FileRecord dataclass creation."""
        path = Path("/test/file.md")
        record = FileRecord(path=path, state=FileState.ELIGIBLE)
        assert record.path == path
        assert record.state == FileState.ELIGIBLE

    def test_file_record_immutability(self):
        """Test that FileRecord is immutable."""
        path = Path("/test/file.md")
        record = FileRecord(path=path, state=FileState.ELIGIBLE)

        with pytest.raises(dataclasses.FrozenInstanceError):
            record.path = Path("/other/file.md")


class TestClassify:
    def test_classify_eligible_file(self):
        """Test classification of eligible markdown files."""
        root = Path("/test")
        file_path = root / "document.md"
        assert _classify(file_path, root) == FileState.ELIGIBLE

    def test_classify_hidden_file(self):
        """Test classification of hidden files."""
        root = Path("/test")
        file_path = root / ".hidden.md"
        assert _classify(file_path, root) == FileState.HIDDEN_FILE

    def test_classify_hidden_directory(self):
        """Test classification of files in hidden directories."""
        root = Path("/test")
        file_path = root / ".hidden" / "document.md"
        assert _classify(file_path, root) == FileState.HIDDEN_DIRECTORY

    def test_classify_nested_hidden_directory(self):
        """Test classification of files in nested hidden directories."""
        root = Path("/test")
        file_path = root / "visible" / ".hidden" / "document.md"
        assert _classify(file_path, root) == FileState.HIDDEN_DIRECTORY

    def test_classify_unsupported_format(self):
        """Test classification of unsupported file formats."""
        root = Path("/test")
        file_path = root / "document.txt"
        assert _classify(file_path, root) == FileState.UNSUPPORTED_FORMAT

    def test_classify_case_insensitive_extension(self):
        """Test that file extension matching is case insensitive."""
        root = Path("/test")
        file_path = root / "document.MD"
        assert _classify(file_path, root) == FileState.ELIGIBLE

    def test_classify_all_supported_formats(self):
        """Test that all supported formats are classified as eligible."""
        root = Path("/test")
        for ext in SUPPORTED_FORMATS:
            file_path = root / f"document{ext}"
            assert _classify(file_path, root) == FileState.ELIGIBLE


class TestPathHelpers:
    def test_get_file_list_path(self):
        """Test _get_file_list_path function."""
        directory_path = "/test/dir"
        expected_path = Path(directory_path) / ".synapso" / "file_list.csv"

        result = _get_file_list_path(directory_path, ensure_present=False)
        assert result == expected_path

    def test_get_ingestion_errors_path(self):
        """Test _get_ingestion_errors_path function."""
        directory_path = "/test/dir"
        expected_path = Path(directory_path) / ".synapso" / "ingestion_errors.jsonl"

        result = _get_ingestion_errors_path(directory_path, ensure_present=False)
        assert result == expected_path


class TestFileWalk:
    def test_file_walk_empty_directory(self, tmp_path):
        """Test _file_walk with empty directory."""
        files = list(_file_walk(str(tmp_path)))
        assert len(files) == 0

    def test_file_walk_with_mixed_files(self, tmp_path):
        """Test _file_walk with various file types."""
        # Create test files
        (tmp_path / "visible.md").touch()
        (tmp_path / ".hidden.md").touch()
        (tmp_path / "document.txt").touch()
        (tmp_path / ".hidden_dir").mkdir()
        (tmp_path / ".hidden_dir" / "file.md").touch()

        files = list(_file_walk(str(tmp_path)))
        assert len(files) == 4

        # Check classifications
        file_states = {f.path.name: f.state for f in files}
        assert file_states["visible.md"] == FileState.ELIGIBLE
        assert file_states[".hidden.md"] == FileState.HIDDEN_FILE
        assert file_states["document.txt"] == FileState.UNSUPPORTED_FORMAT
        assert file_states["file.md"] == FileState.HIDDEN_DIRECTORY


class TestValidateCortexPath:
    def test_validate_cortex_path_valid_directory(self, tmp_path):
        """Test validation of valid directory path."""
        _validate_cortex_path(str(tmp_path))  # Should not raise

    def test_validate_cortex_path_nonexistent(self):
        """Test validation of nonexistent path."""
        with pytest.raises(ValueError, match="is not a directory"):
            _validate_cortex_path("/nonexistent/path")

    def test_validate_cortex_path_file(self, tmp_path):
        """Test validation of file path."""
        test_file = tmp_path / "test.txt"
        test_file.touch()

        with pytest.raises(ValueError, match="is not a directory"):
            _validate_cortex_path(str(test_file))

    def test_validate_cortex_path_hidden_directory(self, tmp_path):
        """Test validation of hidden directory."""
        hidden_dir = tmp_path / ".hidden"
        hidden_dir.mkdir()

        with pytest.raises(ValueError, match="is hidden"):
            _validate_cortex_path(str(hidden_dir))


class TestGetAsyncEngine:
    @patch("src.synapso.cortex_manager.get_config")
    @patch("src.synapso.cortex_manager.MetaStoreFactory")
    @patch("os.open")
    @patch("os.close")
    def test_get_async_engine(
        self, mock_os_close, mock_os_open, mock_factory, mock_get_config
    ):
        """Test get_async_engine function."""
        mock_config = MagicMock()
        mock_config.meta_store.meta_db_type = "sqlite"
        mock_get_config.return_value = mock_config

        mock_meta_store = MagicMock()
        mock_engine = MagicMock()
        mock_meta_store.get_async_engine.return_value = mock_engine
        mock_factory.get_meta_store.return_value = mock_meta_store

        result = get_async_engine()

        mock_factory.get_meta_store.assert_called_once_with("sqlite")
        mock_meta_store.get_async_engine.assert_called_once()
        assert result == mock_engine


class TestCreateCortex:
    @patch("src.synapso.cortex_manager.get_async_engine")
    @patch("src.synapso.cortex_manager.uuid")
    @patch("src.synapso.cortex_manager._validate_cortex_path")
    @patch("os.open")
    @patch("os.close")
    async def test_create_cortex_success(
        self,
        mock_os_close,
        mock_os_open,
        mock_validate_path,
        mock_uuid,
        mock_get_engine,
    ):
        """Test successful cortex creation."""
        # Setup mocks
        mock_uuid.uuid4.return_value.hex = "test_cortex_id"
        mock_session = AsyncMock(spec=AsyncSession)
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine

        with patch("src.synapso.cortex_manager.AsyncSession") as mock_async_session:
            mock_async_session.return_value.__aenter__.return_value = mock_session

            # Test
            result = await create_cortex("Test Cortex", "/test/path")

            # Assertions
            assert result.cortex_id == "test_cortex_id"
            assert result.cortex_name == "Test Cortex"
            assert result.path == "/test/path"
            assert result.created_at is not None
            assert result.updated_at is not None
            assert result.last_indexed_at is None

            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()

    @patch("src.synapso.cortex_manager.get_async_engine")
    @patch("src.synapso.cortex_manager._validate_cortex_path")
    @patch("os.open")
    @patch("os.close")
    async def test_create_cortex_with_none_name(
        self, mock_os_close, mock_os_open, mock_validate_path, mock_get_engine
    ):
        """Test cortex creation with None name."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine

        with patch("src.synapso.cortex_manager.AsyncSession") as mock_async_session:
            mock_async_session.return_value.__aenter__.return_value = mock_session

            result = await create_cortex(None, "/test/path")
            assert result.cortex_name is None

    async def test_create_cortex_invalid_path(self):
        """Test cortex creation with invalid path."""
        with pytest.raises(ValueError):
            await create_cortex("Test", "/nonexistent/path")


class TestGetCortexById:
    @patch("src.synapso.cortex_manager.get_async_engine")
    @patch("os.open")
    @patch("os.close")
    async def test_get_cortex_by_id_success(
        self, mock_os_close, mock_os_open, mock_get_engine
    ):
        """Test successful cortex retrieval by ID."""
        mock_cortex = MagicMock(spec=Cortex)
        mock_row = MagicMock()
        mock_row.__getitem__.return_value = mock_cortex

        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.first.return_value = mock_row
        mock_session.execute.return_value = mock_result

        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine

        with patch("src.synapso.cortex_manager.AsyncSession") as mock_async_session:
            mock_async_session.return_value.__aenter__.return_value = mock_session

            result = await get_cortex_by_id("test_id")
            assert result == mock_cortex

    @patch("src.synapso.cortex_manager.get_async_engine")
    @patch("os.open")
    @patch("os.close")
    async def test_get_cortex_by_id_not_found(
        self, mock_os_close, mock_os_open, mock_get_engine
    ):
        """Test cortex retrieval when cortex doesn't exist."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.first.return_value = None
        mock_session.execute.return_value = mock_result

        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine

        with patch("src.synapso.cortex_manager.AsyncSession") as mock_async_session:
            mock_async_session.return_value.__aenter__.return_value = mock_session

            with pytest.raises(ValueError, match="not found"):
                await get_cortex_by_id("nonexistent_id")


class TestInitializeCortex:
    @pytest.mark.asyncio
    @patch("src.synapso.cortex_manager.get_cortex_by_id")
    @patch("src.synapso.cortex_manager.index_cortex")
    @patch("os.mkdir")
    @patch("os.open")
    @patch("os.close")
    async def test_initialize_cortex_success(
        self,
        mock_os_close,
        mock_os_open,
        mock_makedirs,
        mock_index_cortex,
        mock_get_cortex,
    ):
        """Test successful cortex initialization."""
        mock_cortex = MagicMock()
        mock_cortex.path = "/test/path"
        mock_get_cortex.return_value = mock_cortex
        mock_index_cortex.return_value = True

        result = await initialize_cortex("test_id", index_now=True)

        assert result is True
        mock_makedirs.assert_called_once_with(Path("/test/path/.synapso"), 511)
        mock_index_cortex.assert_called_once_with(cortex_id="test_id")

    @pytest.mark.asyncio
    @patch("src.synapso.cortex_manager.get_cortex_by_id")
    @patch("os.mkdir")
    @patch("os.open")
    @patch("os.close")
    async def test_initialize_cortex_no_index(
        self, mock_os_close, mock_os_open, mock_makedirs, mock_get_cortex
    ):
        """Test cortex initialization without indexing."""
        mock_cortex = MagicMock()
        mock_cortex.path = "/test/path"
        mock_get_cortex.return_value = mock_cortex

        result = await initialize_cortex("test_id", index_now=False)

        assert result is True
        mock_makedirs.assert_called_once_with(Path("/test/path/.synapso"), 511)


class TestIndexCortex:
    @pytest.mark.asyncio
    @patch("src.synapso.cortex_manager.get_cortex_by_id")
    @patch("src.synapso.cortex_manager.ingest_file")
    @patch("os.open")
    @patch("os.close")
    @patch("io.open")
    async def test_index_cortex_success(
        self,
        mock_io_opne,
        mock_os_close,
        mock_os_open,
        mock_ingest_file,
        mock_get_cortex,
    ):
        """Test successful cortex indexing."""
        mock_cortex = MagicMock()
        mock_cortex.path = "/test/path"
        mock_get_cortex.return_value = mock_cortex
        mock_ingest_file.return_value = (True, None)

        with (
            patch("src.synapso.cortex_manager._file_walk") as mock_file_walk,
            patch("builtins.open", create=True) as mock_open,
            patch("csv.writer") as mock_csv_writer,
        ):
            # Setup mock file records
            mock_file_walk.return_value = [
                FileRecord(Path("/test/path/file1.md"), FileState.ELIGIBLE),
                FileRecord(Path("/test/path/file2.txt"), FileState.UNSUPPORTED_FORMAT),
            ]

            mock_writer = MagicMock()
            mock_csv_writer.return_value = mock_writer

            result = await index_cortex("test_id")

            assert result is True
            mock_writer.writerow.assert_called()
            mock_ingest_file.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.synapso.cortex_manager.get_cortex_by_id")
    @patch("os.open")
    @patch("os.close")
    async def test_index_cortex_invalid_id(
        self, mock_os_close, mock_os_open, mock_get_cortex
    ):
        """Test indexing with invalid cortex ID."""
        mock_get_cortex.return_value = None

        with pytest.raises(ValueError, match="Invalid cortex id"):
            await index_cortex("invalid_id")

    @pytest.mark.asyncio
    @patch("src.synapso.cortex_manager.get_cortex_by_id")
    @patch("src.synapso.cortex_manager.ingest_file")
    @patch("os.open")
    @patch("os.close")
    @patch("io.open")
    async def test_index_cortex_ingestion_error(
        self,
        mock_io_open,
        mock_os_close,
        mock_os_open,
        mock_ingest_file,
        mock_get_cortex,
    ):
        """Test cortex indexing with ingestion errors."""
        mock_cortex = MagicMock()
        mock_cortex.path = "/test/path"
        mock_get_cortex.return_value = mock_cortex
        mock_ingest_file.return_value = (False, {"error": "test error"})

        with (
            patch("src.synapso.cortex_manager._file_walk") as mock_file_walk,
            patch("builtins.open", create=True) as mock_open,
            patch("csv.writer") as mock_csv_writer,
            patch("json.dumps") as mock_json_dumps,
        ):
            mock_file_walk.return_value = [
                FileRecord(Path("/test/path/file1.md"), FileState.ELIGIBLE),
            ]

            mock_writer = MagicMock()
            mock_csv_writer.return_value = mock_writer
            mock_json_dumps.return_value = '{"error": "test error"}'

            result = await index_cortex("test_id")

            assert result is True  # Function returns True even with errors
            mock_ingest_file.assert_called_once()


class TestDeleteCortex:
    @pytest.mark.asyncio
    async def test_delete_cortex_not_implemented(self):
        """Test that delete_cortex is not yet implemented."""
        with pytest.raises(NotImplementedError):
            await delete_cortex("test_id")


class TestPurgeCortex:
    @pytest.mark.asyncio
    async def test_purge_cortex_not_implemented(self):
        """Test that purge_cortex is not yet implemented."""
        with pytest.raises(NotImplementedError):
            await purge_cortex("test_id")


# Integration tests
class TestCortexManagerIntegration:
    @pytest.fixture
    def temp_cortex_dir(self):
        """Create a temporary directory for cortex testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_file_walk_integration(self, temp_cortex_dir):
        """Integration test for file walking functionality."""
        temp_path = Path(temp_cortex_dir)

        # Create test structure
        (temp_path / "visible.md").touch()
        (temp_path / ".hidden.md").touch()
        (temp_path / "document.txt").touch()
        (temp_path / ".hidden_dir").mkdir()
        (temp_path / ".hidden_dir" / "file.md").touch()
        (temp_path / "nested" / "visible").mkdir(parents=True)
        (temp_path / "nested" / "visible" / "file.md").touch()

        files = list(_file_walk(temp_cortex_dir))

        # Should find all files except those in hidden directories
        assert len(files) == 5

        # Use resolve() to handle symlinks and normalize paths
        temp_path_resolved = temp_path.resolve()
        file_states = {
            str(f.path.resolve().relative_to(temp_path_resolved)): f.state
            for f in files
        }
        assert file_states["visible.md"] == FileState.ELIGIBLE
        assert file_states[".hidden.md"] == FileState.HIDDEN_FILE
        assert file_states["document.txt"] == FileState.UNSUPPORTED_FORMAT
        assert file_states["nested/visible/file.md"] == FileState.ELIGIBLE
