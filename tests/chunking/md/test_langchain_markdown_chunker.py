import os

import pytest

from src.synapso.chunking.implementations.md.chonkie_recursive_chunker import (
    ChonkieRecursiveChunker,
)
from src.synapso.chunking.implementations.md.langchain_markdown_chunker import (
    LangchainMarkdownChunker,
)
from src.synapso.chunking.interface import Chunk

RESOURCES_DIR = os.path.join(os.path.dirname(__file__), "../../..", "resources")
SAMPLE_MD = os.path.join(RESOURCES_DIR, "sample.md")


@pytest.fixture
def chunker():
    return LangchainMarkdownChunker()


@pytest.fixture
def chonkie_chunker():
    return ChonkieRecursiveChunker()


def test_chunk_file_sample_md(chunker):
    chunks = chunker.chunk_file(SAMPLE_MD)
    # The expected number of chunks depends on the headers in sample.md
    # For this sample, we expect at least one chunk per header section
    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, Chunk)
        assert isinstance(chunk.text, str)
        assert isinstance(chunk.metadata, dict)
        # Metadata should contain at least 'header_1' for each chunk
        assert any(h in chunk.metadata for h in ["header_1", "header_2", "header_3"])
    # Check that the first chunk contains the first section's text
    assert "Local-First vs. Cloud-Native" in chunks[0].metadata.get("header_1", "")


def test_chunk_metadata_correctness(chunker):
    chunks = chunker.chunk_file(SAMPLE_MD)
    for chunk in chunks:
        # All metadata values should be strings
        for v in chunk.metadata.values():
            assert isinstance(v, str)


def test_is_file_supported(chunker):
    assert chunker.is_file_supported("foo.md")
    assert not chunker.is_file_supported("foo.txt")
    assert not chunker.is_file_supported("foo")


def test_empty_file(tmp_path, chunker):
    empty_file = tmp_path / "empty.md"
    empty_file.write_text("")
    chunks = chunker.chunk_file(str(empty_file))
    assert chunks == []


def test_file_with_only_headers(tmp_path, chunker):
    header_file = tmp_path / "headers.md"
    header_file.write_text("# Header1\n## Header2\n### Header3")
    chunks = chunker.chunk_file(str(header_file))
    # Should produce empty or header-only chunks
    for chunk in chunks:
        assert chunk.text.strip() == ""
        assert chunk.metadata


def test_file_with_no_headers(tmp_path, chunker):
    no_header_file = tmp_path / "noheaders.md"
    no_header_file.write_text("Just some text with no markdown headers.")
    chunks = chunker.chunk_file(str(no_header_file))
    # Should produce a single chunk with no header metadata
    assert len(chunks) == 1
    assert chunks[0].text.strip() == "Just some text with no markdown headers."
    assert chunks[0].metadata == {}


def test_nonexistent_file(chunker):
    with pytest.raises(FileNotFoundError):
        chunker.chunk_file("not_a_real_file.md")


def test_permission_error(tmp_path, chunker):
    file_path = tmp_path / "no_permission.md"
    file_path.write_text("test")
    file_path.chmod(0o000)  # Remove all permissions
    try:
        try:
            chunker.chunk_file(str(file_path))
        except PermissionError as e:
            assert "Permission denied" in str(e)
        else:
            pytest.fail("PermissionError not raised")
    finally:
        file_path.chmod(0o644)  # Restore permissions so file can be deleted


def test_is_a_directory_error(tmp_path, chunker):
    dir_path = tmp_path / "adir"
    dir_path.mkdir()
    try:
        try:
            chunker.chunk_file(str(dir_path))
        except IsADirectoryError as e:
            assert "Expected a file but found a directory" in str(e)
        else:
            pytest.fail("IsADirectoryError not raised")
    finally:
        dir_path.rmdir()


def test_unicode_decode_error(tmp_path, chunker):
    file_path = tmp_path / "bad_utf8.md"
    # Write invalid UTF-8 bytes
    with open(file_path, "wb") as f:
        f.write(b"\xff\xfe\xfd")
    try:
        try:
            chunker.chunk_file(str(file_path))
        except UnicodeDecodeError as e:
            assert "File is not valid UTF-8" in str(e)
        else:
            pytest.fail("UnicodeDecodeError not raised")
    finally:
        file_path.unlink()


def test_os_error(monkeypatch, chunker):
    def bad_open(*args, **kwargs):
        raise OSError("Simulated OS error")

    monkeypatch.setattr("builtins.open", bad_open)
    try:
        try:
            chunker.chunk_file(SAMPLE_MD)
        except OSError as e:
            assert "OS error reading file" in str(e)
        else:
            pytest.fail("OSError not raised")
    finally:
        monkeypatch.undo()


def test_generic_exception(monkeypatch, chunker):
    def bad_open(*args, **kwargs):
        raise RuntimeError("Simulated generic error")

    monkeypatch.setattr("builtins.open", bad_open)
    try:
        try:
            chunker.chunk_file(SAMPLE_MD)
        except Exception as e:
            assert "Error reading file" in str(e)
        else:
            pytest.fail("Generic Exception not raised")
    finally:
        monkeypatch.undo()


def test_chonkie_chunk_file_sample_md(chonkie_chunker):
    chunks = chonkie_chunker.chunk_file(SAMPLE_MD)
    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, Chunk)
        assert isinstance(chunk.text, str)
        assert isinstance(chunk.metadata, dict)
        # Metadata should contain start_index, end_index, token_count, level
        for key in ["start_index", "end_index", "token_count", "level"]:
            assert key in chunk.metadata
    # Check that the first chunk contains the first section's text
    assert "Local-First" in chunks[0].text or "Cloud-Native" in chunks[0].text


def test_chonkie_chunk_metadata_correctness(chonkie_chunker):
    chunks = chonkie_chunker.chunk_file(SAMPLE_MD)
    for chunk in chunks:
        # All metadata values should be str or int (converted to str by dataclass default)
        for v in chunk.metadata.values():
            assert isinstance(v, (str, int))


def test_chonkie_is_file_supported(chonkie_chunker):
    assert chonkie_chunker.is_file_supported("foo.md")
    assert not chonkie_chunker.is_file_supported("foo.txt")
    assert not chonkie_chunker.is_file_supported("foo")


def test_chonkie_empty_file(tmp_path, chonkie_chunker):
    empty_file = tmp_path / "empty.md"
    empty_file.write_text("")
    chunks = chonkie_chunker.chunk_file(str(empty_file))
    assert chunks == []


def test_chonkie_file_with_only_headers(tmp_path, chonkie_chunker):
    header_file = tmp_path / "headers.md"
    header_file.write_text("# Header1\n## Header2\n### Header3")
    chunks = chonkie_chunker.chunk_file(str(header_file))
    # Should produce empty or header-only chunks
    for chunk in chunks:
        assert isinstance(chunk.text, str)
        assert chunk.metadata


def test_chonkie_file_with_no_headers(tmp_path, chonkie_chunker):
    no_header_file = tmp_path / "noheaders.md"
    no_header_file.write_text("Just some text with no markdown headers.")
    chunks = chonkie_chunker.chunk_file(str(no_header_file))
    # Should produce at least one chunk
    assert len(chunks) >= 1
    assert "Just some text with no markdown headers." in chunks[0].text


def test_chonkie_nonexistent_file(chonkie_chunker):
    with pytest.raises(FileNotFoundError):
        chonkie_chunker.chunk_file("not_a_real_file.md")
