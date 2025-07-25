import os
import pytest
from src.synapso.chunking.implementations.md.chonkie_recursive_chunker import ChonkieRecursiveChunker
from src.synapso.chunking.interface import Chunk

RESOURCES_DIR = os.path.join(os.path.dirname(__file__), '../../..', 'resources')
SAMPLE_MD = os.path.join(RESOURCES_DIR, 'sample.md')

@pytest.fixture
def chunker():
    return ChonkieRecursiveChunker()

def test_chunk_file_sample_md(chunker):
    chunks = chunker.chunk_file(SAMPLE_MD)
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

def test_chunk_metadata_correctness(chunker):
    chunks = chunker.chunk_file(SAMPLE_MD)
    for chunk in chunks:
        # All metadata values should be str or int (converted to str by dataclass default)
        for v in chunk.metadata.values():
            assert isinstance(v, (str, int))

def test_is_file_supported(chunker):
    assert chunker.is_file_supported('foo.md')
    assert not chunker.is_file_supported('foo.txt')
    assert not chunker.is_file_supported('foo')

def test_empty_file(tmp_path, chunker):
    empty_file = tmp_path / 'empty.md'
    empty_file.write_text('')
    chunks = chunker.chunk_file(str(empty_file))
    assert chunks == []

def test_file_with_only_headers(tmp_path, chunker):
    header_file = tmp_path / 'headers.md'
    header_file.write_text('# Header1\n## Header2\n### Header3')
    chunks = chunker.chunk_file(str(header_file))
    # Should produce empty or header-only chunks
    for chunk in chunks:
        assert isinstance(chunk.text, str)
        assert chunk.metadata

def test_file_with_no_headers(tmp_path, chunker):
    no_header_file = tmp_path / 'noheaders.md'
    no_header_file.write_text('Just some text with no markdown headers.')
    chunks = chunker.chunk_file(str(no_header_file))
    # Should produce at least one chunk
    assert len(chunks) >= 1
    assert 'Just some text with no markdown headers.' in chunks[0].text

def test_nonexistent_file(chunker):
    with pytest.raises(FileNotFoundError):
        chunker.chunk_file('not_a_real_file.md') 