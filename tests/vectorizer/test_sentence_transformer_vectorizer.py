import numpy as np
import pytest

from src.synapso_core.chunking.interface import Chunk
from src.synapso_core.utils import get_content_hash
from src.synapso_core.vectorizer.implementations.sentence_transformer_embeddings import (
    SentenceTransformerVectorizer,
)
from src.synapso_core.vectorizer.interface import Vector


@pytest.fixture(scope="session")
def vectorizer():
    return SentenceTransformerVectorizer()


def test_vectorize_single_chunk(vectorizer):
    chunk = Chunk(text="Hello world.")
    v = vectorizer.vectorize(chunk)
    assert isinstance(v, Vector)
    assert isinstance(v.vector, list)
    assert all(isinstance(x, float) for x in v.vector)
    assert isinstance(v.vector_id, str)
    assert v.metadata.to_dict()["content_hash"] == get_content_hash(chunk.text)
    assert len(v.vector) > 0


def test_vectorize_multiple_chunks(vectorizer):
    chunks = [Chunk(text="Hello world."), Chunk(text="Goodbye world.")]
    vectors = vectorizer.vectorize_batch(chunks)
    assert len(vectors) == 2
    for v, c in zip(vectors, chunks):
        assert isinstance(v.vector, list)
        assert v.metadata.to_dict()["content_hash"] == get_content_hash(c.text)


def test_vectorize_empty_text(vectorizer):
    chunk = Chunk(text="")
    vectors = vectorizer.vectorize_batch([chunk])
    assert len(vectors) == 1
    assert isinstance(vectors[0].vector, list)
    assert len(vectors[0].vector) > 0


def test_vectorize_with_metadata(vectorizer):
    chunk = Chunk(text="Test text", metadata={"foo": "bar"})
    v = vectorizer.vectorize(chunk)
    meta = v.metadata.to_dict()
    assert meta["additional_data"]["foo"] == "bar"
    assert "content_hash" in meta


def test_vector_shape_and_type(vectorizer):
    chunk = Chunk(text="Test shape")
    v = vectorizer.vectorize(chunk)
    arr = np.array(v.vector)
    assert arr.ndim == 1
    assert arr.dtype == float or arr.dtype == np.float32 or arr.dtype == np.float64


def test_content_hash_consistency():
    chunk1 = Chunk(text="Same text")
    chunk2 = Chunk(text="Same text")
    assert get_content_hash(chunk1.text) == get_content_hash(chunk2.text)


def test_unicode_text(vectorizer):
    chunk = Chunk(text="こんにちは世界🌏")
    v = vectorizer.vectorize(chunk)
    assert isinstance(v, Vector)
    assert isinstance(v.vector, list)


def test_duplicate_chunks_same_vector_id(vectorizer):
    chunk1 = Chunk(text="Repeat")
    chunk2 = Chunk(text="Repeat")
    vectors = vectorizer.vectorize_batch([chunk1, chunk2])
    assert vectors[0].vector_id == vectors[1].vector_id


def test_empty_chunk_list(vectorizer):
    vectors = vectorizer.vectorize_batch([])
    assert vectors == []
