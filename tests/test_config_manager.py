import os
import tempfile

import pytest

from src.synapso_core import config_manager

RESOURCES_DIR = os.path.join(os.path.dirname(__file__), "../resources/config")


def test_load_valid_config():
    config_path = os.path.join(RESOURCES_DIR, "test_valid_config.yaml")
    config = config_manager.get_config(config_path)
    assert config.meta_store.meta_db_type == "sqlite"
    assert config.meta_store.meta_db_path == "test_meta.db"
    assert config.private_store.private_db_type == "sqlite"
    assert config.vector_store.vector_db_type == "sqlite"
    assert config.reranker.reranker_type == "bm25"
    assert config.summarizer.summarizer_type == "textrank"
    assert config.vectorizer.vectorizer_type == "sentence_transformer"
    assert config.chunker.chunker_type == "chonkie_recursive"
    assert config.chunker.chunk_size == 500
    assert config.chunker.chunk_overlap == 50


def test_load_invalid_config():
    config_path = os.path.join(RESOURCES_DIR, "test_invalid_config.yaml")
    with pytest.raises(ValueError):
        config_manager.get_config(config_path)


def test_load_empty_config():
    with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
        tmp.write("")
        tmp_path = tmp.name
    try:
        config = config_manager.get_config(tmp_path)
        # Should use all defaults
        assert config.meta_store.meta_db_type == "sqlite"
        assert config.private_store.private_db_type == "sqlite"
        assert config.vector_store.vector_db_type == "sqlite"
        assert config.reranker.reranker_type == "bm25"
        assert config.summarizer.summarizer_type == "textrank"
        assert config.vectorizer.vectorizer_type == "sentence_transformer"
        assert config.chunker.chunker_type == "chonkie_recursive"
    finally:
        os.remove(tmp_path)
