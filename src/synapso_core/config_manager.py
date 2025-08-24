"""
Configuration management for Synapso Core.

This module provides configuration classes and utilities for managing
various components of the Synapso system including data stores,
vectorizers, chunkers, rerankers, and summarizers.
"""

import os
from abc import ABC
from pathlib import Path
from typing import ClassVar

import yaml
from pydantic import BaseModel, field_validator

from .synapso_logger import get_logger

logger = get_logger(__name__)


DEFAULT_SYNAPSO_HOME = Path("~/.synapso").expanduser().resolve()
SYNAPSO_HOME = Path(os.getenv("SYNAPSO_HOME", str(DEFAULT_SYNAPSO_HOME)))
CONFIG_FILE = SYNAPSO_HOME / "config.yaml"


class BaseConfig(BaseModel, ABC):
    """
    Base configuration class with common validation functionality.

    Provides a common interface for all configuration classes
    with type validation capabilities.
    """

    available_types: ClassVar[list[str]]

    @classmethod
    def validate_type_field(cls, value: str, field_name: str) -> str:
        """
        Validate that a field value is one of the available types.

        Args:
            value: The value to validate
            field_name: Name of the field being validated

        Returns:
            The validated value

        Raises:
            ValueError: If the value is not in available_types
        """
        if value not in cls.available_types:
            raise ValueError(
                f"{field_name} must be one of {cls.available_types}, got '{value}'"
            )
        return value


class MetaStoreConfig(BaseConfig):
    """
    Configuration for metadata storage.

    Controls the type and location of the metadata database
    used for storing file and job information.
    """

    available_types: ClassVar[list[str]] = ["sqlite"]
    meta_db_type: str = "sqlite"
    meta_db_path: str = "meta.db"

    @field_validator("meta_db_type")
    @classmethod
    def validate_db_type(cls, v):
        """Validate the database type field."""
        return cls.validate_type_field(v, "meta_db_type")


class PrivateStoreConfig(BaseConfig):
    """
    Configuration for private data storage.

    Controls the type and location of the private database
    used for storing document chunks and content.
    """

    available_types: ClassVar[list[str]] = ["sqlite"]
    private_db_type: str = "sqlite"
    private_db_path: str = "private.db"

    @field_validator("private_db_type")
    @classmethod
    def validate_db_type(cls, v):
        """Validate the database type field."""
        return cls.validate_type_field(v, "private_db_type")


class VectorStoreConfig(BaseConfig):
    """
    Configuration for vector storage.

    Controls the type and location of the vector database
    used for storing document embeddings.
    """

    available_types: ClassVar[list[str]] = ["sqlite"]
    vector_db_type: str = "sqlite"
    vector_db_path: str = "vector.db"

    @field_validator("vector_db_type")
    @classmethod
    def validate_db_type(cls, v):
        """Validate the database type field."""
        return cls.validate_type_field(v, "vector_db_type")


class RerankerConfig(BaseConfig):
    """
    Configuration for document reranking.

    Controls the reranking algorithm and its parameters
    for improving search result relevance.
    """

    available_types: ClassVar[list[str]] = ["bm25", "modernbert", "qwen3"]
    reranker_type: str = "bm25"
    k1: float = 1.2
    b: float = 0.75

    @field_validator("reranker_type")
    @classmethod
    def validate_type(cls, v):
        """Validate the reranker type field."""
        return cls.validate_type_field(v, "reranker_type")


class SummarizerConfig(BaseConfig):
    """
    Configuration for document summarization.

    Controls the summarization algorithm used for
    generating document summaries.
    """

    available_types: ClassVar[list[str]] = ["instruct"]
    summarizer_type: str = "instruct"

    @field_validator("summarizer_type")
    @classmethod
    def validate_type(cls, v):
        """Validate the summarizer type field."""
        return cls.validate_type_field(v, "summarizer_type")


class VectorizerConfig(BaseConfig):
    """
    Configuration for document vectorization.

    Controls the embedding model and device used for
    converting text to vector representations.
    """

    available_types: ClassVar[list[str]] = ["sentence_transformer"]
    vectorizer_type: str = "sentence_transformer"
    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"

    @field_validator("vectorizer_type")
    @classmethod
    def validate_type(cls, v):
        """Validate the vectorizer type field."""
        return cls.validate_type_field(v, "vectorizer_type")


class ChunkerConfig(BaseConfig):
    """
    Configuration for document chunking.

    Controls the chunking algorithm and parameters used
    for splitting documents into processable chunks.
    """

    available_types: ClassVar[list[str]] = [
        "chonkie_recursive",
        "langchain_markdown",
        "simple_txt",
    ]
    chunker_type: str = "chonkie_recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 100

    @field_validator("chunker_type")
    @classmethod
    def validate_type(cls, v):
        """Validate the chunker type field."""
        return cls.validate_type_field(v, "chunker_type")


class GlobalConfig(BaseModel):
    """
    Global configuration container for the entire Synapso system.

    Aggregates all component-specific configurations into
    a single configuration object.
    """

    meta_store: MetaStoreConfig = MetaStoreConfig()
    private_store: PrivateStoreConfig = PrivateStoreConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    reranker: RerankerConfig = RerankerConfig()
    summarizer: SummarizerConfig = SummarizerConfig()
    vectorizer: VectorizerConfig = VectorizerConfig()
    chunker: ChunkerConfig = ChunkerConfig()


def get_config(config_file: str = str(CONFIG_FILE)) -> GlobalConfig:
    """
    Load and parse the global configuration from a YAML file.

    Args:
        config_file: Path to the configuration file

    Returns:
        GlobalConfig: The parsed configuration object

    Raises:
        FileNotFoundError: If the config file doesn't exist
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file {config_file} not found")
    with open(config_file, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
        return GlobalConfig(**(config_dict or {}))


if __name__ == "__main__":
    config = get_config()
    logger.info(config.model_dump_json(indent=2))
