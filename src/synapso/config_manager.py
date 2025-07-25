import os
from typing import ClassVar

import yaml
from pydantic import BaseModel, field_validator

SYNAPSO_HOME = os.getenv("SYNAPSO_HOME", os.getcwd())
CONFIG_FILE = os.path.join(SYNAPSO_HOME, "config.yaml")


class MetaStoreConfig(BaseModel):
    available_db_types: ClassVar[list[str]] = ["sqlite"]
    meta_db_type: str = "sqlite"
    meta_db_path: str = "meta.db"

    @field_validator("meta_db_type")
    def validate_db_type(cls, v):
        if v not in cls.available_db_types:
            raise ValueError(
                f"meta_db_type must be one of {cls.available_db_types}, got '{v}'"
            )
        return v


class PrivateStoreConfig(BaseModel):
    available_db_types: ClassVar[list[str]] = ["sqlite"]
    private_db_type: str = "sqlite"
    private_db_path: str = "private.db"

    @field_validator("private_db_type")
    def validate_db_type(cls, v):
        if v not in cls.available_db_types:
            raise ValueError(
                f"private_db_type must be one of {cls.available_db_types}, got '{v}'"
            )
        return v


class VectorStoreConfig(BaseModel):
    available_db_types: ClassVar[list[str]] = ["sqlite"]
    vector_db_type: str = "sqlite"
    vector_db_path: str = "vector.db"

    @field_validator("vector_db_type")
    def validate_db_type(cls, v):
        if v not in cls.available_db_types:
            raise ValueError(
                f"vector_db_type must be one of {cls.available_db_types}, got '{v}'"
            )
        return v


class RerankerConfig(BaseModel):
    available_types: ClassVar[list[str]] = ["bm25"]
    reranker_type: str = "bm25"
    k1: float = 1.2
    b: float = 0.75

    @field_validator("reranker_type")
    def validate_type(cls, v):
        if v not in cls.available_types:
            raise ValueError(
                f"reranker_type must be one of {cls.available_types}, got '{v}'"
            )
        return v


class SummarizerConfig(BaseModel):
    available_types: ClassVar[list[str]] = ["textrank"]
    summarizer_type: str = "textrank"
    top_k: int = 5
    min_score: float = 0.1

    @field_validator("summarizer_type")
    def validate_type(cls, v):
        if v not in cls.available_types:
            raise ValueError(
                f"summarizer_type must be one of {cls.available_types}, got '{v}'"
            )
        return v


class VectorizerConfig(BaseModel):
    available_types: ClassVar[list[str]] = ["sentence_transformer"]
    vectorizer_type: str = "sentence_transformer"
    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"

    @field_validator("vectorizer_type")
    def validate_type(cls, v):
        if v not in cls.available_types:
            raise ValueError(
                f"vectorizer_type must be one of {cls.available_types}, got '{v}'"
            )
        return v


class ChunkerConfig(BaseModel):
    available_types: ClassVar[list[str]] = ["chonkie_recursive"]
    chunker_type: str = "chonkie_recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 100

    @field_validator("chunker_type")
    def validate_type(cls, v):
        if v not in cls.available_types:
            raise ValueError(
                f"chunker_type must be one of {cls.available_types}, got '{v}'"
            )
        return v


class GlobalConfig(BaseModel):
    meta_store: MetaStoreConfig = MetaStoreConfig()
    private_store: PrivateStoreConfig = PrivateStoreConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    reranker: RerankerConfig = RerankerConfig()
    summarizer: SummarizerConfig = SummarizerConfig()
    vectorizer: VectorizerConfig = VectorizerConfig()
    chunker: ChunkerConfig = ChunkerConfig()


def get_config(config_file: str = CONFIG_FILE) -> GlobalConfig:
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file {config_file} not found")
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
        return GlobalConfig(**(config_dict or {}))


if __name__ == "__main__":
    config = get_config()
    print(config.model_dump_json(indent=2))
