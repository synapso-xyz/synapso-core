import os
from abc import ABC
from typing import ClassVar

import yaml
from pydantic import BaseModel, field_validator

SYNAPSO_HOME = os.getenv("SYNAPSO_HOME", os.getcwd())
CONFIG_FILE = os.path.join(SYNAPSO_HOME, "config.yaml")


class BaseConfig(BaseModel, ABC):
    available_types: ClassVar[list[str]]

    @classmethod
    def validate_type_field(cls, value: str, field_name: str) -> str:
        if value not in cls.available_types:
            raise ValueError(
                f"{field_name} must be one of {cls.available_types}, got '{value}'"
            )
        return value


class MetaStoreConfig(BaseConfig):
    available_db_types: ClassVar[list[str]] = ["sqlite"]
    meta_db_type: str = "sqlite"
    meta_db_path: str = "meta.db"

    @field_validator("meta_db_type")
    def validate_db_type(cls, v):
        return cls.validate_type_field(v, "meta_db_type")


class PrivateStoreConfig(BaseConfig):
    available_db_types: ClassVar[list[str]] = ["sqlite"]
    private_db_type: str = "sqlite"
    private_db_path: str = "private.db"

    @field_validator("private_db_type")
    def validate_db_type(cls, v):
        return cls.validate_type_field(v, "private_db_type")


class VectorStoreConfig(BaseConfig):
    available_db_types: ClassVar[list[str]] = ["sqlite"]
    vector_db_type: str = "sqlite"
    vector_db_path: str = "vector.db"

    @field_validator("vector_db_type")
    def validate_db_type(cls, v):
        return cls.validate_type_field(v, "vector_db_type")


class RerankerConfig(BaseConfig):
    available_types: ClassVar[list[str]] = ["bm25"]
    reranker_type: str = "bm25"
    k1: float = 1.2
    b: float = 0.75

    @field_validator("reranker_type")
    def validate_type(cls, v):
        return cls.validate_type_field(v, "reranker_type")


class SummarizerConfig(BaseConfig):
    available_types: ClassVar[list[str]] = ["textrank"]
    summarizer_type: str = "textrank"
    top_k: int = 5
    min_score: float = 0.1

    @field_validator("summarizer_type")
    def validate_type(cls, v):
        return cls.validate_type_field(v, "summarizer_type")


class VectorizerConfig(BaseConfig):
    available_types: ClassVar[list[str]] = ["sentence_transformer"]
    vectorizer_type: str = "sentence_transformer"
    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"

    @field_validator("vectorizer_type")
    def validate_type(cls, v):
        return cls.validate_type_field(v, "vectorizer_type")


class ChunkerConfig(BaseConfig):
    available_types: ClassVar[list[str]] = ["chonkie_recursive"]
    chunker_type: str = "chonkie_recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 100

    @field_validator("chunker_type")
    def validate_type(cls, v):
        return cls.validate_type_field(v, "chunker_type")


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
