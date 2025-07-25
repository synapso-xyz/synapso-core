import os

import yaml
from pydantic import BaseModel

SYNAPSO_HOME = os.getenv("SYNAPSO_HOME", os.getcwd())
CONFIG_FILE = os.path.join(SYNAPSO_HOME, "config.yaml")


class GlobalConfig(BaseModel):
    meta_store_type: str = "sqlite"
    private_store_type: str = "sqlite"
    vector_store_type: str = "sqlite"
    reranker_type: str = "bm25"
    summarizer_type: str = "textrank"
    vectorizer_type: str = "sentence_transformer"
    chunker_type: str = "chonkie_recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 100


def get_config() -> GlobalConfig:
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"Config file {CONFIG_FILE} not found")
    with open(CONFIG_FILE, "r") as f:
        return GlobalConfig(**yaml.safe_load(f))
