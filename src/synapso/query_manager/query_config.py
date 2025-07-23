from dataclasses import dataclass


@dataclass
class QueryConfig:
    vectorizer_type: str
    reranker_type: str
    summarizer_type: str