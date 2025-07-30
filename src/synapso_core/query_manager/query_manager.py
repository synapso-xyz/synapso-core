from typing import Any

from ..chunking.interface import Chunk
from ..config_manager import GlobalConfig, get_config
from ..persistence.factory import (
    MetaStoreFactory,
    PrivateStoreFactory,
    VectorStoreFactory,
)
from ..persistence.interfaces import VectorStore
from ..reranker.factory import RerankerFactory
from ..reranker.interface import Reranker
from ..summarizer.factory import SummarizerFactory
from ..summarizer.interface import Summarizer
from ..vectorizer.factory import VectorizerFactory
from ..vectorizer.interface import Vectorizer
from .query_config import QueryConfig


def _assure_not_none(obj: Any, name: str) -> Any:
    if obj is None:
        raise ValueError(f"{name} not found")
    return obj


class QueryManager:
    def __init__(self, query_config: QueryConfig):
        global_config: GlobalConfig = get_config()
        self.query_config = query_config
        try:
            self.vectorizer: Vectorizer = _assure_not_none(
                VectorizerFactory.create_vectorizer(
                    global_config.vectorizer.vectorizer_type
                ),
                "vectorizer",
            )
            self.reranker: Reranker = _assure_not_none(
                RerankerFactory.create_reranker(global_config.reranker.reranker_type),
                "reranker",
            )
            self.summarizer: Summarizer = _assure_not_none(
                SummarizerFactory.create_summarizer(
                    global_config.summarizer.summarizer_type
                ),
                "summarizer",
            )
            self.meta_store = MetaStoreFactory.get_meta_store(
                global_config.meta_store.meta_db_type
            )
            self.private_store = PrivateStoreFactory.get_private_store(
                global_config.private_store.private_db_type
            )
            self.vector_store: VectorStore = _assure_not_none(
                VectorStoreFactory.get_vector_store(
                    global_config.vector_store.vector_db_type
                ),
                "vector store",
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize QueryManager components: {e}"
            ) from e

    def query(self, query: str) -> str:
        """
        Query the vector store and return a summary of the results.
        """
        query_chunk = Chunk(text=query)
        query_vector = self.vectorizer.vectorize(query_chunk)
        results = self.vector_store.vector_search(query_vector)
        reranked_results = self.reranker.rerank(results, query_vector)
        vectors_only = [vector for vector, _ in reranked_results]
        summary = self.summarizer.summarize(vectors_only)
        return summary
