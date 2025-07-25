from ..persistence.factory import (
    MetaStoreFactory,
    PrivateStoreFactory,
    VectorStoreFactory,
)
from ..reranker.factory import RerankerFactory
from ..summarizer.factory import SummarizerFactory
from ..vectorizer.factory import VectorizerFactory
from .query_config import QueryConfig


class QueryManager:
    def __init__(self, query_config: QueryConfig):
        self.query_config = query_config
        try:
            self.vectorizer = VectorizerFactory.create_vectorizer(
                query_config.vectorizer_type
            )
            self.reranker = RerankerFactory.create_reranker(query_config.reranker_type)
            self.summarizer = SummarizerFactory.create_summarizer(
                query_config.summarizer_type
            )
            self.meta_store = MetaStoreFactory.get_meta_store()
            self.private_store = PrivateStoreFactory.get_private_store()
            self.vector_store = VectorStoreFactory.get_vector_store()
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize QueryManager components: {e}"
            ) from e

    def query(self, query: str) -> str:
        """
        Query the vector store and return a summary of the results.
        """
        query_vector = self.vectorizer.vectorize(query)
        results = self.vector_store.search(query_vector)
        reranked_results = self.reranker.rerank(results, query_vector)
        vectors_only = [vector for vector, _ in reranked_results]
        summary = self.summarizer.summarize(vectors_only)
        return summary
