import time
from typing import Any, List, Tuple

from .chunking.interface import Chunk
from .config_manager import GlobalConfig, get_config
from .data_store.factory import DataStoreFactory
from .data_store.interfaces import VectorStore
from .reranker.factory import RerankerFactory
from .reranker.interface import Reranker
from .summarizer.factory import Summarizer, SummarizerFactory
from .synapso_logger import get_logger
from .vectorizer.factory import VectorizerFactory
from .vectorizer.interface import Vectorizer

logger = get_logger(__name__)


def _assure_not_none(obj: Any, name: str) -> Any:
    if obj is None:
        raise ValueError(f"{name} not found")
    return obj


class QueryManager:
    def __init__(self):
        start_time = time.time()
        global_config: GlobalConfig = get_config()

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
            self.meta_store = DataStoreFactory.get_meta_store(
                global_config.meta_store.meta_db_type
            )
            self.private_store = DataStoreFactory.get_private_store(
                global_config.private_store.private_db_type
            )
            self.vector_store: VectorStore = _assure_not_none(
                DataStoreFactory.get_vector_store(
                    global_config.vector_store.vector_db_type
                ),
                "vector store",
            )
            logger.info(
                "QueryManager initialized in %s seconds", time.time() - start_time
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize QueryManager components: {e}"
            ) from e

    def query(self, query: str) -> List[Tuple[str, float]]:
        """
        Query the vector store and return a summary of the results.
        """
        start_time = time.time()
        query_chunk = Chunk(text=query)
        query_vector = self.vectorizer.vectorize(query_chunk)
        query_prep_time = time.time()
        logger.info("Query preparation took %s seconds", query_prep_time - start_time)
        results = self.vector_store.vector_search(query_vector)
        results_time = time.time()
        logger.info("Vector search took %s seconds", results_time - query_prep_time)
        results_with_text = [
            (
                result[0],
                self.private_store.get_by_chunk_id(result[0].vector_id),
                result[1],
            )
            for result in results
        ]
        rerank_start_time = time.time()
        reranked_results = self.reranker.rerank(results_with_text, query_vector)
        rerank_time = time.time()
        logger.info("Reranker took %s seconds", rerank_time - rerank_start_time)
        texts_with_scores = [(text, score) for _, text, score in reranked_results]
        summarize_start_time = time.time()
        summary = self.summarizer.summarize(query, texts_with_scores)
        summarize_time = time.time()
        logger.info("Summarizer took %s seconds", summarize_time - summarize_start_time)
        return summary
