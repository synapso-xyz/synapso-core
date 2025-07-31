from .bm25_reranker import BM25Reranker
from .interface import Reranker


class RerankerFactory:
    """
    A factory for creating rerankers.
    """

    available_rerankers = {
        "bm25": BM25Reranker,
    }

    @staticmethod
    def create_reranker(reranker_type: str) -> Reranker | None:
        """
        Create a reranker instance based on the specified type.

        Args:
            reranker_type: Type of reranker to create

        Returns:
            Reranker instance or None if type not found
        """
        if reranker_type not in RerankerFactory.available_rerankers:
            return None

        return RerankerFactory.available_rerankers[reranker_type]()
