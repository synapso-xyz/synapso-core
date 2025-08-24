"""
Reranker factory for Synapso Core.

This module provides a factory for creating reranker instances
based on configuration type strings.
"""

from .bm25_reranker import BM25Reranker
from .interface import Reranker
from .modernbert_reranker import ModernBertReranker
from .qwen3_reranker import Qwen3Reranker


class RerankerFactory:
    """
    Factory for creating reranker instances.

    Provides a centralized way to instantiate different types
    of result rerankers based on configuration.
    """

    available_rerankers = {
        "bm25": BM25Reranker,
        "modernbert": ModernBertReranker,
        "qwen3": Qwen3Reranker,
    }

    @staticmethod
    def create_reranker(reranker_type: str) -> Reranker | None:
        """
        Create a reranker instance based on the specified type.

        Args:
            reranker_type: Type of reranker to create

        Returns:
            Reranker | None: Reranker instance if type is supported, None otherwise
        """
        if reranker_type not in RerankerFactory.available_rerankers:
            return None

        return RerankerFactory.available_rerankers[reranker_type]()
