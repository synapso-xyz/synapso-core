"""
Reranker interface for Synapso Core.

This module defines the abstract interface for reranking search results
to improve relevance and accuracy of document retrieval.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

from ..models import Vector


class Reranker(ABC):
    """
    Abstract interface for result reranking.

    A reranker takes initial search results and reorders them based on
    relevance to the query, improving the quality of document retrieval.
    """

    @abstractmethod
    async def rerank(
        self,
        results: List[Tuple[Vector, str, float]],
        query: Vector,
        query_text: str = "",
    ) -> List[Tuple[Vector, str, float]]:
        """
        Rerank a list of search results.

        Args:
            results: List of tuples containing (vector, text, score)
            query: Query vector for relevance comparison
            query_text: Optional text representation of the query

        Returns:
            List[Tuple[Vector, str, float]]: Reranked list of (vector, text, score) tuples
        """
