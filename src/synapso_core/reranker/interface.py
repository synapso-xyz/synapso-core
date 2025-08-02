from abc import ABC, abstractmethod
from typing import List, Tuple

from ..data_store.models import Vector


class Reranker(ABC):
    """
    A reranker is a class that reranks a list of results.
    """

    @abstractmethod
    def rerank(
        self, results: List[Tuple[Vector, str, float]], query: Vector
    ) -> List[Tuple[Vector, str, float]]:
        """
        Rerank a list of results.

        Args:
            results: List of tuples containing vector, text, and score
            query: Query vector

        Returns:
            List of tuples containing reranked vector, text, and score
        """
        pass
