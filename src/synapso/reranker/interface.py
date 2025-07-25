from abc import ABC, abstractmethod
from typing import List, Tuple

from ..persistence.interfaces import Vector


class Reranker(ABC):
    """
    A reranker is a class that reranks a list of results.
    """

    @abstractmethod
    def rerank(
        self, results: List[Tuple[Vector, float]], query: Vector
    ) -> List[Tuple[Vector, float]]:
        pass
