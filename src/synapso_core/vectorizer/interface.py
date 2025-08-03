from abc import ABC, abstractmethod
from typing import List

from ..chunking.interface import Chunk
from ..models import Vector


class Vectorizer(ABC):
    """
    A vectorizer is a class that vectorizes text.
    """

    def vectorize_batch(self, chunks: List[Chunk]) -> List[Vector]:
        """
        Vectorize a batch of chunks.
        """
        return [self.vectorize(chunk) for chunk in chunks]

    @abstractmethod
    def vectorize(self, chunk: Chunk) -> Vector:
        """
        Vectorize a single chunk.
        """
        pass
