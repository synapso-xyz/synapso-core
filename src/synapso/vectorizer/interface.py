from abc import ABC, abstractmethod
from typing import List

from ..persistence.interfaces import Vector
from ..chunking.interface import Chunk

class Vectorizer(ABC):
    """
    A vectorizer is a class that vectorizes text.
    """
    
    @abstractmethod
    def vectorize(self, chunks: List[Chunk]) -> Vector:
        pass
