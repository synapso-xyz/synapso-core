from abc import ABC, abstractmethod


from ..persistence.interfaces import Vector

class Vectorizer(ABC):
    """
    A vectorizer is a class that vectorizes text.
    """
    
    @abstractmethod
    def vectorize(self, text: str) -> Vector:
        pass
