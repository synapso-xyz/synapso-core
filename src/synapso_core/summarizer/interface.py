from abc import ABC, abstractmethod
from typing import List, Tuple


class Summarizer(ABC):
    """
    A summarizer is a class that summarizes a list of results.
    """

    @abstractmethod
    def summarize(self, question: str, results: List[Tuple[str, float]]) -> str:
        pass
