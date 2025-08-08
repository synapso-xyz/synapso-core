from abc import ABC, abstractmethod
from typing import List, Tuple


class Summarizer(ABC):
    """
    A summarizer is a class that summarizes a list of results.
    """

    @abstractmethod
    def summarize(self, question: str, results: List[Tuple[str, float]]) -> str:
        pass

    @abstractmethod
    async def run_summarizer_stream(
        self, question: str, results: List[Tuple[str, float]]
    ):
        pass
