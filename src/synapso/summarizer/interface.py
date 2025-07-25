from abc import ABC, abstractmethod
from typing import List

from ..persistence.interfaces import Vector


class Summarizer(ABC):
    """
    A summarizer is a class that summarizes a list of results.
    """

    @abstractmethod
    def summarize(self, results: List[Vector]) -> str:
        pass
