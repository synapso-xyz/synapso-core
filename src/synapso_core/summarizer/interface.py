"""
Summarizer interface for Synapso Core.

This module defines the abstract interface for generating summaries
and answers based on search results and user queries.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple


class Summarizer(ABC):
    """
    Abstract interface for document summarization.

    A summarizer takes search results and generates coherent summaries
    or answers to user queries based on the retrieved content.
    """

    @abstractmethod
    async def summarize(self, question: str, results: List[Tuple[str, float]]) -> str:
        """
        Generate a summary answer to a question based on search results.

        Args:
            question: The user's question or query
            results: List of (text, relevance_score) tuples from search

        Returns:
            str: A coherent summary answer to the question
        """

    @abstractmethod
    async def run_summarizer_stream(
        self, question: str, results: List[Tuple[str, float]]
    ):
        """
        Generate a streaming summary answer to a question.

        Args:
            question: The user's question or query
            results: List of (text, relevance_score) tuples from search

        Yields:
            str: Tokens of the generated summary as they become available
        """
