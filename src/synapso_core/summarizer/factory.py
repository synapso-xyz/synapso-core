"""
Summarizer factory for Synapso Core.

This module provides a factory for creating summarizer instances
based on configuration type strings.
"""

from .instruct_summarizer import InstructSummarizer
from .interface import Summarizer


class SummarizerFactory:
    """
    Factory for creating summarizer instances.

    Provides a centralized way to instantiate different types
    of document summarizers based on configuration.
    """

    @staticmethod
    def create_summarizer(summarizer_type: str) -> Summarizer:
        """
        Create a summarizer instance of the specified type.

        Args:
            summarizer_type: Type of summarizer to create

        Returns:
            Summarizer: A new summarizer instance

        Raises:
            ValueError: If the summarizer type is not supported
        """
        available_summarizers = {
            "instruct": InstructSummarizer,
        }
        if summarizer_type not in available_summarizers:
            raise ValueError(f"Invalid summarizer type: {summarizer_type}")
        return available_summarizers[summarizer_type]()
