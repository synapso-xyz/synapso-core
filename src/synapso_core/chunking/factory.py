"""
Chunker factory for Synapso Core.

This module provides a factory for creating chunker instances
based on configuration type strings.
"""

from .implementations import (
    ChonkieRecursiveChunker,
    LangchainMarkdownChunker,
    SimpleTxtChunker,
)
from .interface import Chunker

AVAILABLE_CHUNKER_TYPES = {
    "langchain_markdown": LangchainMarkdownChunker,
    "chonkie_recursive": ChonkieRecursiveChunker,
    "simple_txt": SimpleTxtChunker,
}


class ChunkerFactory:
    """
    Factory for creating chunker instances.

    Provides a centralized way to instantiate different types
    of document chunkers based on configuration.
    """

    @staticmethod
    def create_chunker(chunker_type: str) -> Chunker:
        """
        Create a chunker instance of the specified type.

        Args:
            chunker_type: Type of chunker to create

        Returns:
            Chunker: A new chunker instance

        Raises:
            ValueError: If the chunker type is not supported
        """
        if chunker_type not in AVAILABLE_CHUNKER_TYPES:
            raise ValueError(f"Invalid chunker type: {chunker_type}")
        return AVAILABLE_CHUNKER_TYPES[chunker_type]()
