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
    A factory for creating chunkers.
    """

    @staticmethod
    def create_chunker(chunker_type: str) -> Chunker:
        if chunker_type not in AVAILABLE_CHUNKER_TYPES:
            raise ValueError(f"Invalid chunker type: {chunker_type}")
        return AVAILABLE_CHUNKER_TYPES[chunker_type]()
