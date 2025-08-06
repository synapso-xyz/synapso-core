from .implementations import ChonkieRecursiveChunker, LangchainMarkdownChunker
from .interface import Chunker


class ChunkerFactory:
    """
    A factory for creating chunkers.
    """

    available_chunkers = {
        "langchain_markdown": LangchainMarkdownChunker,
        "chonkie_recursive": ChonkieRecursiveChunker,
        # "custom": CustomChunker,
    }

    @staticmethod
    def create_chunker(chunker_type: str) -> Chunker:
        if chunker_type not in ChunkerFactory.available_chunkers:
            raise ValueError(f"Invalid chunker type: {chunker_type}")
        return ChunkerFactory.available_chunkers[chunker_type]()
