from .interface import Chunker
from .implementations.md.langchain_markdown_chunker import LangchainMarkdownChunker
from .implementations.md.chonkie_recursive_chunker import ChonkieRecursiveChunker

class ChunkerFactory:
    """
    A factory for creating chunkers.
    """

    available_chunkers = {
        "langchain_markdown": LangchainMarkdownChunker,
        "chonkie_recursive": ChonkieRecursiveChunker,
    }

    @staticmethod
    def create_chunker(chunker_type: str) -> Chunker:
        if chunker_type not in ChunkerFactory.available_chunkers:
            raise ValueError(f"Invalid chunker type: {chunker_type}")
        return ChunkerFactory.available_chunkers[chunker_type]()