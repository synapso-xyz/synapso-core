from .interface import Chunker

class ChunkerFactory:
    """
    A factory for creating chunkers.
    """

    @staticmethod
    def create_chunker(chunker_type: str) -> Chunker:
        pass