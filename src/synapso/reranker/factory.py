from .interface import Reranker

class RerankerFactory:
    """
    A factory for creating rerankers.
    """

    @staticmethod
    def create_reranker(reranker_type: str) -> Reranker:
        pass