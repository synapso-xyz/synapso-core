from .interface import Vectorizer


class VectorizerFactory:
    """
    A factory for creating vectorizers.
    """

    @staticmethod
    def create_vectorizer(vectorizer_type: str) -> Vectorizer | None:
        pass
