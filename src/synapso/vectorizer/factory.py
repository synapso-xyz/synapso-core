from .implementations.sentence_transformer_embeddings import (
    SentenceTransformerVectorizer,
)
from .interface import Vectorizer


class VectorizerFactory:
    """
    A factory for creating vectorizers.
    """

    available_vectorizers = {
        "sentence_transformer": SentenceTransformerVectorizer,
    }

    @staticmethod
    def create_vectorizer(vectorizer_type: str) -> Vectorizer:
        if vectorizer_type not in VectorizerFactory.available_vectorizers:
            raise ValueError(f"Vectorizer type {vectorizer_type} not found")
        return VectorizerFactory.available_vectorizers[vectorizer_type]()
