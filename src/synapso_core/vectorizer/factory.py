"""
Vectorizer factory for Synapso Core.

This module provides a factory for creating vectorizer instances
based on configuration type strings.
"""

from .implementations.sentence_transformer_embeddings import (
    SentenceTransformerVectorizer,
)
from .interface import Vectorizer


class VectorizerFactory:
    """
    Factory for creating vectorizer instances.

    Provides a centralized way to instantiate different types
    of text vectorizers based on configuration.
    """

    available_vectorizers = {
        "sentence_transformer": SentenceTransformerVectorizer,
    }

    @staticmethod
    def create_vectorizer(vectorizer_type: str) -> Vectorizer:
        """
        Create a vectorizer instance of the specified type.

        Args:
            vectorizer_type: Type of vectorizer to create

        Returns:
            Vectorizer: A new vectorizer instance

        Raises:
            ValueError: If the vectorizer type is not supported
        """
        if vectorizer_type not in VectorizerFactory.available_vectorizers:
            raise ValueError(f"Vectorizer type {vectorizer_type} not found")
        return VectorizerFactory.available_vectorizers[vectorizer_type]()
