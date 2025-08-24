"""
Vectorization for Synapso Core.

This module provides interfaces and implementations for converting
text documents into vector embeddings for similarity search.
"""

from .factory import VectorizerFactory
from .interface import Vectorizer

__all__ = ["VectorizerFactory", "Vectorizer"]
