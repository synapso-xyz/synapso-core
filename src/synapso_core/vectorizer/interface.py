"""
Vectorizer interface for Synapso Core.

This module defines the abstract interface for converting text chunks
into vector embeddings for similarity search and retrieval.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import List

from ..chunking.interface import Chunk
from ..models import Vector


class Vectorizer(ABC):
    """
    Abstract interface for text vectorization.

    A vectorizer converts text chunks into numerical vector representations
    (embeddings) that can be used for similarity search and retrieval.
    """

    async def vectorize_batch(self, chunks: List[Chunk]) -> List[Vector]:
        """
        Vectorize a batch of text chunks.

        Args:
            chunks: List of text chunks to vectorize

        Returns:
            List[Vector]: List of vector objects with embeddings and metadata
        """
        return await asyncio.gather(*[self.vectorize(chunk) for chunk in chunks])

    @abstractmethod
    async def vectorize(self, chunk: Chunk) -> Vector:
        """
        Vectorize a single text chunk.

        Args:
            chunk: Text chunk to vectorize

        Returns:
            Vector: Vector object containing the embedding and metadata
        """
