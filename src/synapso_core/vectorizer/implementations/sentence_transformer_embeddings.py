"""
Sentence Transformer vectorizer implementation for Synapso Core.

This module provides a vectorizer implementation using the Sentence Transformers
library with MLX backend for generating text embeddings.
"""

from typing import List

from mlx_embeddings import generate  # type: ignore

from ...chunking.interface import Chunk
from ...model_provider import ModelManager, ModelNames
from ...models import Vector, VectorMetadata
from ...utils import get_content_hash
from ..interface import Vectorizer


class SentenceTransformerVectorizer(Vectorizer):
    """
    Vectorizer using Sentence Transformers with MLX backend.

    Converts text chunks into vector embeddings using pre-trained
    transformer models optimized for sentence-level representations.
    """

    def __init__(self):
        """
        Initialize the vectorizer with model references.

        Uses the all-MiniLM-L6-v2 model, which is a fully offline model
        optimized for sentence embeddings.
        """
        # We use the all-MiniLM-L6-v2 model for now. It is a fully offline model.
        self.model = None
        self.tokenizer = None

    async def vectorize(self, chunk: Chunk) -> Vector:
        """
        Vectorize a single text chunk.

        Args:
            chunk: Text chunk to vectorize

        Returns:
            Vector: Vector object with embedding and metadata
        """
        model_manager = ModelManager.get_instance()
        async with model_manager.acquire(
            ModelNames.SENTENCE_TRANSFORMER_EMBEDDINGS_MODEL
        ) as model_provider:
            await model_provider.ensure_loaded()
            self.model = model_provider.model
            self.tokenizer = model_provider.tokenizer

            output = generate(self.model, self.tokenizer, texts=[chunk.text])
            vector_vals = output.text_embeds[0]  # type: ignore
            print(f"Vector values: {vector_vals}")
            return Vector(
                vector=vector_vals.tolist(),
                vector_id=get_content_hash(chunk.text),
                metadata=VectorMetadata(
                    content_hash=get_content_hash(chunk.text),
                    cortex_id=chunk.metadata.get("cortex_id"),
                    source_file_id=chunk.metadata.get("source_file_id"),
                    additional_data=chunk.metadata,
                ),
            )

    async def vectorize_batch(self, chunks: List[Chunk]) -> List[Vector]:
        """
        Vectorize multiple text chunks efficiently.

        Args:
            chunks: List of text chunks to vectorize

        Returns:
            List[Vector]: List of vector objects with embeddings and metadata
        """
        if not chunks:
            return []

        model_manager = ModelManager.get_instance()
        async with model_manager.acquire(
            ModelNames.SENTENCE_TRANSFORMER_EMBEDDINGS_MODEL
        ) as model_provider:
            await model_provider.ensure_loaded()
            self.model = model_provider.model
            self.tokenizer = model_provider.tokenizer
            output = generate(
                self.model, self.tokenizer, texts=[chunk.text for chunk in chunks]
            )
            vector_vals = output.text_embeds  # type: ignore
            return [
                Vector(
                    vector=vector_vals[i].tolist(),
                    vector_id=get_content_hash(chunks[i].text),
                    metadata=VectorMetadata(
                        content_hash=get_content_hash(chunks[i].text),
                        cortex_id=chunks[i].metadata.get("cortex_id"),
                        source_file_id=chunks[i].metadata.get("source_file_id"),
                        additional_data=chunks[i].metadata,
                    ),
                )
                for i in range(len(chunks))
            ]
