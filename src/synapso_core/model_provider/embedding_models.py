"""
Embedding model implementations for Synapso Core.

This module provides concrete implementations of embedding models
using the MLX backend for generating text vector representations.
"""

from mlx_embeddings import load as load_mlx_embeddings

from .base import ModelProviderType, ModelWrapper


class EmbeddingsModel(ModelWrapper):
    """
    Base class for embedding model providers.

    Implements the common loading logic for embedding models
    using the MLX backend with lazy loading support.
    """

    def __init__(self, model_id: str):
        """
        Initialize the embedding model.

        Args:
            model_id: Identifier for the specific model
        """
        super().__init__(model_id, ModelProviderType.EMBEDDINGS)

    async def load(self):
        """
        Load the embedding model and tokenizer.

        Uses MLX's lazy loading to defer model initialization
        until actually needed.
        """
        model, tokenizer = load_mlx_embeddings(self.model_id, lazy=True)
        self._model = model
        self._tokenizer = tokenizer
        self._loaded = True


class ModernBertEmbeddingsModel(EmbeddingsModel):
    """
    ModernBert embeddings model provider.

    Uses the nomicai-modernbert-embed-base-4bit model optimized
    for generating high-quality text embeddings.
    """

    def __init__(self):
        """Initialize with the ModernBert model identifier."""
        self.model_id = "mlx-community/nomicai-modernbert-embed-base-4bit"
        super().__init__(self.model_id)


class SentenceTransformerEmbeddingsModel(EmbeddingsModel):
    """
    SentenceTransformer embeddings model provider.

    Uses the all-MiniLM-L6-v2-bf16 model optimized for
    sentence-level embeddings with reduced dimensionality.
    """

    def __init__(self):
        """Initialize with the SentenceTransformer model identifier."""
        self.model_id = "mlx-community/all-MiniLM-L6-v2-bf16"
        super().__init__(self.model_id)
