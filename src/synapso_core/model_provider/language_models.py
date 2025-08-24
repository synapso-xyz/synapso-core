"""
Language model implementations for Synapso Core.

This module provides concrete implementations of language models
using the MLX backend for text generation and processing.
"""

from mlx_lm.utils import load as load_mlx_lm

from .base import ModelProviderType, ModelWrapper


class LanguageModel(ModelWrapper):
    """
    Base class for language model providers.

    Implements the common loading logic for language models
    using the MLX backend with lazy loading support.
    """

    def __init__(self, model_id: str):
        """
        Initialize the language model.

        Args:
            model_id: Identifier for the specific model
        """
        super().__init__(model_id, ModelProviderType.LANGUAGE_MODEL)

    async def load(self):
        """
        Load the language model and tokenizer.

        Uses MLX's lazy loading to defer model initialization
        until actually needed.
        """
        model, tokenizer = load_mlx_lm(self.model_id, lazy=True)
        self._model = model
        self._tokenizer = tokenizer
        self._loaded = True


class Qwen3LanguageModel(LanguageModel):
    """
    Qwen3 language model provider.

    Uses the Qwen3-Reranker-0.6B model optimized for
    reranking and instruction following tasks.
    """

    def __init__(self):
        """Initialize with the Qwen3 model identifier."""
        self.model_id = "mlx-community/Qwen3-Reranker-0.6B-mlx-6bit"
        super().__init__(self.model_id)


class Llama32LanguageModel(LanguageModel):
    """
    Llama32 language model provider.

    Uses the Llama-3.2-1B-Instruct model optimized for
    instruction following and text generation.
    """

    def __init__(self):
        """Initialize with the Llama32 model identifier."""
        self.model_id = "mlx-community/Llama-3.2-1B-Instruct-4bit"
        super().__init__(self.model_id)


class MistralLanguageModel(LanguageModel):
    """
    Mistral language model provider.

    Uses the TinyMistral-248M model optimized for
    efficient text generation with reduced model size.
    """

    def __init__(self):
        """Initialize with the TinyMistral model identifier."""
        self.model_id = "mlx-community/TinyMistral-248M-8bits"
        super().__init__(self.model_id)
