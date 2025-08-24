"""
Model provider base classes for Synapso Core.

This module defines the base classes and enums for model providers,
including language models and embedding models.
"""

from abc import ABC
from enum import Enum
from typing import Any, Optional


class ModelProviderType(Enum):
    """
    Enumeration of supported model provider types.

    Defines the categories of models that can be managed
    by the model provider system.
    """

    LANGUAGE_MODEL = "language_model"
    EMBEDDINGS = "embeddings"


class ModelWrapper(ABC):
    """
    Abstract base class for model providers.

    Provides a common interface for managing language models
    and embedding models with loading, unloading, and access capabilities.
    """

    def __init__(self, model_id: str, model_type: ModelProviderType):
        """
        Initialize the model wrapper.

        Args:
            model_id: Identifier for the specific model
            model_type: Type of model (language or embeddings)
        """
        self.model_id = model_id
        self.model_type = model_type
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._loaded = False

    async def load(self):
        """
        Load the model and tokenizer.

        This method must be implemented by subclasses to handle
        the specific loading logic for different model types.

        Raises:
            NotImplementedError: Subclasses must implement this method
        """
        raise NotImplementedError("Subclasses must implement this method.")

    async def ensure_loaded(self):
        """
        Ensure the model is loaded, loading it if necessary.

        Checks if the model is already loaded and loads it
        asynchronously if it hasn't been loaded yet.
        """
        if not self._loaded:
            await self.load()
            self._loaded = True

    @property
    def model(self) -> Any:
        """
        Get the loaded model instance.

        Returns:
            The model instance

        Raises:
            ValueError: If the model is not loaded
        """
        if self._model is None:
            raise ValueError("Model not loaded. Call load() first.")
        return self._model

    @property
    def tokenizer(self) -> Any:
        """
        Get the loaded tokenizer instance.

        Returns:
            The tokenizer instance

        Raises:
            ValueError: If the tokenizer is not loaded
        """
        if self._tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load() first.")
        return self._tokenizer

    async def unload(self):
        """
        Unload the model and tokenizer.

        Frees memory by removing references to the loaded
        model and tokenizer instances.
        """
        del self._model
        del self._tokenizer
        self._loaded = False
