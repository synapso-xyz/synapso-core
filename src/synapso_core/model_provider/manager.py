"""
Model provider manager for Synapso Core.

This module provides the ModelManager class for managing the lifecycle
of language models and embedding models with automatic loading/unloading.
"""

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Optional

from .base import ModelWrapper
from .embedding_models import (
    ModernBertEmbeddingsModel,
    SentenceTransformerEmbeddingsModel,
)
from .language_models import (
    Llama32LanguageModel,
    MistralLanguageModel,
    Qwen3LanguageModel,
)


class ModelNames(Enum):
    """
    Enumeration of available model types.

    Defines the specific models that can be managed
    by the ModelManager.
    """

    QWEN3_LANGUAGE_MODEL = 1
    LLAMA32_LANGUAGE_MODEL = 2
    MISTRAL_LANGUAGE_MODEL = 3
    MODERNBERT_EMBEDDINGS_MODEL = 4
    SENTENCE_TRANSFORMER_EMBEDDINGS_MODEL = 5

    def __str__(self) -> str:
        """Return the name of the model."""
        return self.name

    def __repr__(self) -> str:
        """Return the name of the model."""
        return self.name


class _ModelFactory:
    """
    Internal factory for creating model instances.

    Maps model names to their constructor functions
    for instantiation.
    """

    def __init__(self):
        """Initialize the factory with available model mappings."""
        self._models: Dict[ModelNames, Callable[[], ModelWrapper]] = {
            ModelNames.QWEN3_LANGUAGE_MODEL: Qwen3LanguageModel,
            ModelNames.LLAMA32_LANGUAGE_MODEL: Llama32LanguageModel,
            ModelNames.MISTRAL_LANGUAGE_MODEL: MistralLanguageModel,
            ModelNames.MODERNBERT_EMBEDDINGS_MODEL: ModernBertEmbeddingsModel,
            ModelNames.SENTENCE_TRANSFORMER_EMBEDDINGS_MODEL: SentenceTransformerEmbeddingsModel,
        }

    def get_model(self, name: ModelNames) -> ModelWrapper:
        """
        Create a new model instance.

        Args:
            name: The type of model to create

        Returns:
            ModelWrapper: A new model instance
        """
        return self._models[name]()


@dataclass
class _Entry:
    """
    Internal entry for tracking model state.

    Contains the model instance and metadata about its usage
    including last used time, inflight count, and access lock.
    """

    model: ModelWrapper
    last_used: float = field(default_factory=time.time)
    inflight: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class ModelManager:
    """
    Manager for model providers with lifecycle management.

    Provides automatic loading/unloading of models based on usage
    patterns, with thread-safe access and resource cleanup.
    """

    _instance: Optional["ModelManager"] = None

    @staticmethod
    def get_instance():
        """
        Get the singleton instance of ModelManager.

        Returns:
            ModelManager: The global model manager instance
        """
        if ModelManager._instance is None:
            ModelManager._instance = ModelManager()
        return ModelManager._instance

    def __init__(self, idle_ttl: float = 600):
        """
        Initialize the ModelManager.

        Args:
            idle_ttl: Time in seconds before unloading idle models (default: 600)
        """
        self._entries: Dict[ModelNames, _Entry] = {}
        self.idle_ttl = idle_ttl
        self._global_lock = asyncio.Lock()
        self._shutdown = asyncio.Event()
        self._reaper_task: Optional[asyncio.Task] = None
        self._model_factory = _ModelFactory()

    async def start(self):
        """
        Start the model manager and background reaper task.
        """
        self._reaper_task = asyncio.create_task(self._reaper(), name="ModelReaper")

    async def stop(self):
        """
        Stop the model manager and cleanup all models.
        """
        self._shutdown.set()
        if self._reaper_task:
            self._reaper_task.cancel()
            try:
                await self._reaper_task
            except asyncio.CancelledError:
                pass
        # optional: unload everything on shutdown
        for name in list(self._entries.keys()):
            entry = self._entries.get(name)
            if not entry:
                continue
            async with entry.lock:
                try:
                    await entry.model.unload()
                finally:
                    self._entries.pop(name, None)

    @asynccontextmanager
    async def acquire(self, name: ModelNames):
        """
        Acquire a model for use with automatic lifecycle management.

        Args:
            name: The type of model to acquire

        Yields:
            ModelWrapper: The loaded model instance
        """
        # Get or create the entry (prevent duplicate creation with a global lock)
        entry = self._entries.get(name)
        if entry is None:
            async with self._global_lock:
                entry = self._entries.get(name)
                if entry is None:
                    entry = _Entry(model=self._model_factory.get_model(name))
                    self._entries[name] = entry

        # Acquire (load + mark inflight)
        async with entry.lock:
            await entry.model.ensure_loaded()
            entry.inflight += 1
            entry.last_used = time.time()

        try:
            yield entry.model
        finally:
            # Release
            async with entry.lock:
                entry.inflight = max(0, entry.inflight - 1)
                entry.last_used = time.time()

    async def touch(self, name: ModelNames):
        """
        Update the last used time for a model without changing inflight count.

        Args:
            name: The type of model to touch
        """
        entry = self._entries.get(name)
        if not entry:
            return
        async with entry.lock:
            entry.last_used = time.time()

    async def _reaper(self):
        """
        Background task that unloads idle models.

        Periodically checks for models that have been idle for
        longer than the TTL and unloads them to free memory.
        """
        try:
            while not self._shutdown.is_set():
                now = time.time()
                to_unload: list[ModelNames] = []

                # Snapshot names to avoid long-held locks
                for name in list(self._entries.keys()):
                    entry = self._entries.get(name)
                    if entry is None:
                        continue
                    async with entry.lock:
                        idle_for = now - entry.last_used
                        if entry.inflight == 0 and idle_for >= self.idle_ttl:
                            to_unload.append(name)

                # Unload outside first pass to minimize contention
                for name in to_unload:
                    entry = self._entries.get(name)
                    if entry is None:
                        continue
                    async with entry.lock:
                        # Double-check in case someone started using it
                        if (
                            entry.inflight == 0
                            and (time.time() - entry.last_used) >= self.idle_ttl
                        ):
                            await entry.model.unload()
                            self._entries.pop(name, None)

                try:
                    await asyncio.wait_for(self._shutdown.wait(), timeout=10)
                except asyncio.TimeoutError:
                    pass  # normal tick; loop
        except asyncio.CancelledError:
            pass  # shutting down
