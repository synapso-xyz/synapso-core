"""
Data store factory for Synapso Core.

This module provides a factory for creating different types of data stores
including metadata, private chunk, and vector stores with various backends.
"""

from .backend import SqliteMetaStore, SqlitePrivateStore, SqliteVectorStore
from .interfaces import BaseDataStore, MetaStore, PrivateChunkStore, VectorStore

DATA_STORE_BACKENDS = {
    "meta_store": {
        "sqlite": SqliteMetaStore,
    },
    "private_store": {
        "sqlite": SqlitePrivateStore,
    },
    "vector_store": {
        "sqlite": SqliteVectorStore,
    },
}


class DataStoreFactory:
    """
    Factory for creating data store instances.

    Provides centralized creation of different types of data stores
    with configurable backend implementations.
    """

    @staticmethod
    def get_meta_store(backend_identifier: str) -> MetaStore:
        """
        Get a metadata store instance.

        Args:
            backend_identifier: Type of backend to use (e.g., "sqlite")

        Returns:
            MetaStore: A metadata store instance
        """
        return DataStoreFactory._get_backend(backend_identifier, "meta_store")

    @staticmethod
    def get_private_store(backend_identifier: str) -> PrivateChunkStore:
        """
        Get a private chunk store instance.

        Args:
            backend_identifier: Type of backend to use (e.g., "sqlite")

        Returns:
            PrivateChunkStore: A private chunk store instance
        """
        return DataStoreFactory._get_backend(backend_identifier, "private_store")

    @staticmethod
    def get_vector_store(backend_identifier: str) -> VectorStore:
        """
        Get a vector store instance.

        Args:
            backend_identifier: Type of backend to use (e.g., "sqlite")

        Returns:
            VectorStore: A vector store instance
        """
        return DataStoreFactory._get_backend(backend_identifier, "vector_store")

    @staticmethod
    def _get_backend(backend_identifier: str, data_store_type: str) -> BaseDataStore:
        """
        Get a backend instance for the specified data store type.

        Args:
            backend_identifier: Type of backend to use
            data_store_type: Type of data store to create

        Returns:
            BaseDataStore: A data store instance

        Raises:
            ValueError: If the data store type or backend is not supported
        """
        if data_store_type not in DATA_STORE_BACKENDS:
            raise ValueError(
                f"Invalid data store type: {data_store_type}. Available data store types: {DATA_STORE_BACKENDS.keys()}"
            )
        if backend_identifier not in DATA_STORE_BACKENDS[data_store_type]:
            raise ValueError(
                f"Invalid backend identifier for {data_store_type}: {backend_identifier}. Available backends: {DATA_STORE_BACKENDS[data_store_type].keys()}"
            )
        return DATA_STORE_BACKENDS[data_store_type][backend_identifier]()
