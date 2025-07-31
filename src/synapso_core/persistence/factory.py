from .implementations.meta_store import MetaSqliteAdapter
from .implementations.chunk_store import ChunkSqliteAdapter
from .implementations.vector_store import VectorSqliteAdapter
from .interfaces import MetaStore, PrivateChunkStore, VectorStore



class MetaStoreFactory:
    @staticmethod
    def get_meta_store(meta_store_type: str) -> MetaStore:
        """
        Read the global config and prepare the meta store.
        """
        available_meta_stores = {"sqlite": MetaSqliteAdapter}
        if meta_store_type not in available_meta_stores:
            raise ValueError(f"Meta store type {meta_store_type} not found")
        return available_meta_stores[meta_store_type]()


class PrivateStoreFactory:
    @staticmethod
    def get_private_store(private_store_type: str) -> PrivateChunkStore:
        """
        Read the global config and prepare the private store.
        """
        available_private_stores = {"sqlite": ChunkSqliteAdapter}
        if private_store_type not in available_private_stores:
            raise ValueError(f"Private store type {private_store_type} not found")
        return available_private_stores[private_store_type]()


class VectorStoreFactory:
    @staticmethod
    def get_vector_store(vector_store_type: str) -> VectorStore:
        """
        Read the global config and prepare the vector store.
        """
        available_vector_stores = {"sqlite": VectorSqliteAdapter}
        if vector_store_type not in available_vector_stores:
            raise ValueError(f"Vector store type {vector_store_type} not found")
        return available_vector_stores[vector_store_type]()
