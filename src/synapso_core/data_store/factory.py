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
    A factory for creating data stores.
    """

    @staticmethod
    def get_meta_store(backend_identifier: str) -> MetaStore:
        return DataStoreFactory._get_backend(backend_identifier, "meta_store")

    @staticmethod
    def get_private_store(backend_identifier: str) -> PrivateChunkStore:
        return DataStoreFactory._get_backend(backend_identifier, "private_store")

    @staticmethod
    def get_vector_store(backend_identifier: str) -> VectorStore:
        return DataStoreFactory._get_backend(backend_identifier, "vector_store")

    @staticmethod
    def _get_backend(backend_identifier: str, data_store_type: str) -> BaseDataStore:
        if data_store_type not in DATA_STORE_BACKENDS:
            raise ValueError(
                f"Invalid data store type: {data_store_type}. Available data store types: {DATA_STORE_BACKENDS.keys()}"
            )
        if backend_identifier not in DATA_STORE_BACKENDS[data_store_type]:
            raise ValueError(
                f"Invalid backend identifier for {data_store_type}: {backend_identifier}. Available backends: {DATA_STORE_BACKENDS[data_store_type].keys()}"
            )
        return DATA_STORE_BACKENDS[data_store_type][backend_identifier]()
