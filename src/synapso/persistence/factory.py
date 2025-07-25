from .interfaces import MetaStore, PrivateStore, VectorStore


class MetaStoreFactory:
    @staticmethod
    def get_meta_store() -> MetaStore | None:
        """
        Read the global config and prepare the meta store.
        """
        pass


class PrivateStoreFactory:
    @staticmethod
    def get_private_store() -> PrivateStore | None:
        """
        Read the global config and prepare the private store.
        """
        pass


class VectorStoreFactory:
    @staticmethod
    def get_vector_store() -> VectorStore | None:
        """
        Read the global config and prepare the vector store.
        """
        pass
