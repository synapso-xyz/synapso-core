from .interfaces import MetaStore, PrivateStore, VectorStore

class MetaStoreFactory:
    @staticmethod
    def get_meta_store() -> MetaStore:
        """
        Read the global config and prepare the meta store.
        """
        pass

class PrivateStoreFactory:
    @staticmethod
    def get_private_store() -> PrivateStore:
        """
        Read the global config and prepare the private store.
        """
        pass

class VectorStoreFactory:
    @staticmethod
    def get_vector_store() -> VectorStore:
        """
        Read the global config and prepare the vector store.
        """
        pass