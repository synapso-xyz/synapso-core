from abc import abstractmethod

from .base_store import AsyncDataStore


class MetaStore(AsyncDataStore):
    """
    A store for metadata.
    """

    @abstractmethod
    def metastore_setup(self) -> bool:
        """
        Setup the meta store.
        """
        pass

    @abstractmethod
    def metastore_teardown(self) -> bool:
        """
        Teardown the meta store.
        """
        pass
