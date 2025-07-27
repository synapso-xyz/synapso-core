from abc import abstractmethod

from .base_store import AsyncDataStore


class PrivateChunkStore(AsyncDataStore):
    """
    A store for private chunks.
    This would be a simple key-value store that is stored on device.
    The key / chunk_id would be a content hash of the chunk.
    The value would be the private chunk.
    """

    @abstractmethod
    def setup(self) -> bool:
        """
        Setup the private store.
        """
        pass

    @abstractmethod
    def get_by_chunk_id(self, chunk_id: str) -> str:
        """
        Get a private chunk by its chunk_id.
        """
        pass

    @abstractmethod
    def insert(self, chunk_contents: str) -> str:
        """
        Insert a private chunk into the store.
        """
        pass

    @abstractmethod
    def delete(self, chunk_id: str) -> None:
        """
        Delete a private chunk from the store.
        """
        pass

    @abstractmethod
    def teardown(self) -> bool:
        """
        Teardown the private store.
        """
        pass
