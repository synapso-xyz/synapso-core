from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple

from sqlalchemy import Engine
from sqlalchemy.ext.asyncio import AsyncEngine

from ..models import Vector, VectorMetadata
from .data_models import DBCortex


class BaseDataStore(ABC):
    supports_async: bool = False

    @abstractmethod
    def get_sync_engine(self) -> Engine:
        pass


class AsyncDataStore(BaseDataStore):
    supports_async: bool = True

    @abstractmethod
    def get_async_engine(self) -> AsyncEngine:
        pass


class SetupTearDownMixin:
    def setup(self) -> bool:
        raise NotImplementedError("setup not implemented")

    def teardown(self) -> bool:
        raise NotImplementedError("teardown not implemented")


class MetaStore(SetupTearDownMixin, AsyncDataStore):
    """
    A store for metadata.
    """

    data_store_type: str = "meta_store"

    def create_cortex(self, cortex_name: str) -> DBCortex:
        """
        Create a new cortex.
        """
        pass

    def get_cortex_by_name(self, cortex_name: str) -> DBCortex | None:
        """
        Get a cortex by its name.
        """
        pass

    def get_cortex_by_id(self, cortex_id: str) -> DBCortex | None:
        """
        Get a cortex by its id.
        """
        pass

    def list_cortices(self) -> List[DBCortex]:
        """
        List all cortices.
        """
        pass

    def update_cortex(self, updated_cortex: DBCortex) -> DBCortex | None:
        """
        Update a cortex. Returns the updated cortex.
        """
        pass


class PrivateChunkStore(SetupTearDownMixin, AsyncDataStore):
    """
    A store for private chunks.
    This would be a simple key-value store that is stored on device.
    The key / chunk_id would be a content hash of the chunk.
    The value would be the private chunk.
    """

    data_store_type: str = "private_store"

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


class VectorStore(SetupTearDownMixin, BaseDataStore):
    """
    A store for vectors.
    """

    data_store_type: str = "vector_store"

    @abstractmethod
    def insert(self, vector: Vector) -> bool:
        """
        Insert a new vector with its associated metadata into the store.
        """
        pass

    @abstractmethod
    def get_by_id(self, vector_id: str) -> Vector | None:
        """
        Retrieve a vector and its metadata by document ID.
        """
        pass

    @abstractmethod
    def vector_search(
        self, query_vector: Vector, top_k: int = 5, filters: Dict | None = None
    ) -> List[Tuple[Vector, float]]:
        """
        Search for vectors most similar to the query_vector. Optionally filter by metadata.
        Returns a list of tuples of (vector, score).
        """
        pass

    @abstractmethod
    def delete(self, vector_id: str) -> bool:
        """
        Delete a vector and its metadata by document ID.
        """
        pass

    @abstractmethod
    def update_metadata(self, vector_id: str, metadata: VectorMetadata) -> bool:
        """
        Update the metadata for a given document ID.
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """
        Return the total number of vectors in the store.
        """
        pass


@dataclass
class BaseBackendIdentifierMixin:
    backend_identifier: str
