from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

from sqlalchemy import Engine
from sqlalchemy.ext.asyncio import AsyncEngine

from ..models import Vector, VectorMetadata
from .data_models import DBCortex, DBFile, DBFileVersion, Event, IndexingJob


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


class SetupTearDownMixin(ABC):
    @abstractmethod
    def setup(self) -> bool:
        pass

    @abstractmethod
    def teardown(self) -> bool:
        pass


class MetaStore(SetupTearDownMixin, AsyncDataStore):
    """
    A store for metadata.
    """

    data_store_type: str = "meta_store"

    def create_cortex(self, cortex_name: str, cortex_path: str) -> DBCortex:
        """
        Create a new cortex.
        """
        raise NotImplementedError

    def get_cortex_by_name(self, cortex_name: str) -> DBCortex | None:
        """
        Get a cortex by its name.
        """
        raise NotImplementedError

    def get_cortex_by_id(self, cortex_id: str) -> DBCortex | None:
        """
        Get a cortex by its id.
        """
        raise NotImplementedError

    def list_cortices(self) -> List[DBCortex]:
        """
        List all cortices.
        """
        raise NotImplementedError

    def update_cortex(self, updated_cortex: DBCortex) -> DBCortex | None:
        """
        Update a cortex. Returns the updated cortex.
        """
        raise NotImplementedError

    def create_file(self, file: DBFile) -> DBFile:
        """
        Create a new file.
        """
        raise NotImplementedError

    def get_file_by_id(self, file_id: str) -> DBFile | None:
        """
        Get a file by its id.
        """
        raise NotImplementedError

    def update_file(self, updated_file: DBFile) -> DBFile | None:
        """
        Update a file. Returns the updated file.
        """
        raise NotImplementedError

    def get_file_by_path(self, file_path: str) -> DBFile | None:
        """
        Get a file by its path.
        """
        raise NotImplementedError

    def create_file_version(self, file_version: DBFileVersion) -> DBFileVersion:
        """
        Create a new file version.
        """
        raise NotImplementedError

    def get_file_version_by_id(self, file_version_id: str) -> DBFileVersion | None:
        """
        Get a file version by its id.
        """
        raise NotImplementedError

    def get_file_version_by_file_id(self, file_id: str) -> DBFileVersion | None:
        """
        Get a file version by its file id.
        """
        raise NotImplementedError

    def assosiate_chunks(self, file_version_id: str, chunk_ids: List[str]) -> bool:
        """
        Associate chunks with a file version.
        """
        raise NotImplementedError

    def get_file_version_by_chunk_id(self, chunk_id: str) -> DBFileVersion | None:
        """
        Get a file version by its chunk id.
        """
        raise NotImplementedError

    def create_event(self, event: Event) -> Event:
        """
        Create a new event.
        """
        raise NotImplementedError

    def get_events(
        self, cortex_id: str, start_time: datetime, end_time: datetime
    ) -> List[Event]:
        """
        Get events for a given cortex id and time range.
        """
        raise NotImplementedError

    def create_indexing_job(self, indexing_job: IndexingJob) -> IndexingJob:
        """
        Create a new indexing job.
        """
        raise NotImplementedError

    def get_indexing_job_by_id(self, job_id: str) -> IndexingJob | None:
        """
        Get an indexing job by its id.
        """
        raise NotImplementedError

    def update_indexing_job(self, indexing_job: IndexingJob) -> IndexingJob:
        """
        Update an indexing job.
        """
        raise NotImplementedError

    def list_indexing_jobs(self, status: str = "IN_PROGRESS") -> List[IndexingJob]:
        """
        List all indexing jobs.
        """
        raise NotImplementedError


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
    def insert_all(self, chunks: List[str]) -> List[str]:
        """
        Insert a list of private chunks into the store.
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
    def insert_all(self, vectors: List[Vector]) -> List[str]:
        """
        Insert a list of vectors into the store.
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
