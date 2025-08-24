"""
Data store interfaces for Synapso Core.

This module defines the abstract interfaces for different types of data stores
including metadata, private chunk, and vector stores with common functionality.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

from sqlalchemy import Engine
from sqlalchemy.ext.asyncio import AsyncEngine

from ..models import Vector, VectorMetadata
from .data_models import DBCortex, DBFile, DBFileVersion, Event, IndexingJob


class BaseDataStore(ABC):
    """
    Abstract base class for all data stores.

    Provides common functionality and interface requirements
    for different types of data storage backends.
    """

    supports_async: bool = False

    @abstractmethod
    def get_sync_engine(self) -> Engine:
        """
        Get the synchronous SQLAlchemy engine.

        Returns:
            Engine: The database engine for synchronous operations
        """
        pass


class AsyncDataStore(BaseDataStore):
    """
    Abstract base class for asynchronous data stores.

    Extends BaseDataStore with async support capabilities.
    """

    supports_async: bool = True

    @abstractmethod
    def get_async_engine(self) -> AsyncEngine:
        """
        Get the asynchronous SQLAlchemy engine.

        Returns:
            AsyncEngine: The database engine for asynchronous operations
        """
        pass


class SetupTearDownMixin(ABC):
    """
    Mixin providing setup and teardown capabilities.

    Defines the interface for data stores that need
    initialization and cleanup procedures.
    """

    @abstractmethod
    def setup(self) -> bool:
        """
        Initialize the data store.

        Returns:
            bool: True if setup was successful
        """
        pass

    @abstractmethod
    def teardown(self) -> bool:
        """
        Clean up the data store.

        Returns:
            bool: True if teardown was successful
        """
        pass


class MetaStore(SetupTearDownMixin, AsyncDataStore):
    """
    Abstract interface for metadata storage.

    A store for managing metadata about documents, files, cortices,
    and system events. Provides CRUD operations for all metadata entities.
    """

    data_store_type: str = "meta_store"

    def create_cortex(self, cortex_name: str, cortex_path: str) -> DBCortex:
        """
        Create a new cortex (document collection).

        Args:
            cortex_name: Name for the new cortex
            cortex_path: Path to the directory containing documents

        Returns:
            DBCortex: The created cortex record
        """
        raise NotImplementedError

    def get_cortex_by_name(self, cortex_name: str) -> DBCortex | None:
        """
        Get a cortex by its name.

        Args:
            cortex_name: Name of the cortex to retrieve

        Returns:
            DBCortex | None: The cortex record if found, None otherwise
        """
        raise NotImplementedError

    def get_cortex_by_id(self, cortex_id: str) -> DBCortex | None:
        """
        Get a cortex by its ID.

        Args:
            cortex_id: Unique identifier of the cortex

        Returns:
            DBCortex | None: The cortex record if found, None otherwise
        """
        raise NotImplementedError

    def list_cortices(self) -> List[DBCortex]:
        """
        List all available cortices.

        Returns:
            List[DBCortex]: List of all cortex records
        """
        raise NotImplementedError

    def update_cortex(self, updated_cortex: DBCortex) -> DBCortex | None:
        """
        Update a cortex.

        Args:
            updated_cortex: The cortex record with updated values

        Returns:
            DBCortex | None: The updated cortex record if successful, None otherwise
        """
        raise NotImplementedError

    def create_file(self, file: DBFile) -> DBFile:
        """
        Create a new file record.

        Args:
            file: The file record to create

        Returns:
            DBFile: The created file record
        """
        raise NotImplementedError

    def get_file_by_id(self, file_id: str) -> DBFile | None:
        """
        Get a file by its ID.

        Args:
            file_id: Unique identifier of the file

        Returns:
            DBFile | None: The file record if found, None otherwise
        """
        raise NotImplementedError

    def update_file(self, updated_file: DBFile) -> DBFile | None:
        """
        Update a file record.

        Args:
            updated_file: The file record with updated values

        Returns:
            DBFile | None: The updated file record if successful, None otherwise
        """
        raise NotImplementedError

    def get_file_by_path(self, file_path: str) -> DBFile | None:
        """
        Get a file by its path.

        Args:
            file_path: Path to the file

        Returns:
            DBFile | None: The file record if found, None otherwise
        """
        raise NotImplementedError

    def create_file_version(self, file_version: DBFileVersion) -> DBFileVersion:
        """
        Create a new file version.

        Args:
            file_version: The file version record to create

        Returns:
            DBFileVersion: The created file version record
        """
        raise NotImplementedError

    def get_file_version_by_id(self, file_version_id: str) -> DBFileVersion | None:
        """
        Get a file version by its ID.

        Args:
            file_version_id: Unique identifier of the file version

        Returns:
            DBFileVersion | None: The file version record if found, None otherwise
        """
        raise NotImplementedError

    def get_file_version_by_file_id(self, file_id: str) -> DBFileVersion | None:
        """
        Get a file version by its file ID.

        Args:
            file_id: ID of the parent file

        Returns:
            DBFileVersion | None: The file version record if found, None otherwise
        """
        raise NotImplementedError

    def associate_chunks(self, file_version_id: str, chunk_ids: List[str]) -> bool:
        """
        Associate chunks with a file version.

        Args:
            file_version_id: ID of the file version
            chunk_ids: List of chunk IDs to associate

        Returns:
            bool: True if association was successful
        """
        raise NotImplementedError

    def get_file_version_by_chunk_id(self, chunk_id: str) -> DBFileVersion | None:
        """
        Get a file version by its chunk ID.

        Args:
            chunk_id: ID of the chunk

        Returns:
            DBFileVersion | None: The file version record if found, None otherwise
        """
        raise NotImplementedError

    def create_event(self, event: Event) -> Event:
        """
        Create a new event record.

        Args:
            event: The event record to create

        Returns:
            Event: The created event record
        """
        raise NotImplementedError

    def get_events(
        self, cortex_id: str, start_time: datetime, end_time: datetime
    ) -> List[Event]:
        """
        Get events for a given cortex and time range.

        Args:
            cortex_id: ID of the cortex
            start_time: Start of the time range
            end_time: End of the time range

        Returns:
            List[Event]: List of events in the specified range
        """
        raise NotImplementedError

    def create_indexing_job(self, indexing_job: IndexingJob) -> IndexingJob:
        """
        Create a new indexing job.

        Args:
            indexing_job: The indexing job record to create

        Returns:
            IndexingJob: The created indexing job record
        """
        raise NotImplementedError

    def get_indexing_job_by_id(self, job_id: str) -> IndexingJob | None:
        """
        Get an indexing job by its ID.

        Args:
            job_id: Unique identifier of the indexing job

        Returns:
            IndexingJob | None: The indexing job record if found, None otherwise
        """
        raise NotImplementedError

    def update_indexing_job(self, indexing_job: IndexingJob) -> IndexingJob:
        """
        Update an indexing job.

        Args:
            indexing_job: The indexing job record with updated values

        Returns:
            IndexingJob: The updated indexing job record
        """
        raise NotImplementedError

    def list_indexing_jobs(self, status: str = "IN_PROGRESS") -> List[IndexingJob]:
        """
        List indexing jobs with a specific status.

        Args:
            status: Status to filter by (default: "IN_PROGRESS")

        Returns:
            List[IndexingJob]: List of indexing jobs matching the status
        """
        raise NotImplementedError


class PrivateChunkStore(SetupTearDownMixin, AsyncDataStore):
    """
    Abstract interface for private chunk storage.

    A simple key-value store for document chunks stored on device.
    The key (chunk_id) is a content hash of the chunk, and the
    value is the actual chunk content.
    """

    data_store_type: str = "private_store"

    @abstractmethod
    def get_by_chunk_id(self, chunk_id: str) -> str:
        """
        Get a private chunk by its chunk ID.

        Args:
            chunk_id: Content hash identifier of the chunk

        Returns:
            str: The chunk content
        """
        pass

    @abstractmethod
    def insert(self, chunk_contents: str) -> str:
        """
        Insert a private chunk into the store.

        Args:
            chunk_contents: The text content of the chunk

        Returns:
            str: The content hash (chunk ID) of the inserted chunk
        """
        pass

    @abstractmethod
    def delete(self, chunk_id: str) -> None:
        """
        Delete a private chunk from the store.

        Args:
            chunk_id: Content hash identifier of the chunk to delete
        """
        pass


class VectorStore(SetupTearDownMixin, BaseDataStore):
    """
    Abstract interface for vector storage.

    A store for document embeddings (vectors) with metadata,
    supporting similarity search and vector operations.
    """

    data_store_type: str = "vector_store"

    @abstractmethod
    def insert(self, vector: Vector) -> bool:
        """
        Insert a new vector with its associated metadata.

        Args:
            vector: The vector object to insert

        Returns:
            bool: True if insertion was successful
        """
        pass

    @abstractmethod
    def get_by_id(self, vector_id: str) -> Vector | None:
        """
        Retrieve a vector and its metadata by vector ID.

        Args:
            vector_id: Unique identifier of the vector

        Returns:
            Vector | None: The vector object if found, None otherwise
        """
        pass

    @abstractmethod
    def vector_search(
        self, query_vector: Vector, top_k: int = 5, filters: Dict | None = None
    ) -> List[Tuple[Vector, float]]:
        """
        Search for vectors most similar to the query vector.

        Args:
            query_vector: The query vector to search with
            top_k: Maximum number of results to return (default: 5)
            filters: Optional metadata filters to apply

        Returns:
            List[Tuple[Vector, float]]: List of (vector, similarity_score) tuples
        """
        pass

    @abstractmethod
    def delete(self, vector_id: str) -> bool:
        """
        Delete a vector and its metadata by vector ID.

        Args:
            vector_id: Unique identifier of the vector to delete

        Returns:
            bool: True if deletion was successful
        """
        pass

    @abstractmethod
    def update_metadata(self, vector_id: str, metadata: VectorMetadata) -> bool:
        """
        Update the metadata for a given vector ID.

        Args:
            vector_id: Unique identifier of the vector
            metadata: New metadata to apply

        Returns:
            bool: True if update was successful
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """
        Return the total number of vectors in the store.

        Returns:
            int: Total count of stored vectors
        """
        pass


@dataclass
class BaseBackendIdentifierMixin:
    """
    Mixin providing backend identification.

    Adds a backend_identifier field to classes that need
    to identify their storage backend type.
    """

    backend_identifier: str
