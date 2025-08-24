"""
Database models for Synapso Core.

This module defines SQLAlchemy ORM models for the various data stores
including metadata, vector storage, and private chunk storage.
"""

from datetime import datetime

from sqlalchemy import Index, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import Boolean, Integer, String


class MetaStoreBase(DeclarativeBase):
    """Base class for metadata store models."""

    pass


class VectorStoreBase(DeclarativeBase):
    """Base class for vector store models."""

    pass


class PrivateChunkStoreBase(DeclarativeBase):
    """Base class for private chunk store models."""

    pass


class DBCortex(MetaStoreBase):
    """
    Database model representing a document collection (cortex).

    A cortex is a collection of documents stored in a specific directory
    that can be indexed and searched together.
    """

    __tablename__ = "cortex"

    cortex_id: Mapped[str] = mapped_column(primary_key=True)
    cortex_name: Mapped[str] = mapped_column(String, nullable=False)
    path: Mapped[str] = mapped_column(String(length=1024), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        default=func.now(),  # pylint: disable=not-callable
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=func.now(),  # pylint: disable=not-callable
        server_onupdate=func.now(),  # pylint: disable=not-callable
        nullable=False,
    )
    last_indexed_at: Mapped[datetime] = mapped_column(nullable=True)


class DBPrivateChunk(PrivateChunkStoreBase):
    """
    Database model representing a private document chunk.

    Stores the actual text content of document chunks with
    content-based hashing for deduplication.
    """

    __tablename__ = "chunks"
    content_hash: Mapped[str] = mapped_column(String, primary_key=True)
    chunk_content: Mapped[str] = mapped_column(String, nullable=False)


class DBFile(MetaStoreBase):
    """
    Database model representing a file within a cortex.

    Tracks file metadata including path, size, type, and indexing status.
    """

    __tablename__ = "files"
    cortex_id: Mapped[str] = mapped_column(String, nullable=False)
    file_id: Mapped[str] = mapped_column(String, primary_key=True)
    file_name: Mapped[str] = mapped_column(String, nullable=False)
    file_path: Mapped[str] = mapped_column(String, nullable=False, index=True)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    file_type: Mapped[str] = mapped_column(String, nullable=False)
    file_created_at: Mapped[datetime] = mapped_column(nullable=False)
    file_updated_at: Mapped[datetime] = mapped_column(nullable=False)
    eligibility_status: Mapped[str] = mapped_column(String, nullable=False)
    last_indexed_at: Mapped[datetime] = mapped_column(nullable=True)
    __table_args__ = (
        UniqueConstraint("cortex_id", "file_path", name="uq_files_cortex_path"),
    )


class DBFileVersion(MetaStoreBase):
    """
    Database model representing a version of a file.

    Tracks file versions to support incremental updates and
    maintain document history.
    """

    __tablename__ = "file_versions"
    cortex_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    file_version_id: Mapped[str] = mapped_column(String, primary_key=True)
    file_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    file_version_created_at: Mapped[datetime] = mapped_column(nullable=False)
    file_version_invalid_at: Mapped[datetime] = mapped_column(nullable=True)
    file_version_is_valid: Mapped[bool] = mapped_column(
        Boolean, nullable=False, index=True
    )


class FileVersionToChunkId(MetaStoreBase):
    """
    Junction table linking file versions to chunk IDs.

    Maintains the relationship between file versions and their
    associated document chunks with ordering information.
    """

    __tablename__ = "file_version_to_chunk_id"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cortex_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    file_version_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    chunk_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    __table_args__ = (
        Index("idx_fvtci_chunk_id", "chunk_id"),
        Index("idx_fvtci_file_version_id", "file_version_id"),
        UniqueConstraint(
            "cortex_id",
            "file_version_id",
            "chunk_id",
            name="uq_fvtci_cortex_file_version_chunk",
        ),
    )


class Event(MetaStoreBase):
    """
    Database model representing system events.

    Tracks various system events for auditing and monitoring purposes.
    """

    __tablename__ = "events"
    event_id: Mapped[str] = mapped_column(String, primary_key=True)
    cortex_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    event_timestamp: Mapped[datetime] = mapped_column(nullable=False, index=True)


class IndexingJob(MetaStoreBase):
    """
    Database model representing background indexing jobs.

    Tracks the progress and status of document indexing operations
    including file counts and timing information.
    """

    __tablename__ = "background_jobs"
    job_id: Mapped[str] = mapped_column(String, primary_key=True)
    cortex_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    job_type: Mapped[str] = mapped_column(String, nullable=False)
    job_status: Mapped[str] = mapped_column(String, nullable=False, index=True)
    n_total_files: Mapped[int] = mapped_column(Integer, nullable=False)
    n_eligible_files: Mapped[int] = mapped_column(Integer, nullable=False)
    files_processed: Mapped[int] = mapped_column(Integer, nullable=False)
    job_start_time: Mapped[datetime] = mapped_column(nullable=False, index=True)
    job_end_time: Mapped[datetime] = mapped_column(nullable=True)
