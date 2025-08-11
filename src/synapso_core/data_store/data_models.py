from datetime import datetime

from sqlalchemy import func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import Boolean, Integer, String


class MetaStoreBase(DeclarativeBase):
    pass


class VectorStoreBase(DeclarativeBase):
    pass


class PrivateChunkStoreBase(DeclarativeBase):
    pass


class DBCortex(MetaStoreBase):
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
    __tablename__ = "chunks"
    content_hash: Mapped[str] = mapped_column(String, primary_key=True)
    chunk_content: Mapped[str] = mapped_column(String, nullable=False)


class DBFile(MetaStoreBase):
    __tablename__ = "files"
    cortex_id: Mapped[str] = mapped_column(String, nullable=False)
    file_id: Mapped[str] = mapped_column(String, primary_key=True)
    file_name: Mapped[str] = mapped_column(String, nullable=False)
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    file_type: Mapped[str] = mapped_column(String, nullable=False)
    file_created_at: Mapped[datetime] = mapped_column(nullable=False)
    file_updated_at: Mapped[datetime] = mapped_column(nullable=False)
    eligibility_status: Mapped[str] = mapped_column(String, nullable=False)
    last_indexed_at: Mapped[datetime] = mapped_column(nullable=True)


class DBFileVersion(MetaStoreBase):
    __tablename__ = "file_versions"
    cortex_id: Mapped[str] = mapped_column(String, nullable=False)
    file_version_id: Mapped[str] = mapped_column(String, primary_key=True)
    file_id: Mapped[str] = mapped_column(String, nullable=False)
    file_version_created_at: Mapped[datetime] = mapped_column(nullable=False)
    file_version_invalid_at: Mapped[datetime] = mapped_column(nullable=True)
    file_version_is_valid: Mapped[bool] = mapped_column(Boolean, nullable=False)


class FileVersionToChunkId(MetaStoreBase):
    __tablename__ = "file_version_to_chunk_id"
    cortex_id: Mapped[str] = mapped_column(String, nullable=False)
    file_version_id: Mapped[str] = mapped_column(String, nullable=False)
    chunk_id: Mapped[str] = mapped_column(String, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)


class Event(MetaStoreBase):
    __tablename__ = "events"
    event_id: Mapped[str] = mapped_column(String, primary_key=True)
    cortex_id: Mapped[str] = mapped_column(String, nullable=False)
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    event_timestamp: Mapped[datetime] = mapped_column(nullable=False)


class IndexingJob(MetaStoreBase):
    __tablename__ = "background_jobs"
    job_id: Mapped[str] = mapped_column(String, primary_key=True)
    cortex_id: Mapped[str] = mapped_column(String, nullable=False)
    job_type: Mapped[str] = mapped_column(String, nullable=False)
    job_status: Mapped[str] = mapped_column(String, nullable=False)
    n_total_files: Mapped[int] = mapped_column(Integer, nullable=False)
    n_eligible_files: Mapped[int] = mapped_column(Integer, nullable=False)
    files_processed: Mapped[int] = mapped_column(Integer, nullable=False)
    job_start_time: Mapped[datetime] = mapped_column(nullable=False)
    job_end_time: Mapped[datetime] = mapped_column(nullable=True)
