import uuid
from datetime import datetime
from pathlib import Path
from typing import List

from sqlalchemy import select
from sqlalchemy.orm import Session

from ....config_manager import get_config
from ...data_models import (
    DBCortex,
    DBFile,
    DBFileVersion,
    Event,
    FileVersionToChunkId,
    IndexingJob,
    MetaStoreBase,
)
from ...interfaces import MetaStore
from .sqlite_backend_identifier import SqliteBackendIdentifierMixin
from .utils import SqliteEngineMixin


class SqliteMetaStore(SqliteEngineMixin, SqliteBackendIdentifierMixin, MetaStore):
    def __init__(self):
        config = get_config()
        if config.meta_store.meta_db_type != "sqlite":
            raise ValueError("Meta store type is not sqlite")
        self.meta_db_path = config.meta_store.meta_db_path
        self.meta_db_path = str(Path(self.meta_db_path).expanduser().resolve())
        SqliteEngineMixin.__init__(self, self.meta_db_path)
        SqliteBackendIdentifierMixin.__init__(self, backend_identifier="sqlite")

    def create_cortex(self, cortex_name: str, cortex_path: str) -> DBCortex:
        """
        Create a new cortex.
        """
        cortex_id = uuid.uuid4().hex
        cortex_path = str(Path(cortex_path).expanduser().resolve())
        cortex = DBCortex(
            cortex_id=cortex_id,
            cortex_name=cortex_name,
            path=cortex_path,
        )
        with Session(self.get_sync_engine()) as session:
            session.add(cortex)
            session.commit()
            session.refresh(cortex)
            return cortex

    def get_cortex_by_name(self, cortex_name: str) -> DBCortex | None:
        """
        Get a cortex by its name.
        """
        with Session(self.get_sync_engine()) as session:
            stmt = select(DBCortex).where(DBCortex.cortex_name == cortex_name)
            result = session.execute(stmt).scalar_one_or_none()
            return result

    def get_cortex_by_id(self, cortex_id: str) -> DBCortex | None:
        """
        Get a cortex by its id.
        """
        with Session(self.get_sync_engine()) as session:
            stmt = select(DBCortex).where(DBCortex.cortex_id == cortex_id)
            result = session.execute(stmt).scalar_one_or_none()
            return result

    def list_cortices(self) -> List[DBCortex]:
        """
        List all cortices.
        """
        with Session(self.get_sync_engine()) as session:
            stmt = select(DBCortex)
            result = session.execute(stmt).scalars().all()
            return list(result)

    def update_cortex(self, updated_cortex: DBCortex) -> DBCortex | None:
        """
        Update a cortex.
        """
        with Session(self.get_sync_engine()) as session:
            session.add(updated_cortex)
            session.commit()
            session.refresh(updated_cortex)
            return updated_cortex

    def create_file(self, file: DBFile) -> DBFile:
        """
        Create a new file.
        """
        existing_file = self.get_file_by_path(file.file_path)
        if existing_file:
            return existing_file

        with Session(self.get_sync_engine()) as session:
            session.add(file)
            session.commit()
            session.refresh(file)
            return file

    def get_file_by_id(self, file_id: str) -> DBFile | None:
        """
        Get a file by its id.
        """
        with Session(self.get_sync_engine()) as session:
            stmt = select(DBFile).where(DBFile.file_id == file_id)
            result = session.execute(stmt).scalar_one_or_none()
            return result

    def update_file(self, updated_file: DBFile) -> DBFile | None:
        """
        Update a file.
        """
        with Session(self.get_sync_engine()) as session:
            session.add(updated_file)
            session.commit()
            session.refresh(updated_file)
            return updated_file

    def get_file_by_path(self, file_path: str) -> DBFile | None:
        """
        Get a file by its path.
        """
        with Session(self.get_sync_engine()) as session:
            stmt = select(DBFile).where(DBFile.file_path == file_path)
            result = session.execute(stmt).scalar_one_or_none()
            return result

    def create_file_version(self, file_version: DBFileVersion) -> DBFileVersion:
        """
        Create a new file version.
        """
        with Session(self.get_sync_engine()) as session:
            session.add(file_version)
            session.commit()
            session.refresh(file_version)
            return file_version

    def get_file_version_by_id(self, file_version_id: str) -> DBFileVersion | None:
        """
        Get a file version by its id.
        """
        with Session(self.get_sync_engine()) as session:
            stmt = select(DBFileVersion).where(
                DBFileVersion.file_version_id == file_version_id
            )
            result = session.execute(stmt).scalar_one_or_none()
            return result

    def assosiate_chunks(self, file_version_id: str, chunk_ids: List[str]) -> bool:
        """
        Associate chunks with a file version.
        """
        file_version_to_chunk_ids = []
        file_version = self.get_file_version_by_id(file_version_id)
        if not file_version:
            raise ValueError(f"File version {file_version_id} not found")
        cortex_id = file_version.cortex_id

        for idx, chunk_id in enumerate(chunk_ids):
            file_version_to_chunk_id = FileVersionToChunkId(
                file_version_id=file_version_id,
                chunk_id=chunk_id,
                cortex_id=cortex_id,
                chunk_index=idx,
            )
            file_version_to_chunk_ids.append(file_version_to_chunk_id)

        with Session(self.get_sync_engine()) as session:
            session.add_all(file_version_to_chunk_ids)
            session.commit()
        return True

    def get_file_version_by_chunk_id(self, chunk_id: str) -> DBFileVersion | None:
        """
        Get a file version by its chunk id.
        """
        with Session(self.get_sync_engine()) as session:
            stmt = (
                select(DBFileVersion)
                .join(FileVersionToChunkId)
                .where(FileVersionToChunkId.chunk_id == chunk_id)
            )
            result = session.execute(stmt).scalar_one_or_none()
            return result

    def create_event(self, event: Event) -> Event:
        """
        Create a new event.
        """
        with Session(self.get_sync_engine()) as session:
            session.add(event)
            session.commit()
            session.refresh(event)
            return event

    def get_events(
        self, cortex_id: str, start_time: datetime, end_time: datetime
    ) -> List[Event]:
        """
        Get events for a given cortex id and time range.
        """
        with Session(self.get_sync_engine()) as session:
            stmt = select(Event).where(
                Event.cortex_id == cortex_id,
                Event.event_timestamp >= start_time,
                Event.event_timestamp <= end_time,
            )
            result = session.execute(stmt).scalars().all()
            return list(result)

    def create_indexing_job(self, indexing_job: IndexingJob) -> IndexingJob:
        """
        Create a new indexing job.
        """
        with Session(self.get_sync_engine()) as session:
            session.add(indexing_job)
            session.commit()
            session.refresh(indexing_job)
            return indexing_job

    def get_indexing_job_by_id(self, job_id: str) -> IndexingJob | None:
        """
        Get an indexing job by its id.
        """
        with Session(self.get_sync_engine()) as session:
            stmt = select(IndexingJob).where(IndexingJob.job_id == job_id)
            result = session.execute(stmt).scalar_one_or_none()
            return result

    def update_indexing_job(self, indexing_job: IndexingJob) -> IndexingJob:
        """
        Update an indexing job.
        """
        with Session(self.get_sync_engine()) as session:
            session.add(indexing_job)
            session.commit()
            session.refresh(indexing_job)
            return indexing_job

    def list_indexing_jobs(self, status: str = "IN_PROGRESS") -> List[IndexingJob]:
        with Session(self.get_sync_engine()) as session:
            stmt = select(IndexingJob).where(IndexingJob.job_status == status)
            result = session.execute(stmt).scalars().all()
            return list(result)

    def setup(self) -> bool:
        MetaStoreBase.metadata.create_all(self.get_sync_engine())
        return True

    def teardown(self) -> bool:
        self.close()
        return True
