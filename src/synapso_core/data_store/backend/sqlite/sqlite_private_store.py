import logging
from pathlib import Path

from sqlalchemy.orm import Session

from ....config_manager import get_config
from ....utils import get_content_hash
from ...data_models import DBPrivateChunk, PrivateChunkStoreBase
from ...interfaces import PrivateChunkStore
from .sqlite_backend_identifier import SqliteBackendIdentifierMixin
from .utils import SqliteEngineMixin, create_sqlite_db_if_not_exists

logger = logging.getLogger(__name__)


class SqlitePrivateStore(
    SqliteEngineMixin, PrivateChunkStore, SqliteBackendIdentifierMixin
):
    def __init__(self):
        config = get_config()
        if config.private_store.private_db_type != "sqlite":
            raise ValueError("Chunk store type is not sqlite")
        self.chunk_db_path = config.private_store.private_db_path
        self.chunk_db_path = str(Path(self.chunk_db_path).expanduser().resolve())
        logger.info("Chunk DB path: %s", self.chunk_db_path)
        SqliteEngineMixin.__init__(self, self.chunk_db_path)

    def close(self):
        """Explicitly close the database connection."""
        # SQLAlchemy engines should be disposed to close all connections
        if hasattr(self, "_sync_engine"):
            self._sync_engine.dispose()  # type: ignore
        if hasattr(self, "_async_engine"):
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                if self._async_engine is not None:
                    loop.create_task(self._async_engine.dispose())
            except RuntimeError:
                logger.warning("No running event loop found")
                if self._async_engine is not None:
                    asyncio.run(self._async_engine.dispose())
            except Exception as e:
                logger.error("Error disposing async engine: %s", e)
                raise e

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def setup(self) -> bool:
        create_sqlite_db_if_not_exists(self.chunk_db_path)
        self._setup_tables()
        return True

    def teardown(self) -> bool:
        # Implement any necessary teardown logic here
        self.close()
        return True

    def get_by_chunk_id(self, chunk_id: str) -> str | None:
        """
        Get a private chunk by its chunk_id.
        """
        with Session(self.get_sync_engine()) as session:
            chunk = (
                session.query(DBPrivateChunk)
                .filter(DBPrivateChunk.content_hash == chunk_id)
                .first()
            )
            if chunk:
                return chunk.chunk_content
            return None

    def insert(self, chunk_contents: str) -> str:
        """
        Insert a private chunk into the store.
        """
        content_hash = get_content_hash(chunk_contents)
        existing = self.get_by_chunk_id(content_hash)
        if existing:
            logger.info("Chunk already exists: %s", content_hash)
        else:
            pvt_chunk = DBPrivateChunk(
                content_hash=content_hash, chunk_content=chunk_contents
            )
            with Session(self.get_sync_engine()) as session:
                session.add(pvt_chunk)
                session.commit()
        return content_hash

    def delete(self, chunk_id: str) -> None:
        """
        Delete a private chunk from the store.
        """
        with Session(self.get_sync_engine()) as session:
            session.query(DBPrivateChunk).filter(
                DBPrivateChunk.content_hash == chunk_id
            ).delete()
            session.commit()

    def _setup_tables(self) -> None:
        PrivateChunkStoreBase.metadata.create_all(self.get_sync_engine())


if __name__ == "__main__":
    adapter = SqlitePrivateStore()
    adapter.setup()
    logger.info(adapter.get_sync_engine())
    logger.info(adapter.get_async_engine())
