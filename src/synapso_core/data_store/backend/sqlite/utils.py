import logging
import sqlite3
from pathlib import Path

from sqlalchemy import Engine, engine
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

logger = logging.getLogger(__name__)


def create_sqlite_db_if_not_exists(db_path):
    """Create SQLite database file if it doesn't exist."""
    try:
        # Ensure the directory exists
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        if not db_path.exists():
            conn = sqlite3.connect(db_path)
            conn.close()
            logger.info("Database created at: %s", db_path)
        else:
            logger.debug("Database already exists at: %s", db_path)
    except Exception as e:
        logger.error("Failed to create database at %s: %s", db_path, e)
        raise


class SqliteEngineMixin:
    def __init__(self, db_path: str):
        self._db_endpoint = f"sqlite:///{db_path}"
        self._async_db_endpoint = f"sqlite+aiosqlite:///{db_path}"
        self._sync_engine = None
        self._async_engine = None

    def get_sync_engine(self) -> Engine:
        if self._sync_engine is None:
            self._sync_engine = engine.create_engine(self._db_endpoint)
        return self._sync_engine

    def get_async_engine(self) -> AsyncEngine:
        if self._async_engine is None:
            self._async_engine = create_async_engine(self._async_db_endpoint)
        return self._async_engine

    def close(self):
        """Explicitly close the database connections."""
        if self._sync_engine is not None:
            self._sync_engine.dispose()
            self._sync_engine = None

        if self._async_engine is not None:
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._async_engine.dispose())
            except RuntimeError:
                logger.warning("No running event loop found")
                asyncio.run(self._async_engine.dispose())
            except Exception as e:
                logger.error("Error disposing async engine: %s", e)
                raise e
            finally:
                self._async_engine = None
