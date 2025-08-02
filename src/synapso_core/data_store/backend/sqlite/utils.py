import logging
import os
import sqlite3

from sqlalchemy import Engine, engine
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

logger = logging.getLogger(__name__)


def create_sqlite_db_if_not_exists(db_path):
    """Create SQLite database file if it doesn't exist."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        if not os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            conn.close()
            logger.info(f"Database created at: {db_path}")
        else:
            logger.debug(f"Database already exists at: {db_path}")
    except Exception as e:
        logger.error(f"Failed to create database at {db_path}: {e}")
        raise


class SqliteEngineMixin:
    def __init__(self, db_path: str):
        self._db_endpoint = f"sqlite:///{db_path}"
        self._async_db_endpoint = f"sqlite+aiosqlite:///{db_path}"

    def get_sync_engine(self) -> Engine:
        return engine.create_engine(self._db_endpoint)

    def get_async_engine(self) -> AsyncEngine:
        return create_async_engine(self._async_db_endpoint)
