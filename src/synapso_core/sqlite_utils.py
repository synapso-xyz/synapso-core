import os
import sqlite3

from sqlalchemy import Engine, engine
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


def create_sqlite_db_if_not_exists(db_path):
    # Check if the database file already exists
    if not os.path.exists(db_path):
        # Connect to the database (this will create the file)
        conn = sqlite3.connect(db_path)
        conn.close()
        print(f"Database created at: {db_path}")
    else:
        print(f"Database already exists at: {db_path}")


class SqliteEngineMixin:
    def __init__(self, db_path: str):
        self._db_endpoint = f"sqlite:///{db_path}"
        self._async_db_endpoint = f"sqlite+aiosqlite:///{db_path}"

    def get_sync_engine(self) -> Engine:
        return engine.create_engine(self._db_endpoint)

    def get_async_engine(self) -> AsyncEngine:
        return create_async_engine(self._async_db_endpoint)
