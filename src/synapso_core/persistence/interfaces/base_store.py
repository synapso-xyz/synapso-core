from abc import ABC, abstractmethod

from sqlalchemy import Engine
from sqlalchemy.ext.asyncio import AsyncEngine


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
