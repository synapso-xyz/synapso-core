from datetime import datetime

from sqlalchemy import func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import String


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
    created_at: Mapped[datetime] = mapped_column(default=func.now(), nullable=False)  # pylint: disable=not-callable
    updated_at: Mapped[datetime] = mapped_column(
        default=func.now(),  # pylint: disable=not-callable
        onupdate=func.now(),  # pylint: disable=not-callable
        nullable=False,
    )
    last_indexed_at: Mapped[datetime] = mapped_column(nullable=True)


class DBPrivateChunk(PrivateChunkStoreBase):
    __tablename__ = "chunks"
    content_hash: Mapped[str] = mapped_column(String, primary_key=True)
    chunk_content: Mapped[str] = mapped_column(String, nullable=False)
