from datetime import datetime

from sqlalchemy import func
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import String

from .base import Base


class Cortex(Base):
    __tablename__ = "cortex"

    cortex_id: Mapped[str] = mapped_column(primary_key=True)
    cortex_name: Mapped[str] = mapped_column(String, nullable=False)
    path: Mapped[str] = mapped_column(String(length=1024), nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        default=func.now(),
        server_onupdate=func.now(),
        nullable=False,
    )
    last_indexed_at: Mapped[datetime] = mapped_column(nullable=True)
