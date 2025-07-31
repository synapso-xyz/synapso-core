from .base import PrivateChunkStoreBase


class PrivateChunk(PrivateChunkStoreBase):
    __tablename__ = "chunks"
    content_hash: Mapped[str] = mapped_column(String, primary_key=True)
    chunk_content: Mapped[str] = mapped_column(String, nullable=False)
    
    