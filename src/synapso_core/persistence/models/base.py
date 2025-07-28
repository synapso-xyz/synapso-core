from sqlalchemy.orm import DeclarativeBase


class MetaStoreBase(DeclarativeBase):
    pass


class VectorStoreBase(DeclarativeBase):
    pass


class PrivateChunkStoreBase(DeclarativeBase):
    pass
