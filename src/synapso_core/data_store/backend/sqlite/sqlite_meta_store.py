import uuid
from pathlib import Path
from typing import List

from sqlalchemy import select
from sqlalchemy.orm import Session

from ....config_manager import get_config
from ...data_models import DBCortex, MetaStoreBase
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

    def create_cortex(self, cortex_name: str, cortex_path: str) -> DBCortex:
        """
        Create a new cortex.
        """
        cortex_id = uuid.uuid4().hex
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
            return result

    def update_cortex(self, updated_cortex: DBCortex) -> DBCortex | None:
        """
        Update a cortex.
        """
        with Session(self.get_sync_engine()) as session:
            session.add(updated_cortex)
            session.commit()
            session.refresh(updated_cortex)
            return updated_cortex

    def setup(self) -> bool:
        MetaStoreBase.metadata.create_all(self.get_sync_engine())
        return True

    def close(self):
        """Explicitly close the database connections."""
        super().close()

    def teardown(self) -> bool:
        self.close()
        return True
