from ....config_manager import get_config
from ....sqlite_utils import SqliteEngineMixin, create_sqlite_db_if_not_exists
from ...interfaces.meta_store import MetaStore
from ...models.base import MetaStoreBase


class MetaSqliteAdapter(SqliteEngineMixin, MetaStore):
    def __init__(self):
        config = get_config()
        if config.meta_store.meta_db_type != "sqlite":
            raise ValueError("Meta store type is not sqlite")
        self.meta_db_path = config.meta_store.meta_db_path
        SqliteEngineMixin.__init__(self, self.meta_db_path)

    def close(self):
        """Explicitly close the database connection."""
        # The SqliteEngineMixin handles the connection cleanup
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def metastore_setup(self) -> bool:
        create_sqlite_db_if_not_exists(self.meta_db_path)
        self._setup_tables()
        return True

    def metastore_teardown(self) -> bool:
        return super().metastore_teardown()

    def _setup_tables(self) -> None:
        MetaStoreBase.metadata.create_all(self.get_sync_engine())


if __name__ == "__main__":
    adapter = MetaSqliteAdapter()
    adapter.metastore_setup()
    print(adapter.get_sync_engine())
    print(adapter.get_async_engine())
