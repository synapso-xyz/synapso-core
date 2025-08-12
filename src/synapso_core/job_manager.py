from typing import List

from .config_manager import GlobalConfig, get_config
from .data_store import DataStoreFactory
from .data_store.data_models import IndexingJob
from .data_store.interfaces import MetaStore


class JobManager:
    def __init__(self):
        self.config: GlobalConfig = get_config()
        self.meta_store: MetaStore = DataStoreFactory.get_meta_store(
            self.config.meta_store.meta_db_type
        )

    def list_jobs(self, status: str = "IN_PROGRESS") -> List[IndexingJob]:
        return self.meta_store.list_indexing_jobs(status)

    def get_job_by_id(self, job_id: str) -> IndexingJob | None:
        return self.meta_store.get_indexing_job_by_id(job_id)
