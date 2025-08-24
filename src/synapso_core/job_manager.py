"""
Job management for Synapso Core.

This module provides the JobManager class for managing and tracking
indexing jobs and other long-running operations in the system.
"""

from typing import List

from .config_manager import GlobalConfig, get_config
from .data_store import DataStoreFactory
from .data_store.data_models import IndexingJob
from .data_store.interfaces import MetaStore


class JobManager:
    """
    Manages indexing jobs and other long-running operations.

    Provides functionality for tracking job status, retrieving
    job information, and managing job lifecycle.
    """

    def __init__(self):
        """
        Initialize the JobManager with configuration and data stores.
        """
        self.config: GlobalConfig = get_config()
        self.meta_store: MetaStore = DataStoreFactory.get_meta_store(
            self.config.meta_store.meta_db_type
        )

    def list_jobs(self, status: str = "IN_PROGRESS") -> List[IndexingJob]:
        """
        List jobs with a specific status.

        Args:
            status: Job status to filter by (default: "IN_PROGRESS")

        Returns:
            List[IndexingJob]: List of jobs matching the status
        """
        return self.meta_store.list_indexing_jobs(status)

    def get_job_by_id(self, job_id: str) -> IndexingJob | None:
        """
        Retrieve a specific job by its ID.

        Args:
            job_id: Unique identifier for the job

        Returns:
            IndexingJob | None: The job record if found, None otherwise
        """
        return self.meta_store.get_indexing_job_by_id(job_id)
