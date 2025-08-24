"""
Cortex management for Synapso Core.

This module provides the CortexManager class for managing document collections
(cortexes) including creation, indexing, and lifecycle management.
"""

import os
from typing import List

from .config_manager import GlobalConfig, get_config
from .data_store.data_models import DBCortex
from .data_store.factory import DataStoreFactory
from .data_store.interfaces import MetaStore
from .ingestor.document_ingestor import CortexIngestor
from .synapso_logger import get_logger

logger = get_logger(__name__)


def _validate_cortex_path(cortex_path: str) -> None:
    """
    Validate that a cortex path is a valid, visible directory.

    Args:
        cortex_path: Path to validate

    Raises:
        ValueError: If the path is not a directory or is hidden
    """
    if not os.path.isdir(cortex_path):
        raise ValueError(f"Path '{cortex_path}' is not a directory.")
    # Check if the directory is visible (does not start with a dot)
    base_name = os.path.basename(os.path.normpath(cortex_path))
    if base_name.startswith("."):
        raise ValueError(f"Directory '{cortex_path}' is hidden (starts with a dot).")


class CortexManager:
    """
    Manages document collections (cortexes) in the Synapso system.

    Provides functionality for creating, indexing, and managing
    document collections with their associated metadata.
    """

    def __init__(self) -> None:
        """
        Initialize the CortexManager with configuration and data stores.
        """
        self.config: GlobalConfig = get_config()
        self.meta_store: MetaStore = DataStoreFactory.get_meta_store(
            self.config.meta_store.meta_db_type
        )
        self.sync_engine = self.meta_store.get_sync_engine()
        self.cortex_ingestor = CortexIngestor()

    def create_cortex(self, cortex_name: str, folder_path: str) -> DBCortex:
        """
        Create a new cortex (document collection).

        Args:
            cortex_name: Name for the new cortex
            folder_path: Path to the folder containing documents

        Returns:
            DBCortex: The created cortex record

        Raises:
            ValueError: If the folder path is invalid
        """
        _validate_cortex_path(folder_path)
        return self.meta_store.create_cortex(cortex_name, folder_path)

    def get_cortex_by_id(self, cortex_id: str) -> DBCortex | None:
        """
        Retrieve a cortex by its ID.

        Args:
            cortex_id: Unique identifier for the cortex

        Returns:
            DBCortex | None: The cortex record if found, None otherwise
        """
        return self.meta_store.get_cortex_by_id(cortex_id)

    def get_cortex_by_name(self, cortex_name: str) -> DBCortex | None:
        """
        Retrieve a cortex by its name.

        Args:
            cortex_name: Name of the cortex

        Returns:
            DBCortex | None: The cortex record if found, None otherwise
        """
        return self.meta_store.get_cortex_by_name(cortex_name)

    def list_cortices(self) -> List[DBCortex]:
        """
        List all available cortices.

        Returns:
            List[DBCortex]: List of all cortex records
        """
        return self.meta_store.list_cortices()

    def index_cortex(
        self,
        cortex_id: str | None = None,
        cortex_name: str | None = None,
        job_id: str | None = None,
    ) -> bool:
        """
        Index (ingest) a cortex to make its documents searchable.

        Args:
            cortex_id: ID of the cortex to index
            cortex_name: Name of the cortex to index
            job_id: Optional job ID for tracking

        Returns:
            bool: True if indexing completed successfully

        Raises:
            ValueError: If neither cortex_id nor cortex_name is provided
        """
        if not cortex_id and not cortex_name:
            raise ValueError("Either cortex_id or cortex_name must be provided")

        if cortex_name and not cortex_id:
            cortex = self.get_cortex_by_name(cortex_name)
            if not cortex:
                raise ValueError(f"Cortex with name {cortex_name} not found")
            cortex_id = cortex.cortex_id

        if not cortex_id:
            raise ValueError("Unable to find cortex_id")

        result = self.cortex_ingestor.ingest_cortex(cortex_id, job_id)
        return result

    async def delete_cortex(self, cortex_id: str) -> bool:
        """
        Delete a cortex and all its associated data.

        Args:
            cortex_id: ID of the cortex to delete

        Returns:
            bool: True if deletion was successful

        Raises:
            NotImplementedError: This method is not yet implemented
        """
        raise NotImplementedError

    async def purge_cortex(self, cortex_id: str) -> bool:
        """
        Purge all data associated with a cortex.

        Args:
            cortex_id: ID of the cortex to purge

        Returns:
            bool: True if purging was successful

        Raises:
            NotImplementedError: This method is not yet implemented
        """
        raise NotImplementedError
