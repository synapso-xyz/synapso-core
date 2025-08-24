"""
Synapso Core Package

A comprehensive document processing and vector search system that provides:
- Document ingestion and chunking
- Vector embedding and storage
- Semantic search and retrieval
- Job management and orchestration
- Configurable model providers and data stores
"""

# Synapso Core Package

from .config_manager import GlobalConfig, get_config
from .cortex_manager import CortexManager
from .ingestor.document_ingestor import CortexIngestor
from .job_manager import JobManager

__all__ = [
    "get_config",
    "GlobalConfig",
    "CortexManager",
    "CortexIngestor",
    "JobManager",
]
