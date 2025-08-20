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
