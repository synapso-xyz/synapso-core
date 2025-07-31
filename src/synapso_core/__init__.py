# Synapso Core Package

from .config_manager import get_config, GlobalConfig
from .cortex_manager import CortexManager
from .ingestor.document_ingestor import ingest_file

__all__ = [
    "get_config",
    "GlobalConfig", 
    "CortexManager",
    "ingest_file"
]
