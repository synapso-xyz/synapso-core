"""
Document chunking for Synapso Core.

This module provides interfaces and implementations for splitting
documents into smaller, processable chunks for vectorization.
"""

from .factory import ChunkerFactory
from .interface import Chunker

__all__ = ["Chunker", "ChunkerFactory"]
