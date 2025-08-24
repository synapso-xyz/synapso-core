"""
Summarization for Synapso Core.

This module provides interfaces and implementations for generating
document summaries and answering queries based on search results.
"""

from .factory import SummarizerFactory
from .interface import Summarizer

__all__ = ["SummarizerFactory", "Summarizer"]
