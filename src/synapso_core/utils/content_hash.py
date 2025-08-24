"""
Content hashing utilities for Synapso Core.

This module provides functions for generating content-based hashes
used for deduplication and identification.
"""

import hashlib


def get_content_hash(text: str) -> str:
    """
    Generate a SHA-256 hash of the given text content.

    Args:
        text: The text content to hash

    Returns:
        str: Hexadecimal representation of the SHA-256 hash
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
