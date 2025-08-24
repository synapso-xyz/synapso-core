"""
Data models for Synapso Core.

This module defines the core data structures used throughout the system,
including vectors, metadata, and other domain objects.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class VectorMetadata:
    """
    Metadata associated with a vector embedding.

    Contains information about the source and context of a vector,
    including content hash, cortex ID, and source file ID.

    Attributes:
        content_hash: Hash of the vector's content for deduplication
        cortex_id: ID of the cortex this vector belongs to
        source_file_id: ID of the source file this vector was derived from
        additional_data: Additional metadata as key-value pairs
    """

    content_hash: str
    cortex_id: str | None = None
    source_file_id: str | None = None
    additional_data: Dict | None = None

    def to_dict(self) -> Dict:
        """
        Convert the metadata to a dictionary representation.

        Returns:
            Dict: Dictionary representation of the metadata
        """
        return {
            "content_hash": self.content_hash,
            "cortex_id": self.cortex_id,
            "source_file_id": self.source_file_id,
            "additional_data": self.additional_data,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "VectorMetadata":
        """
        Create a VectorMetadata instance from a dictionary.

        Args:
            data: Dictionary containing metadata fields

        Returns:
            VectorMetadata: New instance with the provided data

        Raises:
            ValueError: If the data is invalid or missing required fields
        """
        if not data:
            raise ValueError("Data is empty")
        if not isinstance(data, dict):
            raise ValueError("Data is not a dictionary")
        if not all(
            key in data
            for key in [
                "content_hash",
            ]
        ):
            msg = f"Data is missing required fields. Expected: {['content_hash', 'cortex_id', 'source_file_id', 'additional_data']}. Found: {data.keys()}"
            raise ValueError(msg)

        return cls(
            content_hash=data["content_hash"],
            cortex_id=data.get("cortex_id"),
            source_file_id=data.get("source_file_id"),
            additional_data=data.get("additional_data"),
        )


@dataclass
class Vector:
    """
    A vector embedding with associated metadata.

    Represents a numerical vector (embedding) along with its
    unique identifier and metadata.

    Attributes:
        vector_id: Unique identifier for the vector
        vector: Numerical representation as a list of floats
        metadata: Associated metadata for the vector
    """

    vector_id: str
    vector: List[float]
    metadata: VectorMetadata | None = None
