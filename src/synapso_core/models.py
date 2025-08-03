from dataclasses import dataclass
from typing import Dict, List


@dataclass
class VectorMetadata:
    """
    A metadata object for a vector.
    """

    content_hash: str
    cortex_id: str | None = None
    source_file_id: str | None = None
    additional_data: Dict | None = None

    def to_dict(self) -> Dict:
        return {
            "content_hash": self.content_hash,
            "cortex_id": self.cortex_id,
            "source_file_id": self.source_file_id,
            "additional_data": self.additional_data,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "VectorMetadata":
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
    vector_id: str
    vector: List[float]
    metadata: VectorMetadata | None = None
