from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class VectorMetadata(ABC):
    """
    A metadata object for a vector.
    """

    @abstractmethod
    def to_dict(self) -> Dict:
        pass


@dataclass
class Vector:
    vector_id: str
    vector: List[float]
    metadata: VectorMetadata


class VectorStore(ABC):
    """
    A store for vectors.
    """

    @abstractmethod
    def setup(self) -> bool:
        pass

    @abstractmethod
    def insert(self, vector: Vector) -> bool:
        """
        Insert a new vector with its associated metadata into the store.
        """
        pass

    @abstractmethod
    def get_by_id(self, vector_id: str) -> Vector | None:
        """
        Retrieve a vector and its metadata by document ID.
        """
        pass

    @abstractmethod
    def vector_search(
        self, query_vector: Vector, top_k: int = 5, filters: Dict | None = None
    ) -> List[Tuple[Vector, float]]:
        """
        Search for vectors most similar to the query_vector. Optionally filter by metadata.
        Returns a list of tuples of (vector, score).
        """
        pass

    @abstractmethod
    def delete(self, vector_id: str) -> bool:
        """
        Delete a vector and its metadata by document ID.
        """
        pass

    @abstractmethod
    def update_metadata(self, vector_id: str, metadata: VectorMetadata) -> bool:
        """
        Update the metadata for a given document ID.
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """
        Return the total number of vectors in the store.
        """
        pass

    @abstractmethod
    def teardown(self) -> bool:
        """
        Clean up or release any resources held by the vector store.
        """
        pass
