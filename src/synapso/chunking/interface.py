from abc import ABC, abstractmethod

from typing import List, Dict
from dataclasses import dataclass, field

@dataclass
class Chunk:
    text: str
    metadata: Dict[str, str] = field(default_factory=dict)

class Chunker(ABC):
    """
    A chunker is a class that chunks text into smaller pieces.
    """

    @abstractmethod
    def chunk_file(self, file_path: str) -> List[Chunk]:
        """
        Chunk a file into smaller pieces.
        """
        pass

    @abstractmethod
    def is_file_supported(self, file_path: str) -> bool:
        """
        Check if a file is supported by the chunker.
        """
        pass

    