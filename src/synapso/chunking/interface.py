from abc import ABC, abstractmethod

from typing import Any, List, Dict
from dataclasses import dataclass, field

@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

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

    def read_file(self, file_path: str) -> str:
        """
        Read a file and return the text.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        except Exception as e:
            raise ValueError(f"Error reading file {file_path}: {e}") from e
    