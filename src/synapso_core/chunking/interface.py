import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


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
    def chunk_text(self, text: str) -> List[Chunk]:
        """
        Chunk text into smaller pieces.
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
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        except PermissionError as e:
            raise PermissionError(f"Permission denied: {file_path}") from e
        except IsADirectoryError as e:
            raise IsADirectoryError(
                f"Expected a file but found a directory: {file_path}"
            ) from e
        except UnicodeDecodeError as e:
            raise UnicodeDecodeError(
                "utf-8", b"", 0, 1, f"File is not valid UTF-8: {file_path}"
            ) from e
        except OSError as e:
            raise OSError(f"OS error reading file {file_path}: {e}") from e
        except Exception as e:
            raise Exception(f"Error reading file {file_path}: {e}") from e
