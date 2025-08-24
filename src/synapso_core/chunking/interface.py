"""
Chunking interface for Synapso Core.

This module defines the abstract interface for document chunking,
including the Chunk data structure and Chunker abstract base class.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Chunk:
    """
    Represents a chunk of text with associated metadata.

    Attributes:
        text: The text content of the chunk
        metadata: Additional metadata associated with the chunk
    """

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class Chunker(ABC):
    """
    Abstract base class for document chunking.

    A chunker is responsible for splitting documents into smaller,
    processable pieces while preserving context and metadata.
    """

    @abstractmethod
    async def chunk_file(self, file_path: str) -> List[Chunk]:
        """
        Chunk a file into smaller pieces.

        Args:
            file_path: Path to the file to chunk

        Returns:
            List[Chunk]: List of text chunks with metadata
        """

    @abstractmethod
    async def chunk_text(self, text: str) -> List[Chunk]:
        """
        Chunk text into smaller pieces.

        Args:
            text: Text content to chunk

        Returns:
            List[Chunk]: List of text chunks with metadata
        """

    @abstractmethod
    def is_file_supported(self, file_path: str) -> bool:
        """
        Check if a file is supported by the chunker.

        Args:
            file_path: Path to the file to check

        Returns:
            bool: True if the file type is supported
        """

    def read_file(self, file_path: str) -> str:
        """
        Read a file and return the text content.

        Args:
            file_path: Path to the file to read

        Returns:
            str: The text content of the file

        Raises:
            FileNotFoundError: If the file doesn't exist
            PermissionError: If permission is denied
            IsADirectoryError: If the path is a directory
            UnicodeDecodeError: If the file isn't valid UTF-8
            OSError: For other OS-related errors
            Exception: For other unexpected errors
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
