"""
Simple text chunker for Synapso Core.

This module provides a basic text chunker implementation using
sentence-based segmentation for plain text documents.
"""

from typing import List

from chonkie import SentenceChunker

from ...interface import Chunk, Chunker


class SimpleTxtChunker(Chunker):
    """
    Simple text chunker using sentence-based segmentation.

    Splits plain text documents into chunks based on sentence
    boundaries with configurable size and overlap.
    """

    def __init__(self):
        """
        Initialize the chunker with sentence-based configuration.
        """
        self.chunker = SentenceChunker(
            chunk_size=1024,
            chunk_overlap=30,
        )

    async def chunk_text(self, text: str) -> List[Chunk]:
        """
        Chunk text into smaller pieces based on sentences.

        Args:
            text: Text content to chunk

        Returns:
            List[Chunk]: List of text chunks with metadata including
                        start/end indices, token counts, and sentence boundaries
        """
        chunks = self.chunker.chunk(text)
        return [
            Chunk(
                text=chunk.text,
                metadata={
                    "start_index": chunk.start_index,
                    "end_index": chunk.end_index,
                    "token_count": chunk.token_count,
                    "sentences": chunk.sentences,
                },
            )
            for chunk in chunks
        ]

    async def chunk_file(self, file_path: str) -> List[Chunk]:
        """
        Chunk a text file into smaller pieces.

        Args:
            file_path: Path to the text file

        Returns:
            List[Chunk]: List of text chunks with metadata
        """
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        return await self.chunk_text(text)

    def is_file_supported(self, file_path: str) -> bool:
        """
        Check if the file is a text file.

        Args:
            file_path: Path to the file to check

        Returns:
            bool: True if the file has a .txt extension
        """
        return file_path.endswith(".txt")
