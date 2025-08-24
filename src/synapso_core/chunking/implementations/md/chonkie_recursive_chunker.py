"""
Chonkie recursive chunker for markdown documents.

This module provides a markdown chunker implementation using the
Chonkie library's recursive chunking strategy.
"""

from typing import List

from chonkie import RecursiveChunker

from ....model_provider import ModelManager, ModelNames
from ...interface import Chunk, Chunker


class ChonkieRecursiveChunker(Chunker):
    """
    Markdown chunker using Chonkie's recursive strategy.

    Splits markdown documents into chunks while preserving
    document structure and hierarchy.
    """

    def __init__(self):
        """
        Initialize the chunker with markdown-specific configuration.
        """
        self.chunker = RecursiveChunker.from_recipe("markdown", lang="en")
        self.chunker.chunk_size = 1024

    async def chunk_file(self, file_path: str) -> List[Chunk]:
        """
        Chunk a markdown file into smaller pieces.

        Args:
            file_path: Path to the markdown file

        Returns:
            List[Chunk]: List of text chunks with metadata
        """
        text = self.read_file(file_path)
        return await self.chunk_text(text)

    def is_file_supported(self, file_path: str) -> bool:
        """
        Check if the file is a markdown file.

        Args:
            file_path: Path to the file to check

        Returns:
            bool: True if the file has a .md extension
        """
        return file_path.endswith(".md")

    async def chunk_text(self, text: str) -> List[Chunk]:
        """
        Chunk markdown text into smaller pieces.

        Args:
            text: Markdown text content to chunk

        Returns:
            List[Chunk]: List of text chunks with metadata including
                        start/end indices, token counts, and hierarchy levels
        """
        model_manager = ModelManager.get_instance()
        async with model_manager.acquire(
            ModelNames.SENTENCE_TRANSFORMER_EMBEDDINGS_MODEL
        ) as model_provider:
            await model_provider.ensure_loaded()
            self.chunker.tokenizer = model_provider.tokenizer
            chunks = self.chunker.chunk(text)
            return [
                Chunk(
                    text=chunk.text,
                    metadata={
                        "start_index": chunk.start_index,
                        "end_index": chunk.end_index,
                        "token_count": chunk.token_count,
                        "level": chunk.level,
                    },
                )
                for chunk in chunks
            ]
