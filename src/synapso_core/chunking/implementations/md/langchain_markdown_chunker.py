"""
LangChain markdown chunker for Synapso Core.

This module provides a markdown chunker implementation using the
LangChain text splitting library with header-based segmentation.
"""

from typing import List

from langchain_text_splitters import MarkdownHeaderTextSplitter

from ...interface import Chunk, Chunker


class LangchainMarkdownChunker(Chunker):
    """
    Markdown chunker using LangChain's header-based strategy.

    Splits markdown documents at header boundaries to preserve
    document structure and semantic meaning.
    """

    def __init__(self):
        """
        Initialize the chunker with header-based splitting configuration.
        """
        self.headers_to_split_on = [
            ("#", "header_1"),
            ("##", "header_2"),
            ("###", "header_3"),
        ]
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on
        )

    def chunk_file(self, file_path: str) -> List[Chunk]:  # type: ignore[report-override]
        """
        Chunk a markdown file into smaller pieces.

        Args:
            file_path: Path to the markdown file

        Returns:
            List[Chunk]: List of text chunks with metadata
        """
        text = self.read_file(file_path)
        return self.chunk_text(text)

    def is_file_supported(self, file_path: str) -> bool:
        """
        Check if the file is a markdown file.

        Args:
            file_path: Path to the file to check

        Returns:
            bool: True if the file has a .md extension
        """
        return file_path.endswith(".md")

    def chunk_text(self, text: str) -> List[Chunk]:  # type: ignore[report-override]
        """
        Chunk markdown text into smaller pieces.

        Args:
            text: Markdown text content to chunk

        Returns:
            List[Chunk]: List of text chunks with metadata
        """
        chunks = self.markdown_splitter.split_text(text)
        return [
            Chunk(text=chunk.page_content, metadata=chunk.metadata) for chunk in chunks
        ]
