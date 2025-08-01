from typing import List

from langchain_text_splitters import MarkdownHeaderTextSplitter

from ...interface import Chunk, Chunker


class LangchainMarkdownChunker(Chunker):
    def __init__(self):
        self.headers_to_split_on = [
            ("#", "header_1"),
            ("##", "header_2"),
            ("###", "header_3"),
        ]
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on
        )

    def chunk_file(self, file_path: str) -> List[Chunk]:
        text = self.read_file(file_path)
        return self.chunk_text(text)

    def is_file_supported(self, file_path: str) -> bool:
        return file_path.endswith(".md")

    def chunk_text(self, text: str) -> List[Chunk]:
        chunks = self.markdown_splitter.split_text(text)
        return [
            Chunk(text=chunk.page_content, metadata=chunk.metadata) for chunk in chunks
        ]
