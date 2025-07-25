from chonkie import RecursiveChunker
from ...interface import Chunker, Chunk

from typing import List

class ChonkieRecursiveChunker(Chunker):
    def __init__(self):
        self.chunker = RecursiveChunker.from_recipe("markdown", lang="en")

    def chunk_file(self, file_path: str) -> List[Chunk]:
        text = self.read_file(file_path)
        chunks = self.chunker.chunk(text)
        return [Chunk(text=chunk.text, metadata={
            "start_index": chunk.start_index,
            "end_index": chunk.end_index,
            "token_count": chunk.token_count,
            "level": chunk.level,
        }) for chunk in chunks]
    
    def is_file_supported(self, file_path: str) -> bool:
        return file_path.endswith(".md")