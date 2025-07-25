from chonkie import RecursiveChunker
from ...interface import Chunker, Chunk

from typing import List

class ChonkieRecursiveChunker(Chunker):
    def __init__(self):
        self.chunker = RecursiveChunker.from_recipe("markdown", lang="en")

    def chunk_file(self, file_path: str) -> List[Chunk]:
        with open(file_path, "r") as file:
            text = file.read()
        return self.chunker.chunk(text)