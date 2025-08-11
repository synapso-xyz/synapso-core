from typing import List

from chonkie import SentenceChunker

from ...interface import Chunk, Chunker


class SimpleTxtChunker(Chunker):
    def __init__(self):
        self.chunker = SentenceChunker(chunk_size=256, chunk_overlap=30)

    def chunk_text(self, text: str) -> List[Chunk]:
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

    def is_file_supported(self, file_path: str) -> bool:
        return file_path.endswith(".txt")
