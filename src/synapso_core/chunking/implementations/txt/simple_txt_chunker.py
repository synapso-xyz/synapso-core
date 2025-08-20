from typing import List

from chonkie import SentenceChunker

from ....vectorizer.implementations.sentence_transformer_embeddings import tokenizer
from ...interface import Chunk, Chunker


class SimpleTxtChunker(Chunker):
    def __init__(self):
        self.chunker = SentenceChunker(
            tokenizer._tokenizer,
            chunk_size=1024,
            chunk_overlap=30,
        )

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

    def chunk_file(self, file_path: str) -> List[Chunk]:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        return self.chunk_text(text)

    def is_file_supported(self, file_path: str) -> bool:
        return file_path.endswith(".txt")
