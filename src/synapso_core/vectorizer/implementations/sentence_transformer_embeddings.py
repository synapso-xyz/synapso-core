from typing import List

from mlx_embeddings import generate, load  # type: ignore

from ...chunking.interface import Chunk
from ...models import Vector, VectorMetadata
from ...utils import get_content_hash
from ..interface import Vectorizer

MODEL_NAME = "mlx-community/all-MiniLM-L6-v2-bf16"
model, tokenizer = load(MODEL_NAME)


class SentenceTransformerVectorizer(Vectorizer):
    def __init__(self):
        # We use the all-MiniLM-L6-v2 model for now. It is a fully offline model.
        self.model, self.tokenizer = model, tokenizer

    def vectorize(self, chunk: Chunk) -> Vector:
        output = generate(self.model, self.tokenizer, texts=[chunk.text])
        vector_vals = output.text_embeds[0]
        print(f"Vector values: {vector_vals}")
        return Vector(
            vector=vector_vals.tolist(),
            vector_id=get_content_hash(chunk.text),
            metadata=VectorMetadata(
                content_hash=get_content_hash(chunk.text),
                cortex_id=chunk.metadata.get("cortex_id"),
                source_file_id=chunk.metadata.get("source_file_id"),
                additional_data=chunk.metadata,
            ),
        )

    def vectorize_batch(self, chunks: List[Chunk]) -> List[Vector]:
        if not chunks:
            return []

        output = generate(
            self.model, self.tokenizer, texts=[chunk.text for chunk in chunks]
        )
        vector_vals = output.text_embeds
        return [
            Vector(
                vector=vector_vals[i].tolist(),
                vector_id=get_content_hash(chunks[i].text),
                metadata=VectorMetadata(
                    content_hash=get_content_hash(chunks[i].text),
                    cortex_id=chunks[i].metadata.get("cortex_id"),
                    source_file_id=chunks[i].metadata.get("source_file_id"),
                    additional_data=chunks[i].metadata,
                ),
            )
            for i in range(len(chunks))
        ]
