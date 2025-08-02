import numpy as np
from chonkie import SentenceTransformerEmbeddings

from ...chunking.interface import Chunk
from ...models import Vector, VectorMetadata
from ...utils import get_content_hash
from ..interface import Vectorizer

_embedding_model = SentenceTransformerEmbeddings(model="all-MiniLM-L6-v2")


class SentenceTransformerVectorizer(Vectorizer):
    def __init__(self):
        # We use the all-MiniLM-L6-v2 model for now. It is a fully offline model.
        self.embeddings = _embedding_model

    def vectorize(self, chunk: Chunk) -> Vector:
        vector_vals: np.ndarray = self.embeddings.embed(chunk.text)
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
