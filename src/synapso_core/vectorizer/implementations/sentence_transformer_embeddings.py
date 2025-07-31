from typing import Dict

import numpy as np
from chonkie import SentenceTransformerEmbeddings

from ...chunking.interface import Chunk
from ...persistence.interfaces.vector_store import Vector, VectorMetadata
from ...utils import get_content_hash
from ..interface import Vectorizer

_embedding_model = SentenceTransformerEmbeddings(model="all-MiniLM-L6-v2")


class SentenceTransformerVectorMetadata(VectorMetadata):
    def __init__(self, chunk: Chunk):
        self._metadata = {
            "content_hash": get_content_hash(chunk.text),
            **chunk.metadata,
        }

    def to_dict(self) -> Dict:
        return self._metadata


class SentenceTransformerVectorizer(Vectorizer):
    def __init__(self):
        # We use the all-MiniLM-L6-v2 model for now. It is a fully offline model.
        self.embeddings = _embedding_model

    def vectorize(self, chunk: Chunk) -> Vector:
        vector_vals: np.ndarray = self.embeddings.embed(chunk.text)
        return Vector(
            vector=vector_vals.tolist(),
            vector_id=get_content_hash(chunk.text),
            metadata=SentenceTransformerVectorMetadata(chunk),
        )
