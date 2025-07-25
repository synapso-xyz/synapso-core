from ...persistence.interfaces.vector_store import Vector, VectorMetadata
from ...chunking.interface import Chunk
from chonkie import SentenceTransformerEmbeddings
from typing import List, Dict
from ..interface import Vectorizer
import numpy as np
import hashlib

def _content_hash(chunk: Chunk) -> str:
    return hashlib.sha256(chunk.text.encode('utf-8')).hexdigest()

class SentenceTransformerVectorMetadata(VectorMetadata):
    def __init__(self, chunk: Chunk):
        self._metadata = {
            "content_hash": _content_hash(chunk),
            **chunk.metadata
        }

    def to_dict(self) -> Dict:
        return self._metadata

class SentenceTransformerVectorizer(Vectorizer):
    def __init__(self):
        # We use the all-MiniLM-L6-v2 model for now. It is a fully offline model. 
        self.embeddings = SentenceTransformerEmbeddings(model="all-MiniLM-L6-v2")

    def vectorize(self, chunks: List[Chunk]) -> Vector:
        vectors = []
        for chunk in chunks:
            vector_vals: np.ndarray = self.embeddings.embed(chunk.text)
            vector = Vector(
                vector=vector_vals.tolist(),
                vector_id=_content_hash(chunk),
                metadata=SentenceTransformerVectorMetadata(chunk)
            )
            vectors.append(vector)
        return vectors