# rerank_e5.py
from typing import List, Tuple

import mlx.core as mx  # type: ignore
import numpy as np  # type: ignore
from mlx_embeddings import generate, load  # type: ignore

from synapso_core.models import Vector
from synapso_core.reranker.interface import Reranker
from synapso_core.synapso_logger import get_logger

MODEL = "mlx-community/nomicai-modernbert-embed-base-4bit"
model, tokenizer = load(MODEL)

logger = get_logger(__name__)


class ModernBertReranker(Reranker):
    def __init__(self):
        self.model, self.tokenizer = model, tokenizer

    def rerank(
        self,
        results: List[Tuple[Vector, str, float]],
        query: Vector,
        query_text: str = "",
    ) -> List[Tuple[Vector, str, float]]:
        texts_to_embed = [query_text] + [result[1] for result in results]
        try:
            output = generate(self.model, self.tokenizer, texts_to_embed)
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return results
        embeddings = output.text_embeds
        query_embedding = embeddings[0]
        results_embeddings = embeddings[1:]
        similarity_scores = mx.matmul(query_embedding, results_embeddings.T)
        reranked_results = []
        for idx, r in enumerate(results):
            score = float(similarity_scores[idx])
            reranked_results.append((r[0], r[1], score))

        reranked_results = sorted(reranked_results, key=lambda x: x[2], reverse=True)
        return reranked_results


if __name__ == "__main__":
    reranker = ModernBertReranker()
    results = [
        (Vector(vector_id="1", vector=np.random.rand(1024)), "result 1", 0.5),
        (Vector(vector_id="2", vector=np.random.rand(1024)), "result 2", 0.3),
        (Vector(vector_id="3", vector=np.random.rand(1024)), "result 3", 0.2),
    ]
    query = Vector(vector_id="0", vector=np.random.rand(1024))
    query_text = "query"
    reranker.rerank(results, query, query_text)
