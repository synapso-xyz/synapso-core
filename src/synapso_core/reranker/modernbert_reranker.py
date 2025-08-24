"""
ModernBertReranker for Synapso Core.

This module provides the ModernBertReranker class for reranking search results
using the ModernBert embeddings model. It uses MLX for efficient vector operations.
"""

from typing import List, Tuple

import mlx.core as mx  # type: ignore
from mlx_embeddings import generate  # type: ignore

from ..model_provider import ModelManager, ModelNames
from ..models import Vector
from ..reranker.interface import Reranker
from ..synapso_logger import get_logger

logger = get_logger(__name__)


class ModernBertReranker(Reranker):
    """
    A reranker that uses the ModernBert embeddings model with MLX.
    This implementation loads a 6-bit quantized MLX model from Hugging Face.
    """

    def __init__(self):
        self.tokenizer = None
        self.model = None

    async def rerank(
        self,
        results: List[Tuple[Vector, str, float]],
        query: Vector,
        query_text: str = "",
    ) -> List[Tuple[Vector, str, float]]:
        model_manager = ModelManager.get_instance()
        try:
            texts_to_embed = [query_text] + [result[1] for result in results]
            async with model_manager.acquire(
                ModelNames.MODERNBERT_EMBEDDINGS_MODEL
            ) as model_provider:
                await model_provider.ensure_loaded()
                output = generate(
                    model_provider.model,
                    model_provider.tokenizer,
                    texts_to_embed,
                )
        except Exception as e:
            logger.error("Error generating embeddings: %s", e)
            return results

        embeddings = output.text_embeds  # type: ignore
        query_embedding = embeddings[0]
        results_embeddings = embeddings[1:]
        similarity_scores = mx.matmul(query_embedding, results_embeddings.T)
        reranked_results = []
        for idx, r in enumerate(results):
            score = float(similarity_scores[idx])
            reranked_results.append((r[0], r[1], score))

        reranked_results = sorted(reranked_results, key=lambda x: x[2], reverse=True)
        return reranked_results
