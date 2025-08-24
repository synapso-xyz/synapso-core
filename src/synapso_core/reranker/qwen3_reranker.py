from typing import List, Tuple

import mlx.core as mx

from ..model_provider import ModelManager, ModelNames
from ..models import Vector
from ..synapso_logger import get_logger
from .interface import Reranker

logger = get_logger(__name__)


class Qwen3Reranker(Reranker):
    """
    A reranker that uses the Qwen/Qwen3-Reranker-0.6B model with MLX.
    This implementation loads a 6-bit quantized MLX model from Hugging Face.
    """

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.token_false_id = None
        self.token_true_id = None

    def _setup_tokenizer(self, tokenizer):
        self.token_false_id = tokenizer.convert_tokens_to_ids("no")  # type: ignore
        self.token_true_id = tokenizer.convert_tokens_to_ids("yes")  # type: ignore

    def _format_input(self, query: str, doc: str) -> str:
        """
        Formats the query and document into the specific format
        expected by the Qwen3-Reranker model.
        """
        # This instruction can be customized for the specific retrieval task.
        instruction = (
            "Given a web search query, retrieve relevant passages that answer the query"
        )
        return f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n<Query>: {query}\n<Document>: {doc}<|im_end|>\n<|im_start|>assistant\n"

    async def rerank(
        self,
        results: List[Tuple[Vector, str, float]],
        query: Vector,
        query_text: str = "",
    ) -> List[Tuple[Vector, str, float]]:
        """
        Reranks a list of documents based on their relevance to a query.

        Args:
            results: A list of tuples, where each tuple contains a vector,
                     the document text (string), and an initial score (float).
            query_text: The user's query as a string.

        Returns:
            A sorted list of tuples with updated relevance scores,
            ordered from most to least relevant.
        """
        if not results:
            return []

        # 1. Create pairs of [query, document] for the model
        # The model expects a specific instructional format.
        pairs = [self._format_input(query_text, res[1]) for res in results]

        # 2. Tokenize the pairs
        # We process all pairs in a single batch for efficiency.
        # Padding is enabled to make all sequences the same length.
        model_manager = ModelManager.get_instance()
        async with model_manager.acquire(
            ModelNames.QWEN3_LANGUAGE_MODEL
        ) as model_provider:
            await model_provider.ensure_loaded()
            self.tokenizer = model_provider.tokenizer
            self.model = model_provider.model
            self._setup_tokenizer(self.tokenizer)

            tokens = self.tokenizer._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="np",  # Return numpy arrays, which we'll convert to mx.array
            )

            input_ids = mx.array(tokens["input_ids"])
            attention_mask_2d = mx.array(tokens["attention_mask"])
            attention_mask_4d = attention_mask_2d[:, None, None, :]
            attention_mask_4d = (1.0 - attention_mask_4d) * -1e9
            model_dtype = self.model.model.layers[0].self_attn.q_proj.weight.dtype  # type: ignore
            attention_mask_4d = attention_mask_4d.astype(model_dtype)

            # 3. Get model logits
            # The model outputs logits for the entire vocabulary for each token position.
            logits = self.model(input_ids, attention_mask_4d)

            # 4. Calculate relevance scores
            # The score is determined by the logit difference between 'yes' and 'no'
            # at the last token position of each sequence.
            # We find the index of the last non-padded token for each item in the batch.
            sequence_lengths = mx.sum(attention_mask_2d, axis=1) - 1

            # Gather the logits for the last token of each sequence
            last_token_logits = logits[
                mx.arange(len(sequence_lengths)), sequence_lengths
            ]

            # The final score is the logit of "yes" minus the logit of "no".
            scores = (
                last_token_logits[:, self.token_true_id]
                - last_token_logits[:, self.token_false_id]
            )

            # Ensure computation is complete before using the scores
            mx.eval(scores)

            # 5. Combine original results with new scores and sort
            reranked_results_with_scores = list(zip(results, scores.tolist()))

            # Sort by the new score in descending order
            reranked_results_with_scores.sort(key=lambda x: x[1], reverse=True)

            # 6. Format the output to match the required interface
            final_reranked_list = []
            for original_result, new_score in reranked_results_with_scores:
                # original_result is (Vector, str, float)
                # We replace the old score with the new reranked score.
                final_reranked_list.append(
                    (original_result[0], original_result[1], new_score)
                )

            score_map = {x[1]: x[2] for x in final_reranked_list}
            logger.info("Reranked results: %s", score_map)

            filtered_results = final_reranked_list[:5]
            return filtered_results
