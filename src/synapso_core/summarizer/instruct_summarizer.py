import os
import time
from typing import List, Tuple

from mlx_lm import generate, load

from ..synapso_logger import get_logger

SUMMARIZER_MODEL = "mlx-community/Phi-3-mini-4k-instruct-4bit-no-q-embed"
model, tokenizer = load(SUMMARIZER_MODEL)


logger = get_logger(__name__)


class InstructSummarizer:
    def __init__(self):
        os.environ["TQDM_DISABLE"] = "1"
        os.environ["MLX_DISABLE_WARNINGS"] = "1"
        self.model, self.tokenizer = model, tokenizer
        self.tokenizer.add_eos_token("END_ANSWER")

    def _prepare_prompt(self, question: str, results: List[Tuple[str, float]]) -> str:
        prompt = """
        Answer only from the context and cite sources like [1]. End with END_ANSWER.

        Context:
        """
        for idx, (text, score) in enumerate(results):
            prompt += f"[{idx}] {text}\n"

        prompt += f"""
        Question: {question}
        Answer:
        """

        logger.debug("Prompt: %s", prompt)

        return prompt

    def summarize(self, question: str, results: List[Tuple[str, float]]) -> str:
        start_time = time.time()
        prompt = self._prepare_prompt(question, results)
        prompt_time = time.time()
        logger.info("Prompt prepared in %s seconds", prompt_time - start_time)
        start_time = time.time()
        response = generate(self.model, self.tokenizer, prompt, max_tokens=150)
        response_time = time.time()
        logger.info("Response generated in %s seconds", response_time - start_time)
        return response.split("END_ANSWER")[0].strip()
