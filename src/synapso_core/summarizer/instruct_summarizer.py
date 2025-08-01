from typing import List, Tuple

from mlx_lm import generate, load

SUMMARIZER_MODEL = "mlx-community/Phi-4-mini-instruct-4bit"


class InstructSummarizer:
    def __init__(self):
        self.model, self.tokenizer = load(SUMMARIZER_MODEL)
        self.tokenizer.add_eos_token("END_ANSWER")

    def _prepare_prompt(self, question: str, results: List[Tuple[str, float]]) -> str:
        prompt = """
        Answer the following questions using only the context provided. Cite sources like [1].

        Example 1:
        Context:
        [1] The sun is a star. It provides light and heat to Earth.
        [2] The moon orbits the Earth.

        Question: What provides heat to the Earth?
        Answer: The sun [1].END_ANSWER

        ---

        Example 2:
        Context:
        [1] France is in Europe. Paris is its capital.
        [2] Berlin is the capital of Germany.

        Question: What is the capital of Germany?
        Answer: Berlin [2].END_ANSWER
        ---

        Now answer the next question using only the context. Cite sources like [0], [1]. Keep your answer short and concise.

        Context:

        """
        for idx, (text, score) in enumerate(results):
            prompt += f"[{idx}] {text}\n"

        prompt += f"""
        Question: {question}
        Answer:
        """

    def summarize(self, question: str, results: List[Tuple[str, float]]) -> str:
        prompt = self._prepare_prompt(question, results)
        response = generate(self.model, self.tokenizer, prompt, max_tokens=150)
        return response.split("END_ANSWER")[0].strip()
