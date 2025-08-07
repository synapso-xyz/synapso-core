import asyncio
import os
import time
from typing import List, Tuple

from mlx_lm.generate import generate, stream_generate
from mlx_lm.utils import load

from ..synapso_logger import get_logger

SUMMARIZER_MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"
ASSISTANT_MODEL = "mlx-community/TinyMistral-248M-8bits"

# SUMMARIZER_MODEL = "mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit"

model, tokenizer = load(SUMMARIZER_MODEL)
assistant_model, _ = load(ASSISTANT_MODEL)

logger = get_logger(__name__)


class InstructSummarizer:
    def __init__(self):
        os.environ["TQDM_DISABLE"] = "1"
        os.environ["MLX_DISABLE_WARNINGS"] = "1"
        logger.info("InstructSummarizer initialized with model %s", SUMMARIZER_MODEL)
        self.model, self.tokenizer = model, tokenizer
        self.assistant_model = assistant_model

    def _prepare_prompt(self, question: str, results: List[Tuple[str, float]]) -> str:
        """
        Prepares the prompt for the LLM using a simple format.
        """
        # 1. Define the core instructions in a system prompt for clarity.
        system_prompt = """You are a precise and concise AI assistant.
Your task is to synthesize information to answer a question directly.
Do not repeat the question or summarize the provided context.
Provide only the direct answer, citing the sources as required. 
When you state a fact, you must cite the source like [number].
You dont need to list the sources, just answer the question."""

        # 2. Format the context documents with newlines for readability.
        formatted_context = "\n".join(
            f"[{i + 1}] {doc.strip()}" for i, (doc, _) in enumerate(results)
        )

        # 3. Create the final user message.
        user_prompt = f"""Context:
{formatted_context}

Question: {question}"""

        # 4. Construct the messages list without the pre-filled assistant turn.
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # 5. Use apply_chat_template, which will correctly add the assistant prompt turn.
        # This is the most reliable way to format prompts.
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(  # type: ignore
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback for older tokenizers, now with better formatting.
            prompt = f"{system_prompt}\n\n{user_prompt}\n\nAnswer:"

        logger.info("Prompt: %s", prompt)
        logger.info("Prompt length: %s", len(prompt))
        return prompt

    def summarize(self, question: str, results: List[Tuple[str, float]]) -> str:
        start_time = time.time()
        prompt = self._prepare_prompt(question, results)
        prompt_time = time.time()
        logger.info("Prompt prepared in %s seconds", prompt_time - start_time)
        start_time = time.time()
        response = generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=100,
            draft_model=self.assistant_model,
        )
        response_time = time.time()
        logger.info("Response generated in %s seconds", response_time - start_time)

        # Handle response properly
        if isinstance(response, str):
            # Check for EOS token if it exists
            eos_token = getattr(self.tokenizer, "eos_token", None)
            if eos_token and eos_token in response:
                return response.split(eos_token)[0].strip()
            return response.strip()
        return str(response)

    async def run_summarizer_stream(
        self, question: str, results: List[Tuple[str, float]]
    ):
        start_time = time.time()
        prompt = self._prepare_prompt(question, results)
        prompt_time = time.time()
        logger.info("Prompt prepared in %s seconds", prompt_time - start_time)
        start_time = time.time()
        logger.info("EOS token: %s", getattr(self.tokenizer, "eos_token", None))
        first_token = True

        # Use stream_generate which returns a synchronous generator
        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=256,
            # draft_model=self.assistant_model,
        ):
            if hasattr(response, "token") and response.token == getattr(
                self.tokenizer, "eos_token", None
            ):
                logger.info("EOS token found")
                break
            if hasattr(response, "text") and "<|end|>" in response.text:
                logger.info("End token found")
                yield response.text.split("<|end|>")[0].strip() + "\n"
                break

            if first_token:
                first_token = False
                logger.info("First token time: %s", time.time() - start_time)

            # Yield the token and allow other coroutines to run
            yield response.text if hasattr(response, "text") else str(response)
            await asyncio.sleep(0)  # Allow other coroutines to run

        stream_time = time.time()
        logger.info(
            "Summarizer stream completed in %s seconds", stream_time - start_time
        )
