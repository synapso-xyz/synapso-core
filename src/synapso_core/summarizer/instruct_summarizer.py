"""
InstructSummarizer for Synapso Core.

This module provides the InstructSummarizer class for generating concise summaries
using instruction-tuned language models. It supports both synchronous and streaming
summarization capabilities.
"""

import asyncio
import os
import time
from typing import List, Tuple

from mlx_lm.generate import generate, stream_generate

from ..model_provider import ModelManager, ModelNames
from ..synapso_logger import get_logger
from .interface import Summarizer

logger = get_logger(__name__)


class InstructSummarizer(Summarizer):
    """
    A summarizer that uses instruction-tuned language models to generate concise summaries.

    This class provides both synchronous and streaming summarization capabilities,
    using a main model for generation and an optional draft model for acceleration.
    """

    def __init__(self):
        os.environ["TQDM_DISABLE"] = "1"
        os.environ["MLX_DISABLE_WARNINGS"] = "1"
        self.model = None
        self.tokenizer = None
        self.assistant_model = None
        self.assistant_tokenizer = None

    async def _prepare_prompt(
        self, question: str, results: List[Tuple[str, float]]
    ) -> str:
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
        assert self.tokenizer is not None
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

    async def summarize(self, question: str, results: List[Tuple[str, float]]) -> str:
        start_time = time.time()
        model_manager = ModelManager.get_instance()
        async with (
            model_manager.acquire(ModelNames.LLAMA32_LANGUAGE_MODEL) as model_provider,
            model_manager.acquire(
                ModelNames.MISTRAL_LANGUAGE_MODEL
            ) as assistant_model_provider,
        ):
            await model_provider.ensure_loaded()
            await assistant_model_provider.ensure_loaded()
            self.model = model_provider.model
            self.tokenizer = model_provider.tokenizer
            self.assistant_model = assistant_model_provider.model
            self.assistant_tokenizer = assistant_model_provider.tokenizer

            prompt = await self._prepare_prompt(question, results)
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
        model_manager = ModelManager.get_instance()
        async with (
            model_manager.acquire(ModelNames.LLAMA32_LANGUAGE_MODEL) as model_provider,
            model_manager.acquire(
                ModelNames.MISTRAL_LANGUAGE_MODEL
            ) as assistant_model_provider,
        ):
            await model_provider.ensure_loaded()
            await assistant_model_provider.ensure_loaded()
            self.model = model_provider.model
            self.tokenizer = model_provider.tokenizer
            self.assistant_model = assistant_model_provider.model
            self.assistant_tokenizer = assistant_model_provider.tokenizer

            prompt = await self._prepare_prompt(question, results)
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
