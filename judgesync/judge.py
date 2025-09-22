"""LLM Judge functionality using Azure OpenAI."""

import asyncio
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, AzureOpenAI

from .types import EvaluationItem, ScoreRange

logger = logging.getLogger(__name__)

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class Judge:
    """Manages LLM judge interactions using Azure OpenAI."""

    def __init__(
        self,
        system_prompt: str,
        score_range: ScoreRange = ScoreRange.FIVE_POINT,
        azure_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        deployment_name: Optional[str] = None,
        api_version: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        """Initialize the Judge with Azure OpenAI configuration.

        Args:
            system_prompt: The system prompt that defines how the judge should evaluate.
            score_range: The expected scoring range.
            azure_endpoint: Azure OpenAI endpoint (or set AZURE_OPENAI_ENDPOINT env var).
            api_key: Azure OpenAI API key (or set AZURE_OPENAI_API_KEY env var).
            deployment_name: Azure deployment name (or set AZURE_OPENAI_DEPLOYMENT env var).
            api_version: Azure OpenAI API version.
            temperature: Temperature for response generation (0-2). None uses model default.
                        Note: Not all Azure models support temperature parameter.
        """
        # Load environment variables
        load_dotenv()

        self.system_prompt = system_prompt
        self.score_range = score_range

        # Handle temperature
        if temperature is not None:
            if not 0 <= temperature <= 2:
                raise ValueError("Temperature must be between 0 and 2")
            logger.info(
                f"Temperature set to {temperature}. Note: Not all Azure models support "
                "temperature parameter. If you encounter errors, try setting temperature=None."
            )
        self.temperature = temperature

        # Get Azure configuration from parameters or environment
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.deployment_name = deployment_name or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.api_version = api_version or os.getenv(
            "AZURE_OPENAI_API_VERSION", "2024-02-01"
        )

        if not all([self.azure_endpoint, self.api_key, self.deployment_name]):
            raise ValueError(
                "Azure OpenAI configuration incomplete. Please provide "
                "azure_endpoint, api_key, and deployment_name either as "
                "parameters or as environment variables."
            )

        self.client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
        )

        self.async_client = AsyncAzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
        )

        self.last_response: Optional[Dict[str, Any]] = None

    def score_item(self, item: EvaluationItem) -> float:
        """Score a single evaluation item using the LLM judge.

        Args:
            item: The evaluation item to score.

        Returns:
            The numerical score from the judge.

        Raises:
            ValueError: If the response cannot be parsed as a valid score.
        """
        user_prompt = self._build_user_prompt(item)

        api_params = {
            "model": self.deployment_name,
            "messages": [
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": user_prompt},
            ],
        }

        if self.temperature is not None:
            api_params["temperature"] = self.temperature

        response = self.client.chat.completions.create(**api_params)

        self.last_response = response.model_dump()

        response_text = response.choices[0].message.content.strip()

        score_match = re.search(r"(\d+(?:\.\d+)?)", response_text)
        if not score_match:
            raise ValueError(f"Could not parse score from response: {response_text}")

        try:
            score = float(score_match.group(1))
        except (ValueError, AttributeError) as e:
            raise ValueError(
                f"Could not parse score from response: {response_text}"
            ) from e

        min_score, max_score = self.score_range.value
        if not min_score <= score <= max_score:
            raise ValueError(
                f"Score {score} is outside of expected range [{min_score}, {max_score}]"
            ) from None

        return score

    async def _score_item_async(
        self, item: EvaluationItem
    ) -> Tuple[EvaluationItem, Optional[float]]:
        """Score a single item asynchronously.

        Args:
            item: The evaluation item to score.

        Returns:
            Tuple of (item, score) where score is None if error occurred.
        """
        try:
            user_prompt = self._build_user_prompt(item)

            api_params = {
                "model": self.deployment_name,
                "messages": [
                    {"role": "system", "content": self._build_system_prompt()},
                    {"role": "user", "content": user_prompt},
                ],
            }

            if self.temperature is not None:
                api_params["temperature"] = self.temperature

            response = await self.async_client.chat.completions.create(**api_params)

            response_text = response.choices[0].message.content.strip()

            score_match = re.search(r"(\d+(?:\.\d+)?)", response_text)
            if not score_match:
                logger.warning(
                    f"Could not parse score from response: {response_text[:100]}"
                )
                return item, None

            score = float(score_match.group(1))

            min_score, max_score = self.score_range.value
            if not min_score <= score <= max_score:
                logger.warning(
                    f"Score {score} outside range [{min_score}, {max_score}]"
                )
                return item, None

            return item, score

        except Exception as e:
            logger.error(f"Error scoring item: {e}")
            return item, None

    async def _score_batch_async(
        self,
        items: List[EvaluationItem],
        batch_size: int = 10,
        delay_between_batches: float = 0.1,
    ) -> List[EvaluationItem]:
        """Score items in batches asynchronously.

        Args:
            items: List of items to score.
            batch_size: Number of concurrent requests per batch.
            delay_between_batches: Delay between batches to prevent rate limiting.

        Returns:
            List of items with judge_score populated where successful.
        """
        results = []

        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]

            batch_tasks = [self._score_item_async(item) for item in batch]
            batch_results = await asyncio.gather(*batch_tasks)

            for item, score in batch_results:
                if score is not None:
                    item.judge_score = score
                results.append(item)

            if i + batch_size < len(items):
                await asyncio.sleep(delay_between_batches)

        return results

    def score_items_async(
        self,
        items: List[EvaluationItem],
        batch_size: int = 10,
        delay_between_batches: float = 0.1,
        show_progress: bool = True,
    ) -> List[EvaluationItem]:
        """Score multiple items using async batch processing.

        This is much faster than sequential scoring, especially for large datasets.

        Args:
            items: List of evaluation items to score.
            batch_size: Number of concurrent API calls per batch.
            delay_between_batches: Delay between batches (seconds).
            show_progress: Whether to show progress bar.

        Returns:
            The same items with judge_score populated where successful.

        Example:
            >>> items = [EvaluationItem(...) for _ in range(100)]
            >>> scored_items = judge.score_items_async(items, batch_size=20)
            Processing 100 items in 5 batches...
            Batch 1/5: ████████████ 100%
            ...
        """
        if show_progress:
            n_batches = (len(items) + batch_size - 1) // batch_size
            logger.info(f"Processing {len(items)} items in {n_batches} batches...")

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                try:
                    import nest_asyncio

                    nest_asyncio.apply()
                except ImportError:
                    logger.warning(
                        "nest_asyncio not installed - async may not work in notebooks"
                    )
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        result = asyncio.run(
            self._score_batch_async(items, batch_size, delay_between_batches)
        )

        if show_progress:
            scored_count = sum(1 for item in result if item.judge_score is not None)
            logger.info(f"Successfully scored {scored_count}/{len(items)} items")

        return result

    def score_items(
        self, items: List[EvaluationItem], delay: float = 0.1, use_async: bool = True
    ) -> List[EvaluationItem]:
        """Score multiple evaluation items.

        Args:
            items: List of evaluation items to score.
            delay: Delay between scoring items (for sync mode).
            use_async: Whether to use async batch processing (much faster).

        Returns:
            The same items with judge_score populated.
        """
        if use_async and len(items) > 5:
            # Use async for better performance on larger datasets
            return self.score_items_async(items)

        # Fall back to sequential processing for small datasets
        for i, item in enumerate(items):
            try:
                item.judge_score = self.score_item(item)
                time.sleep(delay)
            except Exception as e:
                logger.error(f"Error scoring item {i}: {e}")
                continue

        return items

    def _build_system_prompt(self) -> str:
        """Build the complete system prompt including score range instructions.

        Returns:
            The formatted system prompt.
        """
        min_val, max_val = self.score_range.value

        range_instruction = (
            f"\nYou must respond with ONLY a number between {min_val} and {max_val}."
        )

        return f"{self.system_prompt}{range_instruction}"

    def _build_user_prompt(self, item: EvaluationItem) -> str:
        """Build the user prompt for evaluation.

        Args:
            item: The evaluation item.

        Returns:
            The formatted user prompt.
        """
        return f"Question: {item.question}\n\nResponse: {item.response}"

    def update_system_prompt(self, new_prompt: str) -> None:
        """Update the system prompt for the judge.

        Args:
            new_prompt: The new system prompt.
        """
        self.system_prompt = new_prompt
