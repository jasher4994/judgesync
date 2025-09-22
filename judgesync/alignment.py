"""Main alignment tracking functionality."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .comparison import ComparisonResults, JudgeComparison
from .data_loader import DataLoader
from .judge import Judge
from .metrics import AlignmentMetrics
from .types import AlignmentResults, EvaluationItem, ScoreRange

logger = logging.getLogger(__name__)


class AlignmentTracker:
    """Main class for tracking alignment between human and LLM judge scores."""

    def __init__(
        self,
        score_range: ScoreRange = ScoreRange.FIVE_POINT,
        system_prompt: Optional[str] = None,
    ):
        """Initialize the alignment tracker.

        Args:
            score_range: The scoring range to use throughout.
            system_prompt: Initial system prompt for the judge (can be set later).
        """
        self.score_range = score_range
        self.data_loader = DataLoader(score_range=score_range)
        self.metrics = AlignmentMetrics(score_range=score_range)
        self.judge: Optional[Judge] = None
        self.history: List[Dict[str, Any]] = []

        if system_prompt:
            self.set_judge(system_prompt)

    def set_judge(
        self,
        system_prompt: str,
        azure_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        deployment_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Configure the LLM judge.

        Args:
            system_prompt: The system prompt for evaluation.
            azure_endpoint: Azure OpenAI endpoint.
            api_key: Azure OpenAI API key.
            deployment_name: Azure deployment name.
            **kwargs: Additional arguments for Judge initialization.
        """
        self.judge = Judge(
            system_prompt=system_prompt,
            score_range=self.score_range,
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            deployment_name=deployment_name,
            **kwargs,
        )

    def load_human_scores_from_csv(
        self,
        filepath: str,
        question_col: str = "question",
        response_col: str = "response",
        human_score_col: str = "human_score",
    ) -> None:
        """Load human evaluation scores from a CSV file.

        Args:
            filepath: Path to the CSV file.
            question_col: Column name for questions.
            response_col: Column name for responses.
            human_score_col: Column name for human scores.
        """
        logger.info(f"Loading human scores from {filepath}")
        self.data_loader.load_from_csv(
            filepath=filepath,
            question_col=question_col,
            response_col=response_col,
            score_col=human_score_col,
        )
        logger.info(f"Loaded {len(self.data_loader.items)} items")

    def add_evaluation_item(
        self,
        question: str,
        response: str,
        human_score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Manually add an evaluation item.

        Args:
            question: The question or prompt.
            response: The response to evaluate.
            human_score: Optional human score.
            metadata: Optional metadata.
        """
        self.data_loader.add_item(
            question=question,
            response=response,
            human_score=human_score,
            metadata=metadata,
        )

    def run_judge(
        self, items: Optional[List[EvaluationItem]] = None
    ) -> List[EvaluationItem]:
        """Run the judge on evaluation items.

        Args:
            items: Optional list of items to judge. If None, uses loaded items.

        Returns:
            List of items with judge scores added.

        Raises:
            ValueError: If no judge is configured.
        """
        if not self.judge:
            raise ValueError("No judge configured. Call set_judge() first.")

        items_to_judge = items or self.data_loader.items

        if not items_to_judge:
            raise ValueError("No items to judge.")

        scored_items = self.judge.score_items(items_to_judge)

        return scored_items

    def calculate_alignment(
        self, items: Optional[List[EvaluationItem]] = None
    ) -> AlignmentResults:
        """Calculate alignment metrics.

        Args:
            items: Optional list of items. If None, uses loaded items.

        Returns:
            AlignmentResults with calculated metrics.
        """
        items_to_analyze = items or self.data_loader.items

        results = self.metrics.calculate(items_to_analyze)

        if self.judge:
            self.history.append(
                {
                    "system_prompt": self.judge.system_prompt,
                    "results": results,
                    "items_count": len(items_to_analyze),
                }
            )

        return results

    def run_alignment_test(self, use_async: bool = True) -> AlignmentResults:
        """Run alignment test with current judge and data.

        Note: This method modifies the loaded items by adding judge_score to each item.

        Args:
            use_async: Whether to use async batch processing for scoring.

        Returns:
            AlignmentResults with calculated metrics.

        Raises:
            ValueError: If no judge is configured or no items are loaded.
        """
        if not self.judge:
            raise ValueError("No judge set. Call set_judge() first.")
        if not self.data_loader.items:
            raise ValueError("No items loaded. Load data first.")

        logger.info("Running alignment test...")

        items = self.data_loader.items
        self.judge.score_items(items, use_async=use_async)

        results = self.metrics.calculate(items)

        self.history.append(
            {
                "system_prompt": self.judge.system_prompt,
                "results": results,
            }
        )

        logger.info(f"Alignment test complete. Kappa: {results.kappa_score:.3f}")
        return results

    def export_prompt(self, filepath: Optional[str] = None) -> str:
        """Export the current judge prompt.

        Args:
            filepath: Optional path to save the prompt to.

        Returns:
            The current system prompt.

        Raises:
            ValueError: If no judge is configured.
        """
        if not self.judge:
            raise ValueError("No judge configured.")

        prompt = self.judge.system_prompt

        if filepath:
            Path(filepath).write_text(prompt)

        return prompt

    def get_best_prompt(self) -> Optional[Dict[str, Any]]:
        """Get the prompt with the best alignment from history.

        Returns:
            Dictionary with prompt and results, or None if no history.
        """
        if not self.history:
            return None

        best = max(self.history, key=lambda x: x["results"].kappa_score)
        return best

    def clear_data(self) -> None:
        """Clear all loaded data items."""
        self.data_loader.clear()

    def summary(self) -> str:
        """Get a summary of the current state.

        Returns:
            String summary of the tracker state.
        """
        summary_parts = [
            "AlignmentTracker Summary:",
            f"  Score Range: {self.score_range.name}",
            f"  Items Loaded: {len(self.data_loader.items)}",
            f"  Items with Human Scores: {len(self.data_loader.get_items_with_human_scores())}",
            f"  Judge Configured: {'Yes' if self.judge else 'No'}",
            f"  Tests Run: {len(self.history)}",
        ]

        if self.history:
            latest = self.history[-1]
            summary_parts.extend(
                [
                    "\nLatest Results:",
                    f"  Kappa Score: {latest['results'].kappa_score:.3f}",
                    f"  Agreement Rate: {latest['results'].agreement_rate:.2%}",
                ]
            )

        return "\n".join(summary_parts)

    def create_comparison(self) -> JudgeComparison:
        """Create a JudgeComparison instance for comparing multiple judges.

        Returns:
            A JudgeComparison instance configured with current settings.

        Raises:
            ValueError: If Azure credentials cannot be found in environment or judge.

        Example:
            >>> tracker = AlignmentTracker()
            >>> # Automatically loads Azure credentials from .env
            >>> comparison = tracker.create_comparison()
            >>> comparison.add_judge("strict", "Be very strict")
            >>> comparison.add_judge("lenient", "Be generous")
            >>> results = comparison.run_comparison(tracker.data_loader.items)
        """
        # Try to get Azure credentials from existing judge first
        if self.judge:
            return JudgeComparison(
                score_range=self.score_range,
                azure_endpoint=self.judge.azure_endpoint,
                api_key=self.judge.api_key,
            )

        # If no judge set, try to load credentials from environment
        import os

        from dotenv import load_dotenv

        load_dotenv()

        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")

        if not azure_endpoint or not api_key:
            raise ValueError(
                "Azure OpenAI credentials not found. Either:\n"
                "1. Call set_judge() first to configure credentials, or\n"
                "2. Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in your .env file"
            )

        return JudgeComparison(
            score_range=self.score_range,
            azure_endpoint=azure_endpoint,
            api_key=api_key,
        )

    def compare_prompts(
        self,
        prompts: Dict[str, str],
        use_async: bool = True,
    ) -> ComparisonResults:
        """Compare multiple prompts to find the best one.

        Args:
            prompts: Dictionary of {name: prompt} to compare.
            use_async: Whether to use async processing.

        Returns:
            ComparisonResults with rankings.

        Example:
            >>> prompts = {
            ...     "strict": "Be very strict in evaluation",
            ...     "balanced": "Provide balanced evaluation",
            ...     "lenient": "Be generous in evaluation",
            ... }
            >>> results = tracker.compare_prompts(prompts)
            >>> print(f"Best prompt: {results.best_judge}")
        """
        if not self.data_loader.items:
            raise ValueError("No items loaded for comparison")

        comparison = self.create_comparison()

        for name, prompt in prompts.items():
            comparison.add_judge(name, prompt)

        return comparison.run_comparison(
            self.data_loader.items,
            use_async=use_async,
        )
