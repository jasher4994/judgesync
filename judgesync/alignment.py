"""Main alignment tracker for JudgeSync."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from .data_loader import DataLoader
from .judge import Judge
from .metrics import AlignmentMetrics
from .types import AlignmentResults, EvaluationItem, ScoreRange


class AlignmentTracker:
    """Main class for tracking and improving judge alignment with human preferences."""

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
        score_col: str = "human_score",
        metadata_cols: Optional[List[str]] = None,
    ) -> None:
        """Load human scores from a CSV file.

        Args:
            filepath: Path to the CSV file.
            question_col: Column name for questions.
            response_col: Column name for responses.
            score_col: Column name for human scores.
            metadata_cols: Optional metadata columns to include.
        """
        self.data_loader.load_from_csv(
            filepath=filepath,
            question_col=question_col,
            response_col=response_col,
            score_col=score_col,
            metadata_cols=metadata_cols,
        )

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

        # Run the judge
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

        # Calculate metrics
        results = self.metrics.calculate(items_to_analyze)

        # Store in history with current prompt
        if self.judge:
            self.history.append(
                {
                    "prompt": self.judge.system_prompt,
                    "results": results,
                    "items_count": len(items_to_analyze),
                }
            )

        return results

    def run_alignment_test(self) -> AlignmentResults:
        """Run a complete alignment test: judge items and calculate metrics.

        Returns:
            AlignmentResults from the test.

        Raises:
            ValueError: If judge not configured or no items loaded.
        """
        if not self.judge:
            raise ValueError("No judge configured. Call set_judge() first.")

        if not self.data_loader.items:
            raise ValueError("No items loaded. Load data first.")

        # Only judge items that don't already have judge scores
        items_to_judge = [
            item for item in self.data_loader.items if item.judge_score is None
        ]

        if items_to_judge:
            self.run_judge(items_to_judge)

        # Calculate alignment on all items
        return self.calculate_alignment()

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

        # Sort by kappa score (higher is better)
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
