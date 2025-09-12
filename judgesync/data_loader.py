"""Data loading functionality for JudgeSync."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .types import EvaluationItem, ScoreRange

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and managing evaluation data."""

    def __init__(self, score_range: ScoreRange = ScoreRange.FIVE_POINT):
        """Initialize the DataLoader.

        Args:
            score_range: The expected range for scores.
        """
        self.score_range = score_range
        self.items: List[EvaluationItem] = []

    def add_item(
        self,
        question: str,
        response: str,
        human_score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Manually add a single evaluation item.

        Args:
            question: The question or prompt.
            response: The response to evaluate.
            human_score: The human-provided score.
            metadata: Optional metadata for the item.
        """
        if human_score is not None:
            self._validate_score(human_score)

        item = EvaluationItem(
            question=question,
            response=response,
            human_score=human_score,
            metadata=metadata,
        )
        self.items.append(item)

    def load_from_csv(
        self,
        filepath: str,
        question_col: str = "question",
        response_col: str = "response",
        score_col: str = "human_score",
        metadata_cols: Optional[List[str]] = None,
        max_rows: int = 10000,
    ) -> None:
        """Load evaluation data from a CSV file.

        Args:
            filepath: Path to the CSV file.
            question_col: Column name for questions.
            response_col: Column name for responses.
            score_col: Column name for human scores.
            metadata_cols: Optional list of column names to include as metadata.
            max_rows: Maximum number of rows to read from the CSV file.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If required columns are missing.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        df = pd.read_csv(filepath, nrows=max_rows)
        if len(df) == max_rows:
            logger.warning(f"CSV truncated to {max_rows} rows")

        required_cols = [question_col, response_col, score_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Process each row
        for _, row in df.iterrows():
            metadata = {}
            if metadata_cols:
                for col in metadata_cols:
                    if col in df.columns:
                        metadata[col] = row[col]

            # Get the score and validate it
            human_score = row[score_col]
            if pd.notna(human_score):
                human_score = float(human_score)
                self._validate_score(human_score)
            else:
                human_score = None

            # Create and add the item
            item = EvaluationItem(
                question=str(row[question_col]),
                response=str(row[response_col]),
                human_score=human_score,
                metadata=metadata if metadata else None,
            )
            self.items.append(item)

    def _validate_score(self, score: float) -> None:
        """Validate that a score is within the expected range.

        Args:
            score: The score to validate.

        Raises:
            ValueError: If the score is outside the expected range.
        """
        min_val, max_val = self.score_range.value
        if not min_val <= score <= max_val:
            raise ValueError(
                f"Score {score} is outside the expected range "
                f"[{min_val}, {max_val}] for {self.score_range.name}"
            )

    def get_items_with_human_scores(self) -> List[EvaluationItem]:
        """Get only items that have human scores.

        Returns:
            List of EvaluationItem objects with human scores.
        """
        return [item for item in self.items if item.human_score is not None]

    def clear(self) -> None:
        """Clear all loaded items."""
        self.items = []

    def __len__(self) -> int:
        """Return the number of loaded items."""
        return len(self.items)

    def __repr__(self) -> str:
        """Return a string representation of the DataLoader."""
        num_with_scores = len(self.get_items_with_human_scores())
        return (
            f"DataLoader(score_range={self.score_range.name}, "
            f"total_items={len(self.items)}, "
            f"items_with_human_scores={num_with_scores})"
        )
