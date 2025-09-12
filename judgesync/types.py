"""Core type definitions for JudgeSync."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class ScoreRange(Enum):
    """Standard scoring ranges for evaluations."""

    BINARY = (0, 1)  # Pass/Fail
    FIVE_POINT = (1, 5)  # Likert scale
    PERCENTAGE = (0, 100)  # Percentage based
    TEN_POINT = (1, 10)  # 1-10 rating


@dataclass
class EvaluationItem:
    """A single item to be evaluated by human and/or judge.

    This is the core unit of comparison - one question/response pair
    that needs to be scored.
    """

    question: str
    response: str
    human_score: Optional[float] = None
    judge_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def has_both_scores(self) -> bool:
        """Check if both human and judge scores are present."""
        return self.human_score is not None and self.judge_score is not None

    def validate_score(self, score: float, score_range: ScoreRange) -> bool:
        """Validate that a score is within the expected range."""
        min_val, max_val = score_range.value
        return min_val <= score <= max_val


@dataclass
class AlignmentResults:
    """Results from comparing judge scores to human scores."""

    kappa_score: float
    agreement_rate: float
    sample_size: int
    raw_scores: Optional[List[tuple[float, float]]] = None
