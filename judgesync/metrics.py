"""Metrics for measuring alignment between human and judge scores."""

from typing import List, cast

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from .types import AlignmentResults, EvaluationItem, ScoreRange


class AlignmentMetrics:
    """Calculate alignment metrics between human and judge scores."""

    def __init__(self, score_range: ScoreRange = ScoreRange.FIVE_POINT):
        """Initialize the metrics calculator.

        Args:
            score_range: The scoring range being used.
        """
        self.score_range = score_range

    def calculate(self, items: List[EvaluationItem]) -> AlignmentResults:
        """Calculate alignment metrics from evaluation items.

        Args:
            items: List of evaluation items with both human and judge scores.

        Returns:
            AlignmentResults containing various metrics.

        Raises:
            ValueError: If items don't have both human and judge scores.
        """
        scored_items = [item for item in items if item.has_both_scores()]

        if not scored_items:
            raise ValueError("No items have both human and judge scores")

        human_scores = [cast(float, item.human_score) for item in scored_items]
        judge_scores = [cast(float, item.judge_score) for item in scored_items]

        kappa = self._calculate_kappa(human_scores, judge_scores)
        agreement = self._calculate_agreement_rate(human_scores, judge_scores)

        results = AlignmentResults(
            kappa_score=kappa,
            agreement_rate=agreement,
            sample_size=len(scored_items),
            raw_scores=list(zip(human_scores, judge_scores)),
        )

        return results

    def _calculate_kappa(
        self, human_scores: List[float], judge_scores: List[float]
    ) -> float:
        """Calculate Cohen's kappa score.

        Args:
            human_scores: List of human scores.
            judge_scores: List of judge scores.

        Returns:
            Cohen's kappa score.
        """
        # For continuous scores, we need to discretize them
        if self.score_range == ScoreRange.PERCENTAGE:
            # Convert to bins for percentage scores
            human_binned = self._bin_percentage_scores(human_scores)
            judge_binned = self._bin_percentage_scores(judge_scores)
            return cohen_kappa_score(human_binned, judge_binned)
        else:
            # For discrete scales, use directly
            human_int = [int(round(s)) for s in human_scores]
            judge_int = [int(round(s)) for s in judge_scores]
            return cohen_kappa_score(human_int, judge_int)

    def _calculate_agreement_rate(
        self,
        human_scores: List[float],
        judge_scores: List[float],
        tolerance: float = 0.5,
    ) -> float:
        """Calculate the rate of agreement between scores.

        Args:
            human_scores: List of human scores.
            judge_scores: List of judge scores.
            tolerance: Acceptable difference for agreement.

        Returns:
            Proportion of scores that agree within tolerance.
        """
        if not human_scores or not judge_scores:
            return 0.0

        agreements = sum(
            1 for h, j in zip(human_scores, judge_scores) if abs(h - j) <= tolerance
        )
        return agreements / len(human_scores)

    def _bin_percentage_scores(self, scores: List[float], n_bins: int = 5) -> List[int]:
        """Bin percentage scores into discrete categories."""
        bins = np.linspace(0, 100, n_bins + 1)
        binned = np.digitize(scores, bins) - 1
        binned = np.clip(binned, 0, n_bins - 1)
        return [int(b) for b in binned]  # Explicitly convert to int

    def calculate_correlation(
        self, items: List[EvaluationItem], method: str = "pearson"
    ) -> float:
        """Calculate correlation between human and judge scores.

        Args:
            items: List of evaluation items with both scores.
            method: Either "pearson" or "spearman".

        Returns:
            Correlation coefficient.
        """
        scored_items = [item for item in items if item.has_both_scores()]

        if not scored_items:
            raise ValueError("No items have both human and judge scores")

        # Type assertion - has_both_scores() guarantees these are not None
        human_scores = [cast(float, item.human_score) for item in scored_items]
        judge_scores = [cast(float, item.judge_score) for item in scored_items]

        if method == "pearson":
            correlation, _ = pearsonr(human_scores, judge_scores)
        elif method == "spearman":
            correlation, _ = spearmanr(human_scores, judge_scores)
        else:
            raise ValueError(f"Unknown correlation method: {method}")

        return correlation

    def get_confusion_matrix(self, items: List[EvaluationItem]) -> np.ndarray:
        """Generate confusion matrix for discrete scores.

        Args:
            items: List of evaluation items with both scores.

        Returns:
            Confusion matrix as numpy array.
        """
        scored_items = [item for item in items if item.has_both_scores()]

        if not scored_items:
            raise ValueError("No items have both human and judge scores")

        # Type assertion - has_both_scores() guarantees these are not None
        human_scores = [cast(float, item.human_score) for item in scored_items]
        judge_scores = [cast(float, item.judge_score) for item in scored_items]

        if self.score_range == ScoreRange.PERCENTAGE:
            human_binned = self._bin_percentage_scores(human_scores)
            judge_binned = self._bin_percentage_scores(judge_scores)
            return confusion_matrix(human_binned, judge_binned)
        else:
            human_int = [int(round(s)) for s in human_scores]
            judge_int = [int(round(s)) for s in judge_scores]
            return confusion_matrix(human_int, judge_int)
