"""Tests for the metrics module."""

import warnings

import numpy as np
import pytest

from judgesync.metrics import AlignmentMetrics
from judgesync.types import EvaluationItem, ScoreRange


class TestAlignmentMetrics:
    """Test the AlignmentMetrics class."""

    def test_init_with_default_score_range(self):
        """Test initialization with default score range."""
        metrics = AlignmentMetrics()
        assert metrics.score_range == ScoreRange.FIVE_POINT

    def test_init_with_custom_score_range(self):
        """Test initialization with custom score range."""
        metrics = AlignmentMetrics(score_range=ScoreRange.PERCENTAGE)
        assert metrics.score_range == ScoreRange.PERCENTAGE

    def test_calculate_with_perfect_agreement(self):
        """Test metrics calculation with perfect agreement."""
        items = [
            EvaluationItem("Q1", "R1", human_score=5, judge_score=5),
            EvaluationItem("Q2", "R2", human_score=3, judge_score=3),
            EvaluationItem("Q3", "R3", human_score=1, judge_score=1),
        ]

        metrics = AlignmentMetrics(score_range=ScoreRange.FIVE_POINT)
        results = metrics.calculate(items)

        assert results.kappa_score == 1.0  # Perfect agreement
        assert results.agreement_rate == 1.0  # 100% agreement
        assert results.sample_size == 3
        assert len(results.raw_scores) == 3

    def test_calculate_with_partial_agreement(self):
        """Test metrics calculation with partial agreement."""
        items = [
            EvaluationItem("Q1", "R1", human_score=5, judge_score=5),
            EvaluationItem("Q2", "R2", human_score=3, judge_score=4),  # Disagree by 1
            EvaluationItem("Q3", "R3", human_score=1, judge_score=1),
        ]

        metrics = AlignmentMetrics(score_range=ScoreRange.FIVE_POINT)
        results = metrics.calculate(items)

        assert results.sample_size == 3
        assert 0 < results.kappa_score < 1.0  # Partial agreement
        assert results.agreement_rate < 1.0

    def test_calculate_with_no_agreement(self):
        """Test metrics calculation with systematic disagreement."""
        items = [
            EvaluationItem("Q1", "R1", human_score=5, judge_score=1),
            EvaluationItem("Q2", "R2", human_score=4, judge_score=2),
            EvaluationItem("Q3", "R3", human_score=3, judge_score=5),
        ]

        metrics = AlignmentMetrics(score_range=ScoreRange.FIVE_POINT)
        results = metrics.calculate(items)

        assert results.kappa_score < 0.5  # Poor agreement
        assert results.agreement_rate < 0.5

    def test_calculate_filters_incomplete_items(self):
        """Test that calculate only uses items with both scores."""
        items = [
            EvaluationItem("Q1", "R1", human_score=5, judge_score=5),
            EvaluationItem("Q2", "R2", human_score=3),  # Missing judge score
            EvaluationItem("Q3", "R3", judge_score=1),  # Missing human score
            EvaluationItem("Q4", "R4", human_score=2, judge_score=2),
        ]

        metrics = AlignmentMetrics()
        results = metrics.calculate(items)

        assert results.sample_size == 2  # Only Q1 and Q4
        assert len(results.raw_scores) == 2

    def test_calculate_raises_error_with_no_complete_items(self):
        """Test that calculate raises error when no items have both scores."""
        items = [
            EvaluationItem("Q1", "R1", human_score=5),
            EvaluationItem("Q2", "R2", judge_score=3),
            EvaluationItem("Q3", "R3"),
        ]

        metrics = AlignmentMetrics()
        with pytest.raises(
            ValueError, match="No items have both human and judge scores"
        ):
            metrics.calculate(items)

    def test_agreement_rate_with_tolerance(self):
        """Test agreement rate calculation with tolerance."""
        items = [
            EvaluationItem("Q1", "R1", human_score=5.0, judge_score=5.0),  # Exact
            EvaluationItem("Q2", "R2", human_score=3.0, judge_score=3.4),  # Within 0.5
            EvaluationItem("Q3", "R3", human_score=2.0, judge_score=3.0),  # Outside 0.5
        ]

        metrics = AlignmentMetrics()
        results = metrics.calculate(items)

        # With default tolerance of 0.5, should have 2/3 agreement
        assert results.agreement_rate == pytest.approx(2 / 3, rel=0.01)

    def test_calculate_correlation_pearson(self):
        """Test Pearson correlation calculation."""
        items = [
            EvaluationItem("Q1", "R1", human_score=1, judge_score=2),
            EvaluationItem("Q2", "R2", human_score=2, judge_score=3),
            EvaluationItem("Q3", "R3", human_score=3, judge_score=4),
            EvaluationItem("Q4", "R4", human_score=4, judge_score=5),
        ]

        metrics = AlignmentMetrics()
        correlation = metrics.calculate_correlation(items, method="pearson")

        assert correlation == pytest.approx(1.0, rel=0.01)  # Perfect linear correlation

    def test_calculate_correlation_spearman(self):
        """Test Spearman correlation calculation."""
        items = [
            EvaluationItem("Q1", "R1", human_score=1, judge_score=1),
            EvaluationItem("Q2", "R2", human_score=2, judge_score=2),
            EvaluationItem("Q3", "R3", human_score=3, judge_score=3),
        ]

        metrics = AlignmentMetrics()
        correlation = metrics.calculate_correlation(items, method="spearman")

        assert correlation == pytest.approx(1.0, rel=0.01)  # Perfect rank correlation

    def test_calculate_correlation_invalid_method(self):
        """Test that invalid correlation method raises error."""
        items = [
            EvaluationItem("Q1", "R1", human_score=1, judge_score=2),
        ]

        metrics = AlignmentMetrics()
        with pytest.raises(ValueError, match="Unknown correlation method"):
            metrics.calculate_correlation(items, method="invalid")

    def test_bin_percentage_scores(self):
        """Test binning of percentage scores."""
        metrics = AlignmentMetrics(score_range=ScoreRange.PERCENTAGE)

        scores = [10, 30, 50, 70, 90]
        binned = metrics._bin_percentage_scores(scores, n_bins=5)

        assert len(binned) == 5
        assert all(0 <= b < 5 for b in binned)  # All in valid bin range

    def test_confusion_matrix_discrete_scores(self):
        """Test confusion matrix generation for discrete scores."""
        items = [
            EvaluationItem("Q1", "R1", human_score=1, judge_score=1),
            EvaluationItem("Q2", "R2", human_score=2, judge_score=3),
            EvaluationItem("Q3", "R3", human_score=3, judge_score=3),
        ]

        metrics = AlignmentMetrics(score_range=ScoreRange.FIVE_POINT)

        # Suppress the sklearn warning about single labels
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            matrix = metrics.get_confusion_matrix(items)

        assert matrix.shape[0] == matrix.shape[1]  # Square matrix
        assert matrix.sum() == 3  # Total of 3 items

    def test_confusion_matrix_percentage_scores(self):
        """Test confusion matrix generation for percentage scores."""
        items = [
            EvaluationItem("Q1", "R1", human_score=10, judge_score=15),
            EvaluationItem("Q2", "R2", human_score=50, judge_score=45),
            EvaluationItem("Q3", "R3", human_score=90, judge_score=85),
        ]

        metrics = AlignmentMetrics(score_range=ScoreRange.PERCENTAGE)
        matrix = metrics.get_confusion_matrix(items)

        assert matrix.shape[0] == matrix.shape[1]  # Square matrix
        assert matrix.sum() == 3  # Total of 3 items

    def test_calculate_with_single_score_value(self):
        """Test calculation when all items have the same score (edge case)."""
        items = [
            EvaluationItem("Q1", "R1", human_score=5, judge_score=5),
            EvaluationItem("Q2", "R2", human_score=5, judge_score=5),
        ]

        metrics = AlignmentMetrics(score_range=ScoreRange.FIVE_POINT)

        # This can cause warnings/nan in kappa calculation - that's expected
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = metrics.calculate(items)

        # When all scores are identical, agreement should be perfect
        assert results.agreement_rate == 1.0
        assert results.sample_size == 2
        # Kappa might be nan or 1.0 depending on implementation
        assert np.isnan(results.kappa_score) or results.kappa_score == 1.0
