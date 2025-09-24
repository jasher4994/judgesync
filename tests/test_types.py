"""Tests for the types module."""

from judgesync.types import AlignmentResults, EvaluationItem, ScoreRange


class TestScoreRange:
    """Test the ScoreRange enum."""

    def test_score_ranges_have_correct_values(self):
        """Test that score ranges have the expected min/max values."""
        assert ScoreRange.BINARY.value == (0, 1)
        assert ScoreRange.FIVE_POINT.value == (1, 5)
        assert ScoreRange.PERCENTAGE.value == (0, 100)
        assert ScoreRange.TEN_POINT.value == (1, 10)

    def test_all_score_ranges_accessible(self):
        """Test that all score ranges can be accessed."""
        ranges = [
            ScoreRange.BINARY,
            ScoreRange.FIVE_POINT,
            ScoreRange.PERCENTAGE,
            ScoreRange.TEN_POINT,
        ]
        assert len(ranges) == 4


class TestEvaluationItem:
    """Test the EvaluationItem dataclass."""

    def test_create_minimal_item(self):
        """Test creating an item with just required fields."""
        item = EvaluationItem(question="What is 2+2?", response="4")
        assert item.question == "What is 2+2?"
        assert item.response == "4"
        assert item.human_score is None
        assert item.judge_score is None
        assert item.metadata is None

    def test_create_full_item(self):
        """Test creating an item with all fields."""
        item = EvaluationItem(
            question="What is 2+2?",
            response="4",
            human_score=5.0,
            judge_score=4.5,
            metadata={"category": "math"},
        )
        assert item.human_score == 5.0
        assert item.judge_score == 4.5
        assert item.metadata == {"category": "math"}

    def test_has_both_scores(self):
        """Test the has_both_scores method."""
        item1 = EvaluationItem(question="Q", response="R")
        assert not item1.has_both_scores()

        item2 = EvaluationItem(question="Q", response="R", human_score=5.0)
        assert not item2.has_both_scores()

        item3 = EvaluationItem(question="Q", response="R", judge_score=4.0)
        assert not item3.has_both_scores()

        item4 = EvaluationItem(
            question="Q", response="R", human_score=5.0, judge_score=4.0
        )
        assert item4.has_both_scores()


class TestAlignmentResults:
    """Test the AlignmentResults dataclass."""

    def test_create_minimal_results(self):
        """Test creating results with minimal data."""
        results = AlignmentResults(
            kappa_score=0.75, agreement_rate=0.85, sample_size=100
        )
        assert results.kappa_score == 0.75
        assert results.agreement_rate == 0.85
        assert results.sample_size == 100
        assert results.raw_scores is None

    def test_create_full_results(self):
        """Test creating results with all fields."""
        raw_scores = [(5, 4), (3, 3), (2, 1)]
        results = AlignmentResults(
            kappa_score=0.65, agreement_rate=0.75, sample_size=3, raw_scores=raw_scores
        )
        assert results.raw_scores == raw_scores
        assert len(results.raw_scores) == 3
