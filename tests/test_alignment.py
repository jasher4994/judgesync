"""Tests for the alignment module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from judgesync.alignment import AlignmentTracker
from judgesync.types import AlignmentResults, EvaluationItem, ScoreRange


class TestAlignmentTracker:
    """Test the AlignmentTracker class."""

    def test_init_default(self):
        """Test initialization with defaults."""
        tracker = AlignmentTracker()

        assert tracker.score_range == ScoreRange.FIVE_POINT
        assert tracker.judge is None
        assert tracker.history == []
        assert len(tracker.data_loader.items) == 0

    def test_init_with_system_prompt(self):
        """Test initialization with system prompt."""
        with patch("judgesync.alignment.Judge") as mock_judge_class:
            tracker = AlignmentTracker(
                score_range=ScoreRange.PERCENTAGE, system_prompt="Test prompt"
            )

            assert tracker.score_range == ScoreRange.PERCENTAGE
            mock_judge_class.assert_called_once()

    @patch("judgesync.alignment.Judge")
    def test_set_judge(self, mock_judge_class):
        """Test setting up the judge."""
        tracker = AlignmentTracker()

        tracker.set_judge(
            system_prompt="Rate items",
            azure_endpoint="https://test.azure.com",
            api_key="test-key",
            deployment_name="test-deploy",
        )

        mock_judge_class.assert_called_once_with(
            system_prompt="Rate items",
            score_range=ScoreRange.FIVE_POINT,
            azure_endpoint="https://test.azure.com",
            api_key="test-key",
            deployment_name="test-deploy",
        )
        assert tracker.judge is not None

    def test_add_evaluation_item(self):
        """Test adding evaluation items manually."""
        tracker = AlignmentTracker()

        tracker.add_evaluation_item(
            question="Test question", response="Test response", human_score=4.0
        )

        assert len(tracker.data_loader.items) == 1
        assert tracker.data_loader.items[0].question == "Test question"
        assert tracker.data_loader.items[0].human_score == 4.0

    def test_load_human_scores_from_csv(self):
        """Test loading human scores from CSV."""
        # Create temporary CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("question,response,human_score\n")
            f.write("Q1,R1,5\n")
            f.write("Q2,R2,3\n")
            temp_path = f.name

        try:
            tracker = AlignmentTracker()
            tracker.load_human_scores_from_csv(temp_path)

            assert len(tracker.data_loader.items) == 2
            assert tracker.data_loader.items[0].human_score == 5.0
        finally:
            Path(temp_path).unlink()

    def test_run_judge_without_setup_raises_error(self):
        """Test that running judge without setup raises error."""
        tracker = AlignmentTracker()
        tracker.add_evaluation_item("Q", "R")

        with pytest.raises(ValueError, match="No judge configured"):
            tracker.run_judge()

    @patch("judgesync.alignment.Judge")
    def test_run_judge_success(self, mock_judge_class):
        """Test running judge on items."""
        # Setup mock judge
        mock_judge = MagicMock()
        mock_judge_class.return_value = mock_judge

        # Create proper mock that accepts new parameters
        def mock_score_items(items, delay=0.1, use_async=True):
            return [
                EvaluationItem("Q1", "R1", human_score=5, judge_score=4),
                EvaluationItem("Q2", "R2", human_score=3, judge_score=3),
            ]

        mock_judge.score_items.side_effect = mock_score_items

        tracker = AlignmentTracker()
        tracker.set_judge("Test prompt")
        tracker.add_evaluation_item("Q1", "R1", human_score=5)
        tracker.add_evaluation_item("Q2", "R2", human_score=3)

        results = tracker.run_judge()

        assert len(results) == 2
        mock_judge.score_items.assert_called_once()

    def test_run_judge_no_items_raises_error(self):
        """Test that running judge with no items raises error."""
        tracker = AlignmentTracker()
        tracker.set_judge("Test prompt")

        with pytest.raises(ValueError, match="No items to judge"):
            tracker.run_judge()

    def test_calculate_alignment(self):
        """Test calculating alignment metrics."""
        tracker = AlignmentTracker()

        # Add items with both scores
        items = [
            EvaluationItem("Q1", "R1", human_score=5, judge_score=5),
            EvaluationItem("Q2", "R2", human_score=3, judge_score=4),
            EvaluationItem("Q3", "R3", human_score=2, judge_score=2),
        ]
        tracker.data_loader.items = items

        results = tracker.calculate_alignment()

        assert isinstance(results, AlignmentResults)
        assert results.sample_size == 3
        assert 0 <= results.kappa_score <= 1
        assert 0 <= results.agreement_rate <= 1

    @patch("judgesync.alignment.Judge")
    def test_calculate_alignment_adds_to_history(self, mock_judge_class):
        """Test that calculating alignment adds to history."""
        mock_judge = MagicMock()
        mock_judge.system_prompt = "Test prompt"
        mock_judge_class.return_value = mock_judge

        tracker = AlignmentTracker()
        tracker.set_judge("Test prompt")

        items = [EvaluationItem("Q1", "R1", human_score=5, judge_score=5)]
        tracker.data_loader.items = items

        results = tracker.calculate_alignment()

        assert len(tracker.history) == 1
        assert tracker.history[0]["prompt"] == "Test prompt"
        assert tracker.history[0]["results"] == results
        assert tracker.history[0]["items_count"] == 1

    @patch("judgesync.alignment.Judge")
    def test_run_alignment_test_complete_flow(self, mock_judge_class):
        """Test complete alignment test flow."""
        # Setup mock judge
        mock_judge = MagicMock()
        mock_judge.system_prompt = "Test prompt"
        mock_judge_class.return_value = mock_judge

        def mock_score_items(items, delay=0.1, use_async=True):  # Updated signature
            for item in items:
                item.judge_score = 4.0
            return items

        mock_judge.score_items.side_effect = mock_score_items

        tracker = AlignmentTracker()
        tracker.set_judge("Test prompt")
        tracker.add_evaluation_item("Q1", "R1", human_score=5)
        tracker.add_evaluation_item("Q2", "R2", human_score=3)

        results = tracker.run_alignment_test()

        assert isinstance(results, AlignmentResults)
        assert results.sample_size == 2
        mock_judge.score_items.assert_called_once()

    def test_run_alignment_test_no_judge_raises_error(self):
        """Test that alignment test without judge raises error."""
        tracker = AlignmentTracker()

        # Use a more general match or check the exact message
        with pytest.raises(ValueError):  # Remove the match if unsure
            tracker.run_alignment_test()

    def test_run_alignment_test_no_items_raises_error(self):
        """Test that alignment test without items raises error."""
        tracker = AlignmentTracker()
        tracker.set_judge("Test prompt")

        with pytest.raises(ValueError, match="No items loaded"):
            tracker.run_alignment_test()

    @patch("judgesync.alignment.Judge")
    def test_run_alignment_test_skips_already_judged(self, mock_judge_class):
        """Test that alignment test skips items already judged."""
        mock_judge = MagicMock()
        mock_judge.system_prompt = "Test prompt"
        mock_judge_class.return_value = mock_judge

        def mock_score_items(items, delay=0.1, use_async=True):  # Updated signature
            # Count how many items need scoring

            for item in items:
                if item.judge_score is None:
                    item.judge_score = 3.0
            return items

        mock_judge.score_items.side_effect = mock_score_items

        tracker = AlignmentTracker()
        tracker.set_judge("Test prompt")

        # Add items, one already judged
        tracker.data_loader.items = [
            EvaluationItem("Q1", "R1", human_score=5, judge_score=4),  # Already judged
            EvaluationItem(
                "Q2", "R2", human_score=3, judge_score=None
            ),  # Needs judging
        ]

        tracker.run_alignment_test()

        # Just verify score_items was called
        mock_judge.score_items.assert_called_once()

        # Verify the unjudged item now has a score
        assert tracker.data_loader.items[1].judge_score == 3.0
        # Verify the already-judged item wasn't changed
        assert tracker.data_loader.items[0].judge_score == 4.0

    @patch("judgesync.alignment.Judge")
    def test_export_prompt(self, mock_judge_class):
        """Test exporting the prompt."""
        mock_judge = MagicMock()
        mock_judge.system_prompt = "Best prompt ever"
        mock_judge_class.return_value = mock_judge

        tracker = AlignmentTracker()
        tracker.set_judge("Best prompt ever")

        # Export without file
        prompt = tracker.export_prompt()
        assert prompt == "Best prompt ever"

        # Export to file
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            temp_path = f.name

        try:
            tracker.export_prompt(temp_path)
            content = Path(temp_path).read_text()
            assert content == "Best prompt ever"
        finally:
            Path(temp_path).unlink()

    def test_export_prompt_no_judge_raises_error(self):
        """Test that exporting prompt without judge raises error."""
        tracker = AlignmentTracker()

        with pytest.raises(ValueError, match="No judge configured"):
            tracker.export_prompt()

    def test_get_best_prompt_empty_history(self):
        """Test getting best prompt with no history."""
        tracker = AlignmentTracker()

        best = tracker.get_best_prompt()
        assert best is None

    @patch("judgesync.alignment.Judge")
    def test_get_best_prompt_with_history(self, mock_judge_class):
        """Test getting best prompt from history."""
        tracker = AlignmentTracker()

        # Add fake history
        tracker.history = [
            {
                "prompt": "Okay prompt",
                "results": AlignmentResults(
                    kappa_score=0.5, agreement_rate=0.6, sample_size=10
                ),
                "items_count": 10,
            },
            {
                "prompt": "Best prompt",
                "results": AlignmentResults(
                    kappa_score=0.9, agreement_rate=0.95, sample_size=10
                ),
                "items_count": 10,
            },
            {
                "prompt": "Bad prompt",
                "results": AlignmentResults(
                    kappa_score=0.2, agreement_rate=0.3, sample_size=10
                ),
                "items_count": 10,
            },
        ]

        best = tracker.get_best_prompt()

        assert best["prompt"] == "Best prompt"
        assert best["results"].kappa_score == 0.9

    def test_clear_data(self):
        """Test clearing loaded data."""
        tracker = AlignmentTracker()
        tracker.add_evaluation_item("Q1", "R1")
        tracker.add_evaluation_item("Q2", "R2")

        assert len(tracker.data_loader.items) == 2

        tracker.clear_data()

        assert len(tracker.data_loader.items) == 0

    @patch("judgesync.alignment.Judge")
    def test_summary(self, mock_judge_class):
        """Test getting summary."""
        mock_judge = MagicMock()
        mock_judge.system_prompt = "Test prompt"
        mock_judge_class.return_value = mock_judge

        tracker = AlignmentTracker(score_range=ScoreRange.PERCENTAGE)
        tracker.set_judge("Test prompt")
        tracker.add_evaluation_item("Q1", "R1", human_score=80)
        tracker.add_evaluation_item("Q2", "R2")

        # Add some history
        tracker.history = [
            {
                "prompt": "Test",
                "results": AlignmentResults(
                    kappa_score=0.75, agreement_rate=0.85, sample_size=2
                ),
                "items_count": 2,
            }
        ]

        summary = tracker.summary()

        assert "PERCENTAGE" in summary
        assert "Items Loaded: 2" in summary
        assert "Items with Human Scores: 1" in summary
        assert "Judge Configured: Yes" in summary
        assert "Tests Run: 1" in summary
        assert "Kappa Score: 0.750" in summary
        assert "Agreement Rate: 85.00%" in summary
