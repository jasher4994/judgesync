"""Tests for the data_loader module."""

import csv
import tempfile
from pathlib import Path

import pytest

from judgesync.data_loader import DataLoader
from judgesync.types import ScoreRange


class TestDataLoader:
    """Test the DataLoader class."""

    def test_init_with_default_score_range(self):
        """Test initialization with default score range."""
        loader = DataLoader()
        assert loader.score_range == ScoreRange.FIVE_POINT
        assert loader.items == []
        assert len(loader) == 0

    def test_init_with_custom_score_range(self):
        """Test initialization with custom score range."""
        loader = DataLoader(score_range=ScoreRange.PERCENTAGE)
        assert loader.score_range == ScoreRange.PERCENTAGE

    def test_add_item_minimal(self):
        """Test adding an item with minimal data."""
        loader = DataLoader()
        loader.add_item(question="What is 2+2?", response="4")

        assert len(loader) == 1
        assert loader.items[0].question == "What is 2+2?"
        assert loader.items[0].response == "4"
        assert loader.items[0].human_score is None

    def test_add_item_with_score(self):
        """Test adding an item with a human score."""
        loader = DataLoader(score_range=ScoreRange.FIVE_POINT)
        loader.add_item(
            question="Test question", response="Test response", human_score=4.0
        )

        assert len(loader) == 1
        assert loader.items[0].human_score == 4.0

    def test_add_item_with_invalid_score_raises_error(self):
        """Test that adding an invalid score raises an error."""
        loader = DataLoader(score_range=ScoreRange.FIVE_POINT)

        with pytest.raises(ValueError, match="outside the expected range"):
            loader.add_item(
                question="Q",
                response="R",
                human_score=10.0,  # Invalid for 1-5 range
            )

    def test_load_from_csv_success(self):
        """Test successfully loading data from CSV."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["question", "response", "human_score"])
            writer.writerow(["What is 2+2?", "4", "5"])
            writer.writerow(["Capital of France?", "Paris", "4"])
            temp_path = f.name

        try:
            loader = DataLoader(score_range=ScoreRange.FIVE_POINT)
            loader.load_from_csv(temp_path)

            assert len(loader) == 2
            assert loader.items[0].question == "What is 2+2?"
            assert loader.items[0].response == "4"
            assert loader.items[0].human_score == 5.0
            assert loader.items[1].question == "Capital of France?"
            assert loader.items[1].human_score == 4.0
        finally:
            Path(temp_path).unlink()  # Clean up temp file

    def test_load_from_csv_with_metadata(self):
        """Test loading CSV with metadata columns."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(
                ["question", "response", "human_score", "category", "difficulty"]
            )
            writer.writerow(["What is 2+2?", "4", "5", "math", "easy"])
            temp_path = f.name

        try:
            loader = DataLoader(score_range=ScoreRange.FIVE_POINT)
            loader.load_from_csv(temp_path, metadata_cols=["category", "difficulty"])

            assert len(loader) == 1
            assert loader.items[0].metadata == {
                "category": "math",
                "difficulty": "easy",
            }
        finally:
            Path(temp_path).unlink()

    def test_load_from_csv_file_not_found(self):
        """Test that loading from non-existent file raises error."""
        loader = DataLoader()

        with pytest.raises(FileNotFoundError):
            loader.load_from_csv("non_existent_file.csv")

    def test_load_from_csv_missing_columns(self):
        """Test that loading CSV with missing columns raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["question", "response"])  # Missing human_score
            writer.writerow(["What is 2+2?", "4"])
            temp_path = f.name

        try:
            loader = DataLoader()
            with pytest.raises(ValueError, match="Missing required columns"):
                loader.load_from_csv(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_get_items_with_human_scores(self):
        """Test filtering items with human scores."""
        loader = DataLoader()

        # Add items with and without scores
        loader.add_item("Q1", "R1", human_score=5.0)
        loader.add_item("Q2", "R2")  # No score
        loader.add_item("Q3", "R3", human_score=3.0)

        items_with_scores = loader.get_items_with_human_scores()

        assert len(items_with_scores) == 2
        assert all(item.human_score is not None for item in items_with_scores)

    def test_clear(self):
        """Test clearing all items."""
        loader = DataLoader()
        loader.add_item("Q1", "R1")
        loader.add_item("Q2", "R2")

        assert len(loader) == 2
        loader.clear()
        assert len(loader) == 0
        assert loader.items == []

    def test_repr(self):
        """Test string representation of DataLoader."""
        loader = DataLoader(score_range=ScoreRange.PERCENTAGE)
        loader.add_item("Q1", "R1", human_score=80)
        loader.add_item("Q2", "R2")

        repr_str = repr(loader)
        assert "PERCENTAGE" in repr_str
        assert "total_items=2" in repr_str
        assert "items_with_human_scores=1" in repr_str
