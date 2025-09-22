"""Tests for the judge module."""

import os
from unittest.mock import MagicMock, patch

import pytest

from judgesync.judge import Judge
from judgesync.types import EvaluationItem, ScoreRange


class TestJudge:
    """Test the Judge class."""

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_DEPLOYMENT": "test-deployment",
        },
    )
    def test_init_with_env_vars(self):
        """Test initialization with environment variables."""
        judge = Judge(
            system_prompt="Rate the response", score_range=ScoreRange.FIVE_POINT
        )

        assert judge.system_prompt == "Rate the response"
        assert judge.score_range == ScoreRange.FIVE_POINT
        assert judge.azure_endpoint == "https://test.openai.azure.com/"
        assert judge.deployment_name == "test-deployment"

    def test_init_with_explicit_params(self):
        """Test initialization with explicit parameters."""
        judge = Judge(
            system_prompt="Rate the response",
            azure_endpoint="https://explicit.openai.azure.com/",
            api_key="explicit-key",
            deployment_name="explicit-deployment",
        )

        assert judge.azure_endpoint == "https://explicit.openai.azure.com/"
        assert judge.deployment_name == "explicit-deployment"

    @patch("judgesync.judge.load_dotenv")
    def test_init_missing_credentials_raises_error(self, mock_load_dotenv):
        """Test that missing credentials raise an error."""
        # Mock load_dotenv to do nothing (don't load .env file)
        mock_load_dotenv.return_value = None

        # Clear environment variables - Python 3.8 compatible syntax
        with patch.dict(
            os.environ,
            {
                "AZURE_OPENAI_ENDPOINT": "",
                "AZURE_OPENAI_API_KEY": "",
                "AZURE_OPENAI_DEPLOYMENT": "",
            },
            clear=True,
        ), pytest.raises(ValueError, match="Azure OpenAI configuration incomplete"):
            Judge(system_prompt="Rate the response")

    @patch("judgesync.judge.AzureOpenAI")
    def test_score_item_success(self, mock_azure_class):
        """Test successfully scoring an item."""
        # Setup mock
        mock_client = MagicMock()
        mock_azure_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "4"
        mock_response.model_dump.return_value = {"test": "data"}
        mock_client.chat.completions.create.return_value = mock_response

        # Create judge and score item
        judge = Judge(
            system_prompt="Rate 1-5",
            score_range=ScoreRange.FIVE_POINT,
            azure_endpoint="https://test.openai.azure.com/",
            api_key="test-key",
            deployment_name="test-deployment",
        )

        item = EvaluationItem(question="Test?", response="Answer")
        score = judge.score_item(item)

        assert score == 4.0
        assert judge.last_response == {"test": "data"}

        # Verify API was called correctly
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "test-deployment"

    @patch("judgesync.judge.AzureOpenAI")
    def test_score_item_with_text_in_response(self, mock_azure_class):
        """Test scoring when judge includes text with the number."""
        mock_client = MagicMock()
        mock_azure_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "4 - Good response"
        mock_response.model_dump.return_value = {}
        mock_client.chat.completions.create.return_value = mock_response

        judge = Judge(
            system_prompt="Rate 1-5",
            score_range=ScoreRange.FIVE_POINT,
            azure_endpoint="https://test.openai.azure.com/",
            api_key="test-key",
            deployment_name="test-deployment",
        )

        item = EvaluationItem(question="Test?", response="Answer")
        score = judge.score_item(item)

        assert score == 4.0  # Should extract the number

    @patch("judgesync.judge.AzureOpenAI")
    def test_score_item_invalid_response(self, mock_azure_class):
        """Test that invalid response raises error."""
        mock_client = MagicMock()
        mock_azure_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "not a number"
        mock_response.model_dump.return_value = {}
        mock_client.chat.completions.create.return_value = mock_response

        judge = Judge(
            system_prompt="Rate 1-5",
            score_range=ScoreRange.FIVE_POINT,
            azure_endpoint="https://test.openai.azure.com/",
            api_key="test-key",
            deployment_name="test-deployment",
        )

        item = EvaluationItem(question="Test?", response="Answer")

        with pytest.raises(ValueError, match="Could not parse score"):
            judge.score_item(item)

    @patch("judgesync.judge.AzureOpenAI")
    def test_score_item_out_of_range(self, mock_azure_class):
        """Test that out-of-range score raises error."""
        mock_client = MagicMock()
        mock_azure_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "10"  # Out of 1-5 range
        mock_response.model_dump.return_value = {}
        mock_client.chat.completions.create.return_value = mock_response

        judge = Judge(
            system_prompt="Rate 1-5",
            score_range=ScoreRange.FIVE_POINT,
            azure_endpoint="https://test.openai.azure.com/",
            api_key="test-key",
            deployment_name="test-deployment",
        )

        item = EvaluationItem(question="Test?", response="Answer")

        with pytest.raises(ValueError, match="outside of expected range"):
            judge.score_item(item)

    @patch("judgesync.judge.AzureOpenAI")
    def test_score_items_batch(self, mock_azure_class):
        """Test scoring multiple items."""
        mock_client = MagicMock()
        mock_azure_class.return_value = mock_client

        # Mock different scores for different items
        mock_response1 = MagicMock()
        mock_response1.choices = [MagicMock()]
        mock_response1.choices[0].message.content = "3"
        mock_response1.model_dump.return_value = {}

        mock_response2 = MagicMock()
        mock_response2.choices = [MagicMock()]
        mock_response2.choices[0].message.content = "5"
        mock_response2.model_dump.return_value = {}

        mock_client.chat.completions.create.side_effect = [
            mock_response1,
            mock_response2,
        ]

        judge = Judge(
            system_prompt="Rate 1-5",
            score_range=ScoreRange.FIVE_POINT,
            azure_endpoint="https://test.openai.azure.com/",
            api_key="test-key",
            deployment_name="test-deployment",
        )

        items = [
            EvaluationItem(question="Q1", response="R1"),
            EvaluationItem(question="Q2", response="R2"),
        ]

        scored_items = judge.score_items(items)

        assert scored_items[0].judge_score == 3.0
        assert scored_items[1].judge_score == 5.0
        assert mock_client.chat.completions.create.call_count == 2

    @patch("judgesync.judge.AzureOpenAI")
    def test_score_items_continues_on_error(self, mock_azure_class):
        """Test that batch scoring continues even if one item fails."""
        mock_client = MagicMock()
        mock_azure_class.return_value = mock_client

        # First call fails, second succeeds
        mock_response_good = MagicMock()
        mock_response_good.choices = [MagicMock()]
        mock_response_good.choices[0].message.content = "4"
        mock_response_good.model_dump.return_value = {}

        mock_client.chat.completions.create.side_effect = [
            Exception("API Error"),
            mock_response_good,
        ]

        judge = Judge(
            system_prompt="Rate 1-5",
            score_range=ScoreRange.FIVE_POINT,
            azure_endpoint="https://test.openai.azure.com/",
            api_key="test-key",
            deployment_name="test-deployment",
        )

        items = [
            EvaluationItem(question="Q1", response="R1"),
            EvaluationItem(question="Q2", response="R2"),
        ]

        scored_items = judge.score_items(items)

        assert scored_items[0].judge_score is None  # Failed
        assert scored_items[1].judge_score == 4.0  # Succeeded

    def test_build_system_prompt(self):
        """Test system prompt building with score range."""
        judge = Judge(
            system_prompt="Be strict",
            score_range=ScoreRange.FIVE_POINT,
            azure_endpoint="https://test.openai.azure.com/",
            api_key="test-key",
            deployment_name="test-deployment",
        )

        full_prompt = judge._build_system_prompt()

        assert "Be strict" in full_prompt
        assert "between 1 and 5" in full_prompt

    def test_build_user_prompt(self):
        """Test user prompt building."""
        judge = Judge(
            system_prompt="Rate",
            azure_endpoint="https://test.openai.azure.com/",
            api_key="test-key",
            deployment_name="test-deployment",
        )

        item = EvaluationItem(question="What is 2+2?", response="The answer is 4")

        user_prompt = judge._build_user_prompt(item)

        assert "What is 2+2?" in user_prompt
        assert "The answer is 4" in user_prompt

    def test_update_system_prompt(self):
        """Test updating the system prompt."""
        judge = Judge(
            system_prompt="Original prompt",
            azure_endpoint="https://test.openai.azure.com/",
            api_key="test-key",
            deployment_name="test-deployment",
        )

        assert judge.system_prompt == "Original prompt"

        judge.update_system_prompt("New prompt")
        assert judge.system_prompt == "New prompt"
