"""LLM Judge functionality using Azure OpenAI."""

import os
from typing import List, Optional, Dict, Any
from openai import AzureOpenAI
from dotenv import load_dotenv

from .types import EvaluationItem, ScoreRange


class Judge:
    """Manages LLM judge interactions using Azure OpenAI."""
    
    def __init__(
        self,
        system_prompt: str,
        score_range: ScoreRange = ScoreRange.FIVE_POINT,
        azure_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        deployment_name: Optional[str] = None,
        api_version: str = "2024-02-01",
    ):
        """Initialize the Judge with Azure OpenAI configuration.
        
        Args:
            system_prompt: The system prompt that defines how the judge should evaluate.
            score_range: The expected scoring range.
            azure_endpoint: Azure OpenAI endpoint (or set AZURE_OPENAI_ENDPOINT env var).
            api_key: Azure OpenAI API key (or set AZURE_OPENAI_API_KEY env var).
            deployment_name: Azure deployment name (or set AZURE_OPENAI_DEPLOYMENT env var).
            api_version: Azure OpenAI API version.
        """
        # Load environment variables
        load_dotenv()
        
        self.system_prompt = system_prompt
        self.score_range = score_range
        
        # Get Azure configuration from parameters or environment
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.deployment_name = deployment_name or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        
        if not all([self.azure_endpoint, self.api_key, self.deployment_name]):
            raise ValueError(
                "Azure OpenAI configuration incomplete. Please provide "
                "azure_endpoint, api_key, and deployment_name either as "
                "parameters or as environment variables."
            )
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            api_version=api_version
        )
        
        # Store for tracking
        self.last_response: Optional[Dict[str, Any]] = None
    
    def score_item(self, item: EvaluationItem) -> float:
        """Score a single evaluation item using the LLM judge.
        
        Args:
            item: The evaluation item to score.
            
        Returns:
            The numerical score from the judge.
            
        Raises:
            ValueError: If the response cannot be parsed as a valid score.
        """
        # Build the user prompt
        user_prompt = self._build_user_prompt(item)
        
        # Get response from Azure OpenAI
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": user_prompt}
            ],
        )
        
        # Store the full response for debugging
        self.last_response = response.model_dump()
        
        # Extract and parse the score
        response_text = response.choices[0].message.content.strip()
        score = self._parse_score(response_text)
        
        # Validate the score is in range
        min_val, max_val = self.score_range.value
        if not min_val <= score <= max_val:
            raise ValueError(
                f"Judge returned score {score} outside of expected range "
                f"[{min_val}, {max_val}]"
            )
        
        return score
    
    def score_items(self, items: List[EvaluationItem]) -> List[EvaluationItem]:
        """Score multiple evaluation items.
        
        Args:
            items: List of evaluation items to score.
            
        Returns:
            The same items with judge_score populated.
        """
        for item in items:
            try:
                item.judge_score = self.score_item(item)
            except Exception as e:
                print(f"Error scoring item: {e}")
                # Continue with other items even if one fails
                continue
        
        return items
    
    def _build_system_prompt(self) -> str:
        """Build the complete system prompt including score range instructions.
        
        Returns:
            The formatted system prompt.
        """
        min_val, max_val = self.score_range.value
        
        range_instruction = f"\nYou must respond with ONLY a number between {min_val} and {max_val}."
        
        return f"{self.system_prompt}{range_instruction}"
    
    def _build_user_prompt(self, item: EvaluationItem) -> str:
        """Build the user prompt for evaluation.
        
        Args:
            item: The evaluation item.
            
        Returns:
            The formatted user prompt.
        """
        return f"Question: {item.question}\n\nResponse: {item.response}"
    
    def _parse_score(self, response_text: str) -> float:
        """Parse the score from the judge's response.
        
        Args:
            response_text: The raw response from the judge.
            
        Returns:
            The parsed numerical score.
            
        Raises:
            ValueError: If the response cannot be parsed as a number.
        """
        # Try to extract just a number from the response
        # Handle cases where judge might add explanation
        try:
            # Take the first word/token and try to parse it
            first_token = response_text.split()[0] if response_text else ""
            return float(first_token)
        except (ValueError, IndexError):
            # If that fails, try the whole response
            try:
                return float(response_text)
            except ValueError:
                raise ValueError(f"Could not parse score from response: '{response_text}'")
    
    def update_system_prompt(self, new_prompt: str) -> None:
        """Update the system prompt for the judge.
        
        Args:
            new_prompt: The new system prompt.
        """
        self.system_prompt = new_prompt