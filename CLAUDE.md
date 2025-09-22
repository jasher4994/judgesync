# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JudgeSync is a Python package for calibrating LLM judges to align with human evaluations. It helps minimize bias in LLM-as-a-judge workflows by finding optimal judge configurations (prompt, model, temperature) that best align with human judgments.

## Development Commands

### Essential Commands
- `make install` - Install package in development mode with dev dependencies and pre-commit hooks
- `make test` - Run tests with coverage: `pytest tests/ -v --cov=judgesync --cov-report=term-missing`
- `make lint` - Run linting checks: `ruff check .` and `mypy judgesync/`
- `make format` - Format code with ruff: `ruff check --fix .` and `ruff format .`
- `make pre-commit` - Run pre-commit hooks on all files
- `make clean` - Remove build artifacts and cache directories

### Package Management
- Built with setuptools, configured in `pyproject.toml`
- Uses ruff for linting and formatting
- MyPy for type checking
- Pytest for testing with coverage
- Pre-commit hooks for code quality

## Architecture

### Core Components

**AlignmentTracker** (`judgesync/alignment.py`): Main class for tracking alignment between human and LLM judge scores. Handles data loading, judge configuration, and evaluation.

**JudgeComparison** (`judgesync/comparison.py`): Manages comparison of multiple judges with different configurations (prompts, models, temperatures). Includes `JudgeConfig` and `ComparisonResults` classes.

**Judge** (`judgesync/judge.py`): Interface to Azure OpenAI for LLM evaluation. Handles API calls and response parsing.

**DataLoader** (`judgesync/data_loader.py`): Loads and validates evaluation data from CSV files with required columns: `question`, `response`, `human_score`.

**AlignmentMetrics** (`judgesync/metrics.py`): Calculates statistical metrics including Cohen's Kappa, correlation, and agreement rates.

**Types** (`judgesync/types.py`): Core data structures including `ScoreRange`, `EvaluationItem`, and `AlignmentResults`.

### Key Features
- Support for multiple scoring scales (binary, 5-point, 10-point, percentage)
- Async batch processing for efficiency
- Statistical alignment metrics (Cohen's Kappa, correlation, agreement rate)
- Visualization and comparison charts
- Azure OpenAI integration

## Configuration

### Environment Variables
Required for Azure OpenAI integration:
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_DEPLOYMENT`

### Code Style
- Line length: 88 characters
- Target Python version: 3.8+
- Uses ruff for linting with pycodestyle, pyflakes, isort, pep8-naming, and pyupgrade rules
- Double quotes for strings
- Space indentation

## Testing

Tests are located in `tests/` directory with coverage for all main components:
- `test_alignment.py` - AlignmentTracker functionality
- `test_comparison.py` - Judge comparison features
- `test_data_loader.py` - Data loading and validation
- `test_judge.py` - LLM judge interface
- `test_metrics.py` - Statistical calculations
- `test_types.py` - Type definitions and validation

Run individual test files: `pytest tests/test_alignment.py -v`

## Dependencies

Core dependencies:
- pandas, numpy, scikit-learn (data processing and metrics)
- openai, azure-identity (Azure OpenAI integration)
- python-dotenv (environment configuration)

Development dependencies in `[project.optional-dependencies]` include pytest, ruff, mypy, and pre-commit.
