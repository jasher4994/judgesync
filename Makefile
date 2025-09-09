.PHONY: help install test lint format clean

help:
    @echo "Available commands:"
    @echo "  make install    Install package in development mode"
    @echo "  make test      Run tests with coverage"
    @echo "  make lint      Run linting checks"
    @echo "  make format    Format code with ruff"
    @echo "  make clean     Remove build artifacts"
    @echo "  make pre-commit Run pre-commit on all files"

install:
    pip install -e ".[dev]"
    pre-commit install

test:
    pytest tests/ -v --cov=judgesync --cov-report=term-missing

lint:
    ruff check .
    mypy judgesync/

format:
    ruff check --fix .
    ruff format .

clean:
    rm -rf build/
    rm -rf dist/
    rm -rf *.egg-info
    rm -rf .pytest_cache/
    rm -rf .ruff_cache/
    rm -rf .mypy_cache/
    find . -type d -name __pycache__ -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete

pre-commit:
    pre-commit run --all-files