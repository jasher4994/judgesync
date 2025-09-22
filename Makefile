.PHONY: help install test lint format clean pre-commit build check-dist upload-test upload

help:
	@echo "Available commands:"
	@echo "  make install    Install package in development mode"
	@echo "  make test      Run tests with coverage"
	@echo "  make lint      Run linting checks"
	@echo "  make format    Format code with ruff"
	@echo "  make clean     Remove build artifacts"
	@echo "  make pre-commit Run pre-commit on all files"
	@echo "  make build     Build distribution packages"
	@echo "  make check-dist Check distribution package integrity"
	@echo "  make upload-test Upload to Test PyPI"
	@echo "  make upload    Upload to production PyPI"

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

build: clean
	python3 -m build . --outdir dist/

check-dist: build
	python3 -m twine check dist/*

upload-test: check-dist
	python3 -m twine upload --repository testpypi dist/*

upload: check-dist
	@echo "WARNING: This will upload to PRODUCTION PyPI!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		python3 -m twine upload dist/*; \
	else \
		echo "Upload cancelled."; \
	fi
