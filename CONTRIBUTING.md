# Contributing to JudgeSync

Thanks for your interest in improving JudgeSync! We welcome issues, discussions, and pull requests. This guide will help you get set up and contribute effectively.

## 🧰 Local Development Setup

1. **Clone** the repository and create a virtual environment (Python 3.9+).
2. **Install** dependencies — include the development extras so tests and linters run:

   ```bash
   python -m pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks** (runs linting before every commit):

   ```bash
   pre-commit install
   ```

## 🏗️ Project Workflow

- **Code style & linting:** we use [Ruff](https://docs.astral.sh/ruff/) for linting/formatting and `mypy` for type checks.

  ```bash
  make lint    # ruff check + mypy
  make format  # ruff --fix + ruff format
  ```

- **Testing:** run the suite (with coverage) before opening a PR.

  ```bash
  make test
  ```

  Tests live under `tests/` and target the modules in `judgesync/`.

- **Documentation:** update relevant docstrings or markdown guides if your change affects behavior or developer experience.

## 🔁 Making Changes

1. **Branch** off `main` with a descriptive name (`feature/prompt-comparison`, `fix/data-loader-validation`, etc.).
2. Implement your changes, keeping functions small and well-typed.
3. **Write or update tests** to cover new behavior and edge cases.
4. Run `make lint` and `make test` locally; ensure everything passes.
5. Commit with clear messages (e.g., `fix: handle percentage bins in metrics`).

## 📬 Pull Requests

- Fill out the PR template (if present) and describe *why* the change is needed.
- Reference related issues using “Closes #123”.
- Include test results in the PR description (copy the relevant command outputs).
- Keep PRs focused—prefer several small PRs over one large one.
- Expect maintainer review; be responsive to feedback.

## 🧪 Useful Commands

| Task                 | Command                                   |
| -------------------- | ----------------------------------------- |
| Install dev deps     | `python -m pip install -e ".[dev]"`       |
| Run tests            | `make test`                               |
| Run lint/type checks | `make lint`                               |
| Auto-format          | `make format`                             |
| Clean artifacts      | `make clean`                              |
| Build package        | `make build`                              |
| Check distribution   | `make check-dist`                         |

## 📣 Communication

- Use GitHub issues for bugs, features, or questions.
- Tag maintainers (`@jasher4994`) for critical fixes.
- For significant changes, consider starting a discussion/issue first.

Thanks again for contributing—your improvements help keep JudgeSync aligned and reliable!
