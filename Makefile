.PHONY: install test lint format typecheck audit build check clean help

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------
PYTHON     := python
POETRY     := poetry
SRC        := src/medrisk_health_analytics
TESTS      := tests

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
install:  ## Install all dependencies (including dev)
	$(POETRY) install

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
test:  ## Run unit tests with coverage report
	$(POETRY) run pytest --cov=$(SRC) --cov-report=term-missing --cov-report=html

test-fast:  ## Run tests without coverage (faster)
	$(POETRY) run pytest -x

# ---------------------------------------------------------------------------
# Linting and formatting
# ---------------------------------------------------------------------------
lint:  ## Run ruff linter
	$(POETRY) run ruff check $(SRC) $(TESTS)

format:  ## Run black formatter
	$(POETRY) run black $(SRC) $(TESTS)

format-check:  ## Check formatting without applying changes (for CI)
	$(POETRY) run black --check $(SRC) $(TESTS)

# ---------------------------------------------------------------------------
# Type checking
# ---------------------------------------------------------------------------
typecheck:  ## Run mypy type checker
	$(POETRY) run mypy $(SRC)

audit:  ## Audit installed dependencies for known vulnerabilities
	$(POETRY) run pip-audit --local --skip-editable

build:  ## Build wheel and source distribution
	$(POETRY) check --lock
	$(POETRY) build

# ---------------------------------------------------------------------------
# All checks (CI equivalent)
# ---------------------------------------------------------------------------
check: lint format-check typecheck test build  ## Run all quality and packaging checks

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
clean:  ## Remove build artifacts and cache files
	rm -rf .pytest_cache htmlcov .coverage dist build
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
