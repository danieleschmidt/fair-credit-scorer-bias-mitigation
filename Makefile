.PHONY: help install install-dev test lint format security type-check build clean dev docs serve-docs coverage benchmark mutation all
.DEFAULT_GOAL := help

# Variables
PYTHON := python
PIP := pip
PACKAGE_NAME := fair_credit_scorer_bias_mitigation
SRC_DIR := src
TEST_DIR := tests
COVERAGE_THRESHOLD := 80

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install package dependencies
	$(PIP) install -e .

install-dev: ## Install package with development dependencies
	$(PIP) install -e .[dev,test,lint,docs]
	pre-commit install

test: ## Run tests with coverage
	pytest $(TEST_DIR) \
		--cov=$(SRC_DIR) \
		--cov-report=html \
		--cov-report=term-missing \
		--cov-fail-under=$(COVERAGE_THRESHOLD) \
		-v

test-fast: ## Run tests without coverage (faster)
	pytest $(TEST_DIR) -v

test-parallel: ## Run tests in parallel
	pytest $(TEST_DIR) -n auto --dist=worksteal

lint: ## Run linting (ruff)
	ruff check $(SRC_DIR) $(TEST_DIR)

lint-fix: ## Run linting with auto-fix
	ruff check $(SRC_DIR) $(TEST_DIR) --fix

format: ## Format code with black
	black $(SRC_DIR) $(TEST_DIR)

format-check: ## Check if code is formatted correctly
	black --check $(SRC_DIR) $(TEST_DIR)

security: ## Run security checks with bandit
	bandit -r $(SRC_DIR) -f json -o security-report.json
	bandit -r $(SRC_DIR)

type-check: ## Run type checking with mypy
	mypy $(SRC_DIR) --ignore-missing-imports

quality: lint format-check type-check security ## Run all quality checks

build: ## Build package
	$(PYTHON) -m build

clean: ## Clean build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -delete
	find . -name "*.pyc" -delete

dev: install-dev ## Setup development environment
	@echo "Development environment ready!"
	@echo "Run 'make test' to run tests"
	@echo "Run 'make lint' to run linting"

docs: ## Build documentation
	mkdocs build

serve-docs: ## Serve documentation locally
	mkdocs serve

coverage: ## Generate and open coverage report
	pytest $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=html
	@echo "Coverage report generated in htmlcov/"
	@which open > /dev/null && open htmlcov/index.html || echo "Open htmlcov/index.html in your browser"

benchmark: ## Run performance benchmarks
	pytest $(TEST_DIR) -m performance -v

architecture: ## Generate architecture diagram
	$(PYTHON) -m $(SRC_DIR).architecture_review

mutation: ## Run mutation testing
	mutmut run

mutation-show: ## Show mutation testing results
	mutmut show

mutation-html: ## Generate mutation testing HTML report
	mutmut html

all: clean install-dev quality test build ## Run complete CI pipeline locally

# Docker commands
docker-build: ## Build Docker image
	docker build -t $(PACKAGE_NAME):latest .

docker-run: ## Run Docker container
	docker run --rm -it $(PACKAGE_NAME):latest

docker-test: ## Run tests in Docker
	docker run --rm $(PACKAGE_NAME):latest make test

# Git hooks
pre-commit: ## Run pre-commit hooks
	pre-commit run --all-files

# Release commands
release-patch: ## Create patch release
	bump2version patch

release-minor: ## Create minor release
	bump2version minor

release-major: ## Create major release
	bump2version major