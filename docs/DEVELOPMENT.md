# Developer Guide

This guide provides comprehensive instructions for setting up, developing, and contributing to the Fair Credit Scorer: Bias Mitigation project.

## Quick Start

### Prerequisites

- Python 3.8+ (recommended: 3.11)
- Git
- Docker (optional, for containerized development)
- VS Code (recommended IDE)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd fair-credit-scorer-bias-mitigation
   ```

2. **Set up Python environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install --upgrade pip setuptools wheel
   ```

3. **Install dependencies**:
   ```bash
   pip install -e .[dev]
   ```

4. **Set up pre-commit hooks**:
   ```bash
   pre-commit install
   ```

5. **Verify installation**:
   ```bash
   python -m src.run_tests
   fairness-eval --help
   ```

## Development Environment Options

### Option 1: DevContainer (Recommended)

For the most consistent development experience:

1. **Prerequisites**: VS Code + Dev Containers extension + Docker
2. **Setup**: Open project in VS Code → "Reopen in Container"
3. **Benefits**: 
   - Identical environment for all developers
   - Pre-configured tools and extensions
   - Isolated from host system

### Option 2: Local Development

For direct local development:

1. **Follow Quick Start instructions above**
2. **Install VS Code extensions**:
   ```bash
   code --install-extension ms-python.python
   code --install-extension charliermarsh.ruff
   code --install-extension ms-python.black-formatter
   # See .vscode/extensions.json for full list
   ```

### Option 3: Docker Compose

For containerized services:

```bash
docker-compose up -d
docker-compose exec app bash
```

## Project Structure

```
fair-credit-scorer-bias-mitigation/
├── .devcontainer/          # DevContainer configuration
├── .github/                # GitHub workflows and templates
├── .vscode/                # VS Code settings and tasks
├── architecture/           # Generated architecture diagrams
├── config/                 # Configuration files
├── docs/                   # Documentation
│   ├── adr/               # Architecture Decision Records
│   ├── guides/            # User and developer guides
│   └── runbooks/          # Operational procedures
├── src/                    # Source code
│   ├── fair_credit_scorer_bias_mitigation/  # Main package
│   ├── architecture_review.py
│   ├── baseline_model.py
│   ├── bias_mitigator.py
│   ├── data_loader_preprocessor.py
│   ├── evaluate_fairness.py
│   ├── fairness_metrics.py
│   └── repo_hygiene_bot.py
├── tests/                  # Test suite
│   ├── integration/       # Integration tests
│   ├── performance/       # Performance tests
│   └── e2e/              # End-to-end tests
├── requirements.txt        # Production dependencies
├── requirements-dev.txt    # Development dependencies
├── pyproject.toml         # Project configuration
└── Makefile              # Common development tasks
```

## Development Workflow

### 1. Feature Development

1. **Create feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes** following code standards

3. **Run quality checks**:
   ```bash
   make lint        # Code linting
   make typecheck   # Type checking
   make security    # Security scan
   make test        # Run tests
   make coverage    # Test coverage
   ```

4. **Commit changes**:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

5. **Push and create PR**:
   ```bash
   git push origin feature/your-feature-name
   ```

### 2. Testing

#### Unit Tests
```bash
pytest tests/ -v                    # All tests
pytest tests/test_fairness_metrics.py  # Specific module
pytest -k "test_demographic_parity"    # Specific test pattern
```

#### Integration Tests
```bash
pytest tests/integration/ -v
```

#### Performance Tests
```bash
pytest tests/performance/ -v --benchmark-only
```

#### Coverage Report
```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

### 3. Code Quality

#### Linting and Formatting
```bash
ruff check src tests           # Linting
ruff check src tests --fix     # Auto-fix issues
black src tests                # Code formatting
```

#### Type Checking
```bash
mypy src                       # Type checking
```

#### Security Scanning
```bash
bandit -r src                  # Security scan
```

#### Pre-commit Hooks
All quality checks run automatically on commit via pre-commit hooks.

## Code Standards

### Python Code Style

- **Line length**: 88 characters (Black default)
- **Import sorting**: Ruff with Black profile
- **Type hints**: Required for public APIs
- **Docstrings**: Google style for all public functions/classes

Example function:
```python
def calculate_demographic_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray, 
    sensitive_features: np.ndarray
) -> Dict[str, float]:
    """Calculate demographic parity metrics.
    
    Args:
        y_true: True binary labels (0/1)
        y_pred: Predicted binary labels (0/1)
        sensitive_features: Protected group indicators
        
    Returns:
        Dictionary containing demographic parity metrics
        
    Raises:
        ValueError: If arrays have different lengths
    """
```

### Testing Standards

- **Coverage target**: 80%+ for new code
- **Test naming**: `test_<function>_<scenario>_<expected>`
- **Fixtures**: Use pytest fixtures for common test data
- **Mocking**: Mock external dependencies

Example test:
```python
def test_demographic_parity_equal_groups_returns_zero(sample_data):
    """Test demographic parity with equal group outcomes."""
    y_true, y_pred, sensitive_features = sample_data
    result = calculate_demographic_parity(y_true, y_pred, sensitive_features)
    assert abs(result['demographic_parity_difference']) < 0.01
```

### Git Commit Standards

Use conventional commits:
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Test additions or changes
- `chore:` Maintenance tasks

## Available Commands

### Make Commands
```bash
make help          # Show all available commands
make install       # Install dependencies
make test          # Run tests
make lint          # Run linting
make format        # Format code
make typecheck     # Type checking
make security      # Security scan
make coverage      # Test coverage report
make docs          # Build documentation
make clean         # Clean build artifacts
make all           # Run all quality checks
```

### Direct Commands
```bash
fairness-eval                    # Run fairness evaluation CLI
repo-hygiene-bot                 # Run repository hygiene bot
python -m src.run_tests          # Run test suite
python -m src.architecture_review # Generate architecture diagrams
```

## Debugging

### VS Code Debugging

Pre-configured launch configurations:
- **Run Fairness Evaluation CLI**: Debug the main CLI with different parameters
- **Debug Tests**: Debug specific test files
- **Repository Hygiene Bot**: Debug the repo hygiene automation

### Common Issues

#### Import Errors
Ensure PYTHONPATH includes src directory:
```bash
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
```

#### Missing Dependencies
```bash
pip install -e .[dev]  # Reinstall with dev dependencies
```

#### Pre-commit Hook Failures
```bash
pre-commit run --all-files  # Run all hooks manually
pre-commit clean            # Clean hook cache
pre-commit install          # Reinstall hooks
```

## Contributing Guidelines

### Pull Request Process

1. **Fork and clone** the repository
2. **Create feature branch** from main
3. **Implement changes** following code standards
4. **Add tests** for new functionality
5. **Update documentation** if needed
6. **Ensure all checks pass**:
   - All tests pass
   - Code coverage maintained
   - Linting passes
   - Type checking passes
   - Security scan passes
7. **Create pull request** with clear description
8. **Address review feedback**

### Review Criteria

- Code follows project standards
- Tests cover new functionality
- Documentation is updated
- No security vulnerabilities
- Performance impact considered
- Breaking changes documented

## Architecture

### Key Design Principles

- **Modularity**: Separate concerns into focused modules
- **Testability**: Design for easy unit testing
- **Extensibility**: Support for new fairness metrics and mitigation techniques
- **Reproducibility**: Deterministic results with proper random seeding
- **Performance**: Efficient implementations for large datasets

### Adding New Fairness Metrics

1. **Add metric function** to `fairness_metrics.py`
2. **Add tests** in `tests/test_fairness_metrics.py`
3. **Update CLI** to support new metric
4. **Update documentation**

### Adding New Mitigation Techniques

1. **Implement technique** in `bias_mitigator.py`
2. **Add to evaluation pipeline** in `evaluate_fairness.py`
3. **Add CLI support** with new method option
4. **Add comprehensive tests**

## Performance Considerations

- **Dataset size**: Current implementation optimized for datasets up to 1M samples
- **Memory usage**: Monitor memory with large datasets, consider chunking
- **CPU usage**: Parallel processing available for cross-validation
- **Caching**: Intermediate results cached for repeated evaluations

## Security Guidelines

- **Never commit secrets** (use .env files, excluded from git)
- **Validate inputs** in all public functions
- **Use secure defaults** for random number generation
- **Regular dependency updates** via Dependabot
- **Security scanning** in CI/CD pipeline

## Getting Help

- **Documentation**: Check docs/ directory
- **Issues**: Search existing GitHub issues
- **Discussions**: Use GitHub Discussions for questions
- **Code Review**: Request review from maintainers

## Resources

- [Project Architecture](../ARCHITECTURE.md)
- [API Usage Guide](../API_USAGE_GUIDE.md)
- [Architecture Decision Records](adr/)
- [Fairlearn Documentation](https://fairlearn.org/)
- [Python Best Practices](https://docs.python-guide.org/)