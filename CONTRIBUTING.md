# Contributing to Fair Credit Scorer

Thank you for your interest in contributing to Fair Credit Scorer! This document provides guidelines for contributing to the project.

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When creating a bug report, include:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples** with code samples
- **Describe the behavior you observed** and what you expected
- **Include system information** (OS, Python version, package version)

Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md).

### Suggesting Enhancements

Enhancement suggestions are welcome! Please:

- **Use a clear and descriptive title**
- **Provide a detailed description** of the suggested enhancement
- **Explain why this enhancement would be useful**
- **Include examples** of how the feature would work

Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md).

### Security Vulnerabilities

**Do not create public issues for security vulnerabilities.** Please follow our [Security Policy](SECURITY.md) for responsible disclosure.

## Development Process

### Setting Up Development Environment

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/your-username/fair-credit-scorer-bias-mitigation.git
   cd fair-credit-scorer-bias-mitigation
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -e .[dev]
   pre-commit install
   ```

4. **Verify setup**
   ```bash
   make test
   make lint
   ```

### Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write tests for new functionality
   - Follow the coding standards
   - Update documentation as needed

3. **Run quality checks**
   ```bash
   make quality  # Runs linting, formatting, and type checking
   make test     # Runs the test suite
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```
   
   We use [Conventional Commits](https://www.conventionalcommits.org/). Use:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation
   - `refactor:` for code refactoring
   - `test:` for adding tests
   - `chore:` for maintenance tasks

5. **Push and create a pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

### Pull Request Guidelines

- **Use the pull request template**
- **Include tests** for new functionality
- **Update documentation** if needed
- **Keep changes focused** - one feature per PR
- **Write clear commit messages**
- **Respond to feedback** promptly

## Coding Standards

### Python Style Guide

- **Follow PEP 8** with line length of 88 characters
- **Use type hints** for function signatures
- **Write docstrings** in Google style
- **Use meaningful variable names**

### Code Quality

We use these tools to maintain code quality:

- **Ruff**: Linting and code analysis
- **Black**: Code formatting
- **MyPy**: Type checking
- **Bandit**: Security analysis

Run all checks with:
```bash
make quality
```

### Testing

- **Write tests** for all new functionality
- **Maintain test coverage** above 80%
- **Use descriptive test names**
- **Include both unit and integration tests**

Test categories:
- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.performance` - Performance benchmarks

Run tests with:
```bash
make test              # All tests
make test-fast         # Unit tests only
pytest -m integration  # Integration tests only
```

### Documentation

- **Update relevant documentation** for changes
- **Include code examples** where helpful
- **Use clear, concise language**
- **Follow the existing style**

Build documentation locally:
```bash
make docs
make serve-docs
```

## Architecture Guidelines

### Project Structure

```
src/
├── __init__.py                    # Package initialization
├── data_loader_preprocessor.py    # Data handling
├── baseline_model.py             # Model implementations
├── bias_mitigator.py             # Bias mitigation techniques
├── fairness_metrics.py           # Metrics computation
├── evaluate_fairness.py          # Pipeline orchestration
├── health_check.py               # Monitoring utilities
├── monitoring.py                 # Application monitoring
└── fair_credit_scorer_bias_mitigation/
    ├── __init__.py               # Public API
    └── cli.py                    # Command-line interface
```

### Adding New Features

When adding new features:

1. **Design interfaces first** - Think about the API
2. **Keep modules focused** - Single responsibility principle
3. **Add comprehensive tests** - Test all code paths
4. **Document the feature** - Include examples
5. **Consider backwards compatibility** - Avoid breaking changes

### Performance Considerations

- **Profile before optimizing** - Measure performance impact
- **Use appropriate data structures** - Consider memory usage
- **Cache expensive computations** - Where appropriate
- **Test with realistic data sizes** - Ensure scalability

## Release Process

The project uses semantic versioning and automated releases:

1. **Conventional commits** trigger version bumps
2. **CI/CD pipeline** runs tests and builds
3. **Automated release** creates tags and publishes packages
4. **Changelog** is automatically generated

## Getting Help

- **Documentation**: Check the [docs](docs/) directory
- **Issues**: Search existing [GitHub Issues](https://github.com/username/fair-credit-scorer-bias-mitigation/issues)
- **Discussions**: Use [GitHub Discussions](https://github.com/username/fair-credit-scorer-bias-mitigation/discussions)

## Recognition

Contributors are recognized in:
- **CHANGELOG.md** for their contributions
- **README.md** contributors section
- **Release notes** for significant contributions

Thank you for contributing to Fair Credit Scorer! 🎉