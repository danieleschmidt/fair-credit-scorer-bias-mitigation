# Installation

This guide covers different ways to install Fair Credit Scorer and its dependencies.

## Requirements

- Python 3.8 or higher
- pip package manager
- Git (for development installation)

## Installation Methods

### 1. From PyPI (Recommended)

```bash
pip install fair-credit-scorer-bias-mitigation
```

### 2. From Source (Development)

For the latest features or to contribute to development:

```bash
# Clone the repository
git clone https://github.com/username/fair-credit-scorer-bias-mitigation.git
cd fair-credit-scorer-bias-mitigation

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e .[dev]
```

### 3. Using Docker

```bash
# Pull the image
docker pull ghcr.io/username/fair-credit-scorer-bias-mitigation:latest

# Run with default settings
docker run --rm -it ghcr.io/username/fair-credit-scorer-bias-mitigation:latest fairness-eval --help
```

### 4. Development with Docker Compose

```bash
# Clone and enter directory
git clone https://github.com/username/fair-credit-scorer-bias-mitigation.git
cd fair-credit-scorer-bias-mitigation

# Start development environment
docker-compose up -d app

# Access the container
docker-compose exec app bash
```

## Virtual Environment Setup

We strongly recommend using a virtual environment:

### Using venv

```bash
# Create virtual environment
python -m venv fair-credit-env

# Activate it
source fair-credit-env/bin/activate  # Linux/macOS
# fair-credit-env\Scripts\activate   # Windows

# Install the package
pip install fair-credit-scorer-bias-mitigation
```

### Using conda

```bash
# Create environment
conda create -n fair-credit python=3.11

# Activate environment
conda activate fair-credit

# Install the package
pip install fair-credit-scorer-bias-mitigation
```

## Verification

Verify your installation:

```bash
# Check CLI is available
fairness-eval --help

# Run a quick test
python -c "import src.fairness_metrics; print('Installation successful!')"
```

## Optional Dependencies

### Development Tools

For contributing to the project:

```bash
pip install -e .[dev]
```

This includes:
- Testing frameworks (pytest, coverage)
- Code quality tools (ruff, black, mypy)
- Pre-commit hooks
- Documentation tools

### Documentation

To build documentation locally:

```bash
pip install -e .[docs]
mkdocs serve
```

### Performance Testing

For running benchmarks:

```bash
pip install pytest-benchmark
```

## Troubleshooting

### Common Issues

#### ImportError: No module named 'src'

Make sure you're installing in editable mode for development:
```bash
pip install -e .
```

#### Permission Errors

Use `--user` flag to install for current user only:
```bash
pip install --user fair-credit-scorer-bias-mitigation
```

#### Dependency Conflicts

Create a fresh virtual environment:
```bash
python -m venv fresh-env
source fresh-env/bin/activate
pip install fair-credit-scorer-bias-mitigation
```

#### Docker Issues

Ensure Docker is running and you have sufficient permissions:
```bash
# Check Docker status
docker --version
docker ps

# Pull the latest image
docker pull ghcr.io/username/fair-credit-scorer-bias-mitigation:latest
```

### Getting Help

If you encounter issues:

1. Check our [Troubleshooting Guide](../reference/troubleshooting.md)
2. Search existing [GitHub Issues](https://github.com/username/fair-credit-scorer-bias-mitigation/issues)
3. Create a new issue with:
   - Your operating system
   - Python version
   - Installation method used
   - Complete error message

## Next Steps

After installation:

1. Read the [Quick Start Guide](quickstart.md)
2. Explore [Basic Usage](../user-guide/basic-usage.md)
3. Try the [Examples](../user-guide/examples.md)