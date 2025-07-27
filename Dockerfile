# Multi-stage build for optimized production image
FROM python:3.13-slim as base

# Security: Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Development stage
FROM base as development

# Install development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -e .[dev,test,lint,docs]

# Copy source code
COPY . .

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

USER appuser

# Default command for development
CMD ["bash"]

# Production stage
FROM base as production

WORKDIR /app

# Copy only production requirements
COPY pyproject.toml requirements.txt ./

# Install only production dependencies
RUN pip install --upgrade pip && \
    pip install . && \
    pip cache purge

# Copy source code (minimal)
COPY src/ ./src/
COPY README.md LICENSE ./

# Security: Remove package manager and unnecessary files
RUN apt-get remove -y git && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/* && \
    rm -rf /root/.cache

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import src.fairness_metrics; print('OK')" || exit 1

# Default command
CMD ["fairness-eval", "--help"]

# Test stage for CI
FROM development as test

# Copy test files
COPY tests/ ./tests/

# Run tests
RUN make test

# Final production image
FROM production as final