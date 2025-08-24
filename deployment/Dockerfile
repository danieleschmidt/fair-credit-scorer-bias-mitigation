FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y     gcc     && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY pyproject.toml ./

# Install the application
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 appuser
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3     CMD python3 -c "from src.health_check import get_health_status; print('OK' if get_health_status()['overall_health_score'] > 0.8 else exit(1))"

EXPOSE 8080

CMD ["python3", "-m", "src.api.fairness_api"]
