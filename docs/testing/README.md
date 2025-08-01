# Testing Guide

This document provides comprehensive guidance on testing practices, frameworks, and procedures for the Fair Credit Scorer project.

## Testing Philosophy

Our testing approach follows the testing pyramid principle:
- **Unit Tests** (70%): Fast, isolated tests for individual components
- **Integration Tests** (20%): Tests for component interactions
- **End-to-End Tests** (10%): Full workflow validation

## Test Organization

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py             # Shared fixtures and configuration
├── pytest.ini             # Pytest configuration
├── fixtures/               # Test data and fixtures
│   ├── __init__.py
│   └── sample_data.py      # Sample datasets and utilities
├── unit/                   # Unit tests (fast, isolated)
├── integration/            # Integration tests
├── contract/               # API contract tests
├── e2e/                   # End-to-end tests
├── performance/           # Performance and benchmark tests
├── security/              # Security-focused tests
└── regression/            # Regression test suite
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m unit                    # Unit tests only
pytest -m integration            # Integration tests only
pytest -m "not slow"             # Skip slow tests
pytest -m "fairness and unit"    # Fairness unit tests
```

### Advanced Test Execution

```bash
# Parallel execution
pytest -n auto                   # Auto-detect CPU cores
pytest -n 4                      # Use 4 processes

# Performance testing
pytest -m performance --benchmark-only

# Generate detailed reports
pytest --html=report.html --self-contained-html
pytest --junitxml=junit.xml      # For CI/CD integration
```

### Development Workflow

```bash
# Quick smoke tests during development
pytest -m smoke

# Test a specific file
pytest tests/test_fairness_metrics.py

# Test with verbose output
pytest -v tests/test_model_training.py

# Run tests on file change (requires pytest-watch)
ptw --runner "pytest -m unit"
```

## Test Categories and Markers

### Core Test Markers

- `@pytest.mark.unit`: Fast, isolated unit tests
- `@pytest.mark.integration`: Component integration tests
- `@pytest.mark.e2e`: End-to-end workflow tests
- `@pytest.mark.slow`: Tests that take >5 seconds
- `@pytest.mark.performance`: Performance benchmarks

### Domain-Specific Markers

- `@pytest.mark.fairness`: Fairness and bias testing
- `@pytest.mark.model`: ML model testing
- `@pytest.mark.data`: Data processing tests
- `@pytest.mark.api`: API endpoint tests
- `@pytest.mark.cli`: Command-line interface tests
- `@pytest.mark.security`: Security validation tests

### Infrastructure Markers

- `@pytest.mark.mock`: Tests using mocks/patches
- `@pytest.mark.tmp`: Tests requiring temporary files
- `@pytest.mark.network`: Tests requiring network access
- `@pytest.mark.gpu`: Tests requiring GPU resources

## Test Data and Fixtures

### Using Sample Data

```python
def test_fairness_metrics_calculation(sample_processed_data, sample_sensitive_attributes):
    X, y = sample_processed_data
    sensitive_attr = sample_sensitive_attributes
    
    # Test fairness metrics calculation
    metrics = calculate_fairness_metrics(y_true=y, y_pred=y, sensitive_attributes=sensitive_attr)
    
    assert 'demographic_parity_difference' in metrics
    assert 'equalized_odds_difference' in metrics
```

### Creating Custom Fixtures

```python
@pytest.fixture
def trained_model(sample_processed_data):
    """Fixture providing a trained model for testing."""
    X, y = sample_processed_data
    from sklearn.linear_model import LogisticRegression
    
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    return model
```

### Temporary File Testing

```python
def test_data_export(tmp_path, sample_credit_data):
    """Test data export functionality."""
    output_file = tmp_path / "exported_data.csv"
    
    # Test the export function
    export_data(sample_credit_data, str(output_file))
    
    # Verify file was created and has correct content
    assert output_file.exists()
    imported_data = pd.read_csv(output_file)
    assert len(imported_data) == len(sample_credit_data)
```

## Performance Testing

### Benchmark Tests

```python
@pytest.mark.performance
def test_model_prediction_performance(benchmark, trained_model, sample_processed_data):
    """Benchmark model prediction performance."""
    X, _ = sample_processed_data
    
    # Benchmark the prediction
    result = benchmark(trained_model.predict, X)
    
    # Verify predictions were made
    assert len(result) == len(X)
```

### Load Testing with Locust

```bash
# Basic load test
locust -f load_testing/locustfile.py --host=http://localhost:8080

# Headless load test with specific parameters
locust -f load_testing/locustfile.py --host=http://localhost:8080 \
       --users 50 --spawn-rate 10 --run-time 300s --headless

# Stress testing scenario
locust -f load_testing/locustfile.py --host=http://localhost:8080 \
       --users 200 --spawn-rate 50 --run-time 600s --headless
```

## Contract Testing

### API Contract Tests

```python
@pytest.mark.contract
def test_scoring_api_contract():
    """Test that scoring API maintains its contract."""
    # Test request schema
    valid_request = {
        "applicant_data": {
            "age": 35,
            "income": 75000,
            "credit_history_length": 10
        },
        "model_version": "v1.2.0"
    }
    
    response = client.post("/api/v1/score", json=valid_request)
    
    # Verify response structure
    assert response.status_code == 200
    data = response.json()
    
    required_fields = ["credit_score", "risk_category", "confidence", "fairness_metrics"]
    assert all(field in data for field in required_fields)
    
    # Verify data types
    assert isinstance(data["credit_score"], (int, float))
    assert isinstance(data["risk_category"], str)
    assert 0 <= data["confidence"] <= 1
```

## Security Testing

### Input Validation Tests

```python
@pytest.mark.security
@pytest.mark.parametrize("malicious_input", [
    {"age": -1},  # Invalid age
    {"income": "'; DROP TABLE users; --"},  # SQL injection attempt
    {"credit_history_length": 10**10},  # Unreasonably large number
    {},  # Empty input
])
def test_input_validation_security(malicious_input):
    """Test that API properly validates and rejects malicious input."""
    response = client.post("/api/v1/score", json={"applicant_data": malicious_input})
    
    # Should return 400 Bad Request for invalid input
    assert response.status_code == 400
    assert "error" in response.json()
```

### Data Privacy Tests

```python
@pytest.mark.security
def test_sensitive_data_not_logged(sample_credit_data, caplog):
    """Ensure sensitive data is not logged."""
    with caplog.at_level(logging.INFO):
        process_applicant_data(sample_credit_data)
    
    # Check that sensitive fields are not in logs
    log_content = caplog.text.lower()
    sensitive_patterns = ["ssn", "social", "credit_card", "password"]
    
    for pattern in sensitive_patterns:
        assert pattern not in log_content
```

## Fairness Testing

### Bias Detection Tests

```python
@pytest.mark.fairness
def test_demographic_parity_calculation():
    """Test demographic parity calculation accuracy."""
    # Create test data with known bias
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 1, 1, 0, 0, 1])  # Biased predictions
    sensitive_attr = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])
    
    dp_diff = demographic_parity_difference(y_pred, sensitive_attributes=sensitive_attr)
    
    # Verify calculation
    # Group A: 2/4 = 0.5 positive rate
    # Group B: 2/4 = 0.5 positive rate
    # Difference: |0.5 - 0.5| = 0.0
    assert abs(dp_diff - 0.0) < 1e-10
```

### Model Fairness Validation

```python
@pytest.mark.fairness
@pytest.mark.integration
def test_model_fairness_constraints():
    """Test that trained models meet fairness constraints."""
    # Train model with fairness constraints
    model = train_fair_model(
        X_train, y_train, sensitive_attributes,
        fairness_constraint='demographic_parity',
        tolerance=0.1
    )
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Verify fairness constraints are met
    dp_diff = demographic_parity_difference(y_pred, sensitive_attributes_test)
    assert abs(dp_diff) <= 0.1, f"Demographic parity violation: {dp_diff}"
```

## Continuous Integration Testing

### CI/CD Pipeline Integration

```yaml
# .github/workflows/test.yml example
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', 3.11]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e .[dev,test]
    
    - name: Run unit tests
      run: pytest -m "unit and not slow" --cov=src --cov-report=xml
    
    - name: Run integration tests
      run: pytest -m integration --maxfail=5
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### Quality Gates

```bash
# Coverage threshold (configured in pytest.ini)
pytest --cov=src --cov-fail-under=85

# Performance regression detection
pytest -m performance --benchmark-compare=baseline.json

# Security scanning
bandit -r src/ -f json -o security-report.json
safety check --json --output safety-report.json
```

## Test Data Management

### Data Generation

```python
# Generate realistic test data
from faker import Faker
fake = Faker()

def generate_applicant_data(n=1000, seed=42):
    """Generate synthetic applicant data for testing."""
    fake.seed_instance(seed)
    
    data = []
    for _ in range(n):
        applicant = {
            'age': fake.random_int(18, 80),
            'income': fake.random_int(20000, 200000),
            'credit_score': fake.random_int(300, 850),
            'employment_status': fake.random_element(['employed', 'unemployed', 'retired']),
            'education_level': fake.random_element(['high_school', 'bachelor', 'master', 'phd'])
        }
        data.append(applicant)
    
    return pd.DataFrame(data)
```

### Data Versioning

```bash
# Use DVC for test data versioning
dvc add tests/data/large_test_dataset.csv
git add tests/data/large_test_dataset.csv.dvc
git commit -m "Add versioned test dataset"
```

## Debugging Failed Tests

### Common Debugging Strategies

```bash
# Run with verbose output and no capture
pytest -vs test_failing_test.py

# Drop into debugger on failure
pytest --pdb test_failing_test.py

# Debug specific test with ipdb
pytest --pdbcls=IPython.terminal.debugger:Pdb test_failing_test.py

# Show local variables in tracebacks
pytest --tb=long test_failing_test.py
```

### Test Isolation Issues

```python
# Use fresh instances for each test
@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global state before each test."""
    # Clear any global caches
    clear_model_cache()
    reset_configuration()
    yield
    # Cleanup after test
    cleanup_temp_files()
```

## Best Practices

### Writing Effective Tests

1. **Follow AAA Pattern**: Arrange, Act, Assert
2. **Use Descriptive Names**: Test names should explain what they test
3. **Keep Tests Independent**: Each test should run in isolation
4. **Test Edge Cases**: Include boundary conditions and error cases
5. **Mock External Dependencies**: Use mocks for external services

### Test Maintenance

1. **Regular Test Review**: Remove obsolete tests, update as code evolves
2. **Performance Monitoring**: Track test execution time trends
3. **Coverage Analysis**: Aim for high coverage but focus on critical paths
4. **Documentation**: Keep test documentation current with code changes

### Error Handling Testing

```python
@pytest.mark.unit
def test_graceful_error_handling():
    """Test that errors are handled gracefully."""
    with pytest.raises(ValidationError, match="Invalid input format"):
        process_invalid_data("malformed_input")
    
    # Test that partial failures don't crash the system
    result = process_batch_with_errors(mixed_valid_invalid_data)
    assert result.successful_count > 0
    assert result.error_count > 0
    assert len(result.errors) == result.error_count
```

## Performance Benchmarks

### Baseline Performance Targets

- Model training: < 30 seconds for 10K samples
- Prediction latency: < 100ms for single prediction
- Batch prediction: < 1 second for 100 predictions
- Fairness metrics calculation: < 500ms for 1K predictions
- API response time: < 200ms (95th percentile)

### Regression Testing

```python
@pytest.mark.performance
def test_prediction_performance_regression(benchmark):
    """Ensure prediction performance doesn't regress."""
    # Load baseline performance data
    baseline = load_performance_baseline()
    
    # Benchmark current implementation
    result = benchmark(model.predict, test_data)
    
    # Assert performance hasn't degraded significantly
    performance_ratio = benchmark.stats.mean / baseline.mean
    assert performance_ratio < 1.2, f"Performance degraded by {(performance_ratio-1)*100:.1f}%"
```

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Locust Documentation](https://locust.io/)
- [Fairlearn Testing Guide](https://fairlearn.org/main/user_guide/assessment/)
- [Security Testing Guide](https://owasp.org/www-project-web-security-testing-guide/)

## Contributing to Tests

When adding new features or fixing bugs:

1. Write tests first (TDD approach)
2. Ensure all existing tests pass
3. Add appropriate test markers
4. Update test documentation
5. Consider performance implications
6. Test edge cases and error conditions
7. Add integration tests for new workflows