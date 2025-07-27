"""Shared test configuration and fixtures."""

import tempfile
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

from src.config import reset_config
from src.data_loader_preprocessor import generate_synthetic_credit_data as generate_data


@pytest.fixture(autouse=True)
def reset_configuration():
    """Automatically reset configuration singleton before each test.
    
    This ensures test isolation by resetting the configuration state
    before each test runs, preventing configuration changes in one test
    from affecting other tests.
    """
    reset_config()
    yield
    # Optional cleanup after test - reset again
    reset_config()


@pytest.fixture(scope="session")
def temp_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_data_path(temp_data_dir):
    """Generate sample data for testing."""
    data_path = temp_data_dir / "sample_data.csv"
    generate_data(str(data_path), n_samples=1000, random_state=42)
    return str(data_path)


@pytest.fixture
def small_data_path(temp_data_dir):
    """Generate small dataset for fast testing."""
    data_path = temp_data_dir / "small_data.csv"
    generate_data(str(data_path), n_samples=100, random_state=42)
    return str(data_path)


@pytest.fixture
def sample_predictions():
    """Generate sample prediction data for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    return {
        'y_true': np.random.randint(0, 2, n_samples),
        'y_pred': np.random.randint(0, 2, n_samples),
        'y_prob': np.random.random(n_samples),
        'protected_attr': np.random.randint(0, 2, n_samples)
    }


@pytest.fixture
def sample_dataframe():
    """Generate sample DataFrame for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    return pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples),
        'protected_attribute': np.random.randint(0, 2, n_samples),
        'target': np.random.randint(0, 2, n_samples)
    })


@pytest.fixture
def mock_model_results():
    """Mock model results for testing."""
    return {
        'accuracy': 0.85,
        'precision': 0.80,
        'recall': 0.75,
        'f1_score': 0.77,
        'demographic_parity_difference': 0.15,
        'equalized_odds_difference': 0.12,
        'roc_auc': 0.88
    }


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: unit tests"
    )
    config.addinivalue_line(
        "markers", "performance: performance benchmarks"
    )
    config.addinivalue_line(
        "markers", "slow: slow running tests"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Mark integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Mark performance tests
        if "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        
        # Mark e2e tests as slow
        if "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.slow)