"""Pytest configuration and fixtures for test isolation."""

import pytest
from src.config import reset_config


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