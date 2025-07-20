"""Test the src package initialization and lazy imports."""

import pytest
from unittest.mock import patch, MagicMock


def test_all_exports():
    """Test that __all__ contains expected exports."""
    import src
    expected = [
        "ArchitectureReview",
        "compute_fairness_metrics", 
        "evaluate_model",
        "expgrad_demographic_parity",
        "load_credit_data",
        "load_credit_dataset",
        "postprocess_equalized_odds",
        "reweight_samples",
        "run_cross_validation",
        "run_pipeline",
        "train_baseline_model",
    ]
    assert src.__all__ == expected


def test_lazy_import_valid_attribute():
    """Test that valid attributes are imported lazily."""
    import src
    
    # Should successfully import a valid function
    func = src.load_credit_data
    assert callable(func)
    
    # Should successfully import a valid class
    cls = src.ArchitectureReview
    assert hasattr(cls, "__init__")


def test_lazy_import_invalid_attribute():
    """Test that invalid attributes raise AttributeError."""
    import src
    
    with pytest.raises(AttributeError, match="module 'src' has no attribute 'invalid_name'"):
        _ = src.invalid_name


def test_getattr_module_import():
    """Test the module import mechanism in __getattr__."""
    from src import __getattr__
    
    # Test importing a function from baseline_model
    result = __getattr__("train_baseline_model")
    assert callable(result)
    
    # Test importing from fairness_metrics
    result = __getattr__("compute_fairness_metrics")
    assert callable(result)


def test_getattr_with_mock_importlib():
    """Test __getattr__ with mocked importlib to verify import path."""
    import src
    
    with patch('importlib.import_module') as mock_import:
        mock_module = MagicMock()
        mock_module.train_baseline_model = MagicMock()
        mock_import.return_value = mock_module
        
        result = src.__getattr__("train_baseline_model")
        
        mock_import.assert_called_once_with(".baseline_model", "src")
        assert result == mock_module.train_baseline_model


def test_all_lazy_imports_work():
    """Test that all items in __all__ can be successfully imported."""
    import src
    
    for name in src.__all__:
        # This should not raise an exception
        obj = getattr(src, name)
        assert obj is not None