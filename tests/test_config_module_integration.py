"""
Integration tests for configuration management system.

These tests verify that the centralized configuration system works
correctly with all modules without mocking.
"""

import pytest
import pandas as pd
from src.config import Config, get_config


class TestConfigModuleIntegration:
    """Integration tests for config system with actual modules."""
    
    def test_baseline_model_real_integration(self):
        """Test baseline model actually uses config values."""
        from src.baseline_model import train_baseline_model
        
        # Get real config
        config = get_config()
        expected_max_iter = config.model.logistic_regression.max_iter
        expected_solver = config.model.logistic_regression.solver
        
        # Create test data
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8],
            'feature2': [2, 4, 6, 8, 10, 12, 14, 16]
        })
        y = [0, 1, 0, 1, 0, 1, 0, 1]
        
        # Train model using config defaults
        model = train_baseline_model(X, y)
        
        # Verify model uses config values
        assert model.max_iter == expected_max_iter
        assert model.solver == expected_solver
    
    def test_data_loader_real_integration(self):
        """Test data loader actually uses config random_state."""
        from src.data_loader_preprocessor import generate_synthetic_credit_data
        
        # Get real config
        config = get_config()
        expected_random_state = config.data.random_state
        
        # Generate data twice with default parameters
        X1, y1, sensitive1 = generate_synthetic_credit_data(n_samples=50)
        X2, y2, sensitive2 = generate_synthetic_credit_data(n_samples=50)
        
        # Results should be identical (same random state from config)
        import numpy as np
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
        np.testing.assert_array_equal(sensitive1, sensitive2)
    
    def test_explainability_real_integration(self):
        """Test explainability module uses config properly."""
        from src.model_explainability import ModelExplainer
        from src.baseline_model import train_baseline_model
        
        # Create test model and data
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10]
        })
        y = [0, 1, 0, 1, 0]
        
        model = train_baseline_model(X, y)
        
        # Create explainer (should use config for random_state)
        explainer = ModelExplainer(model, X)
        
        # Test that explainer works
        test_instance = pd.DataFrame({
            'feature1': [3],
            'feature2': [6]
        })
        
        explanation = explainer.explain_prediction(test_instance)
        
        # Verify explanation structure
        assert 'shap_values' in explanation
        assert 'feature_importance' in explanation
        assert 'prediction' in explanation
        assert len(explanation['feature_importance']) == 2
    
    def test_config_consistency_across_modules(self):
        """Test that all modules see the same config values."""
        # Import all modules that use config
        from src.baseline_model import get_config as baseline_get_config
        from src.bias_mitigator import get_config as bias_get_config
        from src.data_loader_preprocessor import get_config as data_get_config
        from src.model_explainability import get_config as explain_get_config
        
        # Get config from each module
        config1 = baseline_get_config()
        config2 = bias_get_config()
        config3 = data_get_config()
        config4 = explain_get_config()
        
        # All should be the same instance or have same values
        assert config1.model.logistic_regression.max_iter == config2.model.bias_mitigation.max_iter
        assert config1.data.random_state == config3.data.random_state
        # Test that explainability config exists and is consistent
        assert hasattr(config4, 'explainability')
        if hasattr(config4, 'explainability'):
            assert config3.data.random_state == config4.explainability.random_state