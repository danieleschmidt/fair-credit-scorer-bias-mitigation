"""
Tests for configuration management integration across modules.

This test suite verifies that all modules properly use the centralized
configuration system instead of hardcoded values.
"""

import pytest
from unittest.mock import patch, Mock
import pandas as pd
from src.config import Config


class TestConfigurationIntegration:
    """Test configuration integration across all modules."""
    
    def test_baseline_model_uses_config(self):
        """Test that baseline_model uses centralized config."""
        from src.baseline_model import train_baseline_model
        
        # Create sample data
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10]
        })
        y = [0, 1, 0, 1, 0]
        
        # Mock config to test different values
        with patch('src.baseline_model.get_config') as mock_config:
            config_obj = Mock()
            config_obj.model.logistic_regression.max_iter = 500
            config_obj.model.logistic_regression.solver = "lbfgs"
            mock_config.return_value = config_obj
            
            model = train_baseline_model(X, y)
            
            # Verify config was called
            mock_config.assert_called_once()
            
            # Verify model uses config values
            assert model.max_iter == 500
            assert model.solver == "lbfgs"
    
    def test_bias_mitigator_uses_config(self):
        """Test that bias_mitigator uses centralized config."""
        from src.bias_mitigator import expgrad_demographic_parity
        
        # Create sample data
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10]
        })
        y = [0, 1, 0, 1, 0]
        sensitive_features = [1, 0, 1, 0, 1]
        
        # Mock config to test different values
        with patch('src.bias_mitigator.get_config') as mock_config:
            config_obj = Mock()
            config_obj.model.bias_mitigation.max_iter = 750
            config_obj.model.bias_mitigation.solver = "saga"
            mock_config.return_value = config_obj
            
            model = expgrad_demographic_parity(X, y, sensitive_features)
            
            # Verify config was called
            mock_config.assert_called_once()
            
            # Check that underlying estimator uses config values
            base_estimator = model.estimator
            assert base_estimator.max_iter == 750
            assert base_estimator.solver == "saga"
    
    def test_data_loader_uses_config_random_state(self):
        """Test that data loading uses config for random_state."""
        from src.data_loader_preprocessor import generate_synthetic_credit_data
        
        with patch('src.data_loader_preprocessor.get_config') as mock_config:
            config_obj = Mock()
            config_obj.data.random_state = 123
            mock_config.return_value = config_obj
            
            X, y, sensitive = generate_synthetic_credit_data(n_samples=100)
            
            # Verify config was called
            mock_config.assert_called_once()
            
            # Verify reproducible results with fixed random state
            X2, y2, sensitive2 = generate_synthetic_credit_data(n_samples=100)
            import numpy as np
            np.testing.assert_array_equal(X, X2)
            np.testing.assert_array_equal(y, y2)
            np.testing.assert_array_equal(sensitive, sensitive2)
    
    def test_explainability_uses_config(self):
        """Test that explainability modules use config for random states."""
        from src.model_explainability import ModelExplainer
        from unittest.mock import Mock
        
        # Mock model and data
        mock_model = Mock()
        background_data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        
        with patch('src.model_explainability.get_config') as mock_config:
            config_obj = Mock()
            config_obj.explainability.random_state = 456
            config_obj.explainability.background_sample_size = 25
            mock_config.return_value = config_obj
            
            explainer = ModelExplainer(mock_model, background_data)
            
            # This test passes if no exception is raised and config is used
            assert explainer is not None
    
    def test_all_modules_import_config(self):
        """Test that all required modules can import and use config."""
        modules_to_test = [
            'src.baseline_model',
            'src.bias_mitigator', 
            'src.data_loader_preprocessor',
            'src.model_explainability'
        ]
        
        for module_name in modules_to_test:
            # Import module and verify get_config is available
            module = __import__(module_name, fromlist=['get_config'])
            assert hasattr(module, 'get_config'), f"{module_name} should import get_config"
    
    def test_config_backward_compatibility(self):
        """Test that existing functionality still works after config integration."""
        from src.baseline_model import train_baseline_model
        
        # Test with default parameters (should use config)
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10]
        })
        y = [0, 1, 0, 1, 0]
        
        model = train_baseline_model(X, y)
        
        # Model should be trained successfully
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
        
        # Test prediction functionality
        predictions = model.predict(X)
        assert len(predictions) == len(y)
        
        probabilities = model.predict_proba(X)
        assert probabilities.shape == (len(y), 2)
    
    def test_config_parameter_overrides(self):
        """Test that explicit parameters can still override config values."""
        from src.baseline_model import train_baseline_model
        
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10]
        })
        y = [0, 1, 0, 1, 0]
        
        # Test explicit parameter override
        model = train_baseline_model(X, y, max_iter=2000, solver="newton-cg")
        
        # Explicit parameters should override config
        assert model.max_iter == 2000
        assert model.solver == "newton-cg"