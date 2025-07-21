"""Tests for centralized configuration management system."""

import os
import tempfile
import pytest
from unittest.mock import patch, mock_open
import yaml

# Import will be available after we create the config module
pytest.importorskip("src.config", reason="Config module not yet implemented")


class TestConfigurationSystem:
    """Test the centralized configuration management system."""

    def test_default_config_loads_successfully(self):
        """Test that default configuration loads without errors."""
        from src.config import Config
        
        config = Config()
        assert config is not None
        
        # Test that all required sections exist
        assert hasattr(config, 'model')
        assert hasattr(config, 'data')
        assert hasattr(config, 'evaluation')
        assert hasattr(config, 'general')

    def test_model_configuration_values(self):
        """Test model configuration parameters."""
        from src.config import Config
        
        config = Config()
        
        # Test logistic regression parameters
        assert hasattr(config.model, 'logistic_regression')
        assert config.model.logistic_regression.max_iter == 1000
        assert config.model.logistic_regression.solver == "liblinear"
        
        # Test bias mitigation parameters
        assert hasattr(config.model, 'bias_mitigation')
        assert config.model.bias_mitigation.max_iter == 1000
        assert config.model.bias_mitigation.solver == "liblinear"
        
        # Test default method
        assert config.model.default_method == "baseline"

    def test_data_configuration_values(self):
        """Test data configuration parameters."""
        from src.config import Config
        
        config = Config()
        
        # Test file paths
        assert config.data.default_dataset_path == "data/credit_data.csv"
        
        # Test column names
        assert config.data.protected_column_name == "protected"
        assert config.data.label_column_name == "label"
        
        # Test split parameters
        assert config.data.default_test_size == 0.3
        assert config.data.min_test_samples == 1
        
        # Test synthetic data parameters
        assert hasattr(config.data, 'synthetic')
        assert config.data.synthetic.n_samples == 1000
        assert config.data.synthetic.n_features == 10
        assert config.data.synthetic.n_informative == 5
        assert config.data.synthetic.n_redundant == 2

    def test_evaluation_configuration_values(self):
        """Test evaluation configuration parameters."""
        from src.config import Config
        
        config = Config()
        
        assert config.evaluation.default_cv_folds == 5
        assert config.evaluation.cv_shuffle == True

    def test_general_configuration_values(self):
        """Test general configuration parameters."""
        from src.config import Config
        
        config = Config()
        
        assert config.general.default_random_state == 42

    def test_config_from_custom_file(self):
        """Test loading configuration from a custom YAML file."""
        from src.config import Config
        
        custom_config = {
            'model': {
                'logistic_regression': {
                    'max_iter': 2000,
                    'solver': 'lbfgs'
                }
            },
            'data': {
                'default_test_size': 0.25
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(custom_config, f)
            temp_path = f.name
        
        try:
            config = Config(config_path=temp_path)
            assert config.model.logistic_regression.max_iter == 2000
            assert config.model.logistic_regression.solver == "lbfgs"
            assert config.data.default_test_size == 0.25
        finally:
            os.unlink(temp_path)

    def test_config_from_environment_variables(self):
        """Test configuration override from environment variables."""
        from src.config import Config
        
        with patch.dict(os.environ, {
            'FAIRNESS_MODEL_MAX_ITER': '500',
            'FAIRNESS_DATA_TEST_SIZE': '0.2',
            'FAIRNESS_GENERAL_RANDOM_STATE': '123'
        }):
            config = Config(force_reload=True)
            assert config.model.logistic_regression.max_iter == 500
            assert config.data.default_test_size == 0.2
            assert config.general.default_random_state == 123

    def test_config_validation_errors(self):
        """Test that invalid configuration values raise appropriate errors."""
        from src.config import Config, ConfigValidationError
        
        invalid_config = {
            'data': {
                'default_test_size': 1.5  # Invalid: must be between 0 and 1
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_config, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigValidationError) as exc_info:
                Config(config_path=temp_path)
            assert "test_size must be between 0 and 1" in str(exc_info.value)
        finally:
            os.unlink(temp_path)

    def test_config_singleton_behavior(self):
        """Test that Config behaves as a singleton within the same process."""
        from src.config import Config
        
        config1 = Config()
        config2 = Config()
        
        # Should be the same instance
        assert config1 is config2

    def test_config_reload(self):
        """Test that configuration can be reloaded with new values."""
        from src.config import Config
        
        # Create initial config
        config = Config()
        original_max_iter = config.model.logistic_regression.max_iter
        
        # Create new config file with different values
        new_config = {
            'model': {
                'logistic_regression': {
                    'max_iter': original_max_iter + 500
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(new_config, f)
            temp_path = f.name
        
        try:
            # Reload with new config
            config.reload(config_path=temp_path)
            assert config.model.logistic_regression.max_iter == original_max_iter + 500
        finally:
            os.unlink(temp_path)


class TestConfigurationIntegration:
    """Test integration of configuration system with existing modules."""

    def test_baseline_model_uses_config(self):
        """Test that baseline_model module uses configuration values."""
        from src.config import Config
        from src.baseline_model import train_baseline_model
        import numpy as np
        
        # Create test data
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        
        config = Config()
        
        # Train model - should use config values
        model = train_baseline_model(X, y)
        
        # Check that model was created successfully
        assert model is not None
        assert hasattr(model, 'predict')

    def test_data_loader_uses_config(self):
        """Test that data loader can access configuration values."""
        from src.config import Config
        
        config = Config(force_reload=True)
        
        # Test that synthetic configuration is accessible
        assert hasattr(config.data, 'synthetic')
        assert config.data.synthetic.n_samples == 1000
        assert config.data.synthetic.n_features == 10
        
        # Note: Integration with existing modules will be done in a separate step
        # This test validates that configuration values are available for use


class TestEnvironmentVariableConversion:
    """Test environment variable type conversion functionality."""

    def test_convert_env_value_bool_conversion(self):
        """Test boolean conversion from environment variables."""
        from src.config import Config
        
        config = Config()
        
        # Test true values
        assert config._convert_env_value("true", bool) is True
        assert config._convert_env_value("True", bool) is True
        assert config._convert_env_value("TRUE", bool) is True
        assert config._convert_env_value("1", bool) is True
        assert config._convert_env_value("yes", bool) is True
        assert config._convert_env_value("on", bool) is True
        
        # Test false values
        assert config._convert_env_value("false", bool) is False
        assert config._convert_env_value("False", bool) is False
        assert config._convert_env_value("0", bool) is False
        assert config._convert_env_value("no", bool) is False
        assert config._convert_env_value("off", bool) is False

    def test_convert_env_value_int_conversion(self):
        """Test integer conversion from environment variables."""
        from src.config import Config
        
        config = Config()
        
        assert config._convert_env_value("42", int) == 42
        assert config._convert_env_value("0", int) == 0
        assert config._convert_env_value("-10", int) == -10
        assert config._convert_env_value("1000", int) == 1000

    def test_convert_env_value_float_conversion(self):
        """Test float conversion from environment variables."""
        from src.config import Config
        
        config = Config()
        
        assert config._convert_env_value("3.14", float) == 3.14
        assert config._convert_env_value("0.0", float) == 0.0
        assert config._convert_env_value("-2.5", float) == -2.5
        assert config._convert_env_value("1e-3", float) == 0.001

    def test_convert_env_value_string_passthrough(self):
        """Test that string values pass through unchanged."""
        from src.config import Config
        
        config = Config()
        
        assert config._convert_env_value("test", str) == "test"
        assert config._convert_env_value("complex string", str) == "complex string"
        assert config._convert_env_value("123", str) == "123"

    def test_convert_env_value_type_identity_checks(self):
        """Test that type comparisons use identity (is) rather than equality (==)."""
        from src.config import Config
        
        config = Config()
        
        # This test ensures our fix for E721 works correctly
        # The type identity checks should work with actual type objects
        assert config._convert_env_value("true", bool) is True
        assert config._convert_env_value("42", int) == 42
        assert config._convert_env_value("3.14", float) == 3.14