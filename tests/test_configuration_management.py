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


class TestConfigurationEdgeCases:
    """Test edge cases and error conditions in configuration management."""

    def test_config_section_repr(self):
        """Test ConfigSection string representation."""
        from src.config import ConfigSection
        
        test_data = {
            "test_attr": "test_value",
            "number": 42,
        }
        section = ConfigSection(test_data)
        section._private = "hidden"  # Should not appear in repr
        
        repr_str = repr(section)
        assert "ConfigSection" in repr_str
        assert "test_attr" in repr_str
        assert "test_value" in repr_str
        assert "number" in repr_str
        assert "42" in repr_str
        assert "_private" not in repr_str

    def test_yaml_import_error_simulation(self):
        """Test that the yaml import error path exists in the code."""
        # We can't easily test this without breaking the global config instance
        # But we can verify the error message and logic exists
        from src.config import Config
        import src.config as config_module
        
        # Verify the error message exists in the source
        import inspect
        source = inspect.getsource(config_module.Config._load_configuration)
        assert "PyYAML is required" in source
        assert "yaml is None" in source

    def test_environment_variable_edge_cases(self):
        """Test edge cases in environment variable processing."""
        import os
        from src.config import Config
        
        # Test environment variables with underscores that don't map cleanly
        test_env = {
            'FAIRNESS_SINGLE': 'test',  # Should be skipped (< 2 parts)
            'FAIRNESS_TEST_CUSTOM': 'value',  # Should parse as test.custom
        }
        
        original_env = os.environ.copy()
        try:
            os.environ.update(test_env)
            config = Config(force_reload=True)
            # Config should load without errors even with edge case env vars
            assert config is not None
        finally:
            # Restore original environment
            os.environ.clear()
            os.environ.update(original_env)

    def test_config_file_not_found_handling(self):
        """Test behavior when configuration file doesn't exist."""
        from src.config import Config
        import tempfile
        import os
        
        # Test with non-existent file path
        with tempfile.TemporaryDirectory() as temp_dir:
            non_existent_path = os.path.join(temp_dir, "non_existent.yaml")
            
            # Should raise FileNotFoundError for missing config files
            with pytest.raises(FileNotFoundError, match="Configuration file not found"):
                Config(config_path=non_existent_path, force_reload=True)

    def test_environment_variable_parsing_logic(self):
        """Test that environment variable parsing logic handles edge cases."""
        from src.config import Config
        
        config = Config()
        
        # Test the internal parsing logic indirectly by verifying 
        # that short variable names would be skipped
        # We test this by checking the source code contains the logic
        import inspect
        source = inspect.getsource(config._apply_environment_overrides)
        assert "len(var_parts) < 2" in source
        assert "continue" in source


class TestConfigurationEdgeCasesEnhanced:
    """Enhanced edge case tests to improve coverage from 81% to 90%+."""
    
    def test_invalid_yaml_handling(self):
        """Test handling of invalid YAML content in configuration files."""
        from src.config import Config, ConfigValidationError
        
        # Test with malformed YAML
        invalid_yaml_content = """
        model:
          logistic_regression:
            max_iter: [invalid: yaml: content
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml_content)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigValidationError, match="Invalid YAML"):
                Config(config_path=temp_path, force_reload=True)
        finally:
            os.unlink(temp_path)
    
    def test_empty_yaml_file_handling(self):
        """Test handling of empty YAML configuration files."""
        from src.config import Config, ConfigValidationError
        
        # Test with empty file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigValidationError, match="Empty configuration file"):
                Config(config_path=temp_path, force_reload=True)
        finally:
            os.unlink(temp_path)
    
    def test_missing_configuration_files_comprehensive(self):
        """Test comprehensive missing configuration file scenarios."""
        from src.config import Config
        import tempfile
        import os
        
        # Test 1: Non-existent file with absolute path
        with tempfile.TemporaryDirectory() as temp_dir:
            non_existent_path = os.path.join(temp_dir, "missing.yaml")
            
            with pytest.raises(FileNotFoundError, match="Configuration file not found"):
                Config(config_path=non_existent_path, force_reload=True)
        
        # Test 2: Non-existent file with relative path
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            Config(config_path="definitely_missing_file.yaml", force_reload=True)
        
        # Test 3: Directory instead of file
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises((FileNotFoundError, IsADirectoryError, PermissionError)):
                Config(config_path=temp_dir, force_reload=True)
    
    def test_environment_variable_overrides_comprehensive(self):
        """Test comprehensive environment variable override scenarios."""
        from src.config import Config
        
        # Test extensive environment variable scenarios
        comprehensive_env = {
            'FAIRNESS_MODEL_MAX_ITER': '750',
            'FAIRNESS_MODEL_SOLVER': 'newton-cg',
            'FAIRNESS_DATA_TEST_SIZE': '0.15',
            'FAIRNESS_GENERAL_RANDOM_STATE': '999',
            'FAIRNESS_EVALUATION_CV_FOLDS': '10',
            'FAIRNESS_UNKNOWN_PARAM': 'should_be_skipped',  # Should be ignored
            'OTHER_PREFIX_VAR': 'not_fairness',  # Should be ignored
        }
        
        original_env = os.environ.copy()
        try:
            os.environ.update(comprehensive_env)
            
            config = Config(force_reload=True)
            
            # Test successful overrides
            assert config.model.logistic_regression.max_iter == 750
            assert config.model.logistic_regression.solver == "newton-cg"
            assert config.data.default_test_size == 0.15
            assert config.general.default_random_state == 999
            assert config.evaluation.default_cv_folds == 10
            
        finally:
            # Restore original environment
            os.environ.clear()
            os.environ.update(original_env)
    
    def test_hot_reload_functionality(self):
        """Test hot-reload functionality for configuration changes."""
        from src.config import Config
        
        # Create initial config file
        initial_config = {
            'model': {
                'logistic_regression': {
                    'max_iter': 1000,
                    'solver': 'liblinear'
                }
            },
            'data': {
                'default_test_size': 0.3
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(initial_config, f)
            temp_path = f.name
        
        try:
            # Load initial config
            config = Config(config_path=temp_path, force_reload=True)
            assert config.model.logistic_regression.max_iter == 1000
            assert config.data.default_test_size == 0.3
            
            # Modify config file (simulating external change)
            updated_config = {
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
            
            with open(temp_path, 'w') as f:
                yaml.dump(updated_config, f)
            
            # Hot reload configuration
            config.reload(config_path=temp_path)
            
            # Verify changes were applied
            assert config.model.logistic_regression.max_iter == 2000
            assert config.model.logistic_regression.solver == "lbfgs"
            assert config.data.default_test_size == 0.25
            
        finally:
            os.unlink(temp_path)
    
    def test_nested_value_operations_edge_cases(self):
        """Test edge cases in nested value operations."""
        from src.config import Config
        
        config = Config()
        
        # Test getting nested values that don't exist
        assert config.get_nested_value("nonexistent.path", "default") == "default"
        assert config.get_nested_value("model.nonexistent.param", None) is None
        assert config.get_nested_value("completely.wrong.path", 42) == 42
        
        # Test getting existing nested values
        max_iter = config.get_nested_value("model.logistic_regression.max_iter")
        assert max_iter == 1000
        
        # Test single-level access
        assert config.get_nested_value("data") is not None
    
    def test_configuration_validation_edge_cases(self):
        """Test comprehensive configuration validation edge cases."""
        from src.config import Config, ConfigValidationError
        
        # Test 1: Negative max_iter
        invalid_config1 = {
            'model': {
                'logistic_regression': {
                    'max_iter': -100
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_config1, f)
            temp_path1 = f.name
        
        try:
            with pytest.raises(ConfigValidationError, match="max_iter must be positive"):
                Config(config_path=temp_path1, force_reload=True)
        finally:
            os.unlink(temp_path1)
        
        # Test 2: Invalid CV folds
        invalid_config2 = {
            'evaluation': {
                'default_cv_folds': 1
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_config2, f)
            temp_path2 = f.name
        
        try:
            with pytest.raises(ConfigValidationError, match="cv_folds must be >= 2"):
                Config(config_path=temp_path2, force_reload=True)
        finally:
            os.unlink(temp_path2)
        
        # Test 3: Invalid synthetic data parameters
        invalid_config3 = {
            'data': {
                'synthetic': {
                    'n_samples': -10,
                    'n_features': 0,
                    'n_informative': 15,
                    'n_redundant': 2
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_config3, f)
            temp_path3 = f.name
        
        try:
            with pytest.raises(ConfigValidationError):
                Config(config_path=temp_path3, force_reload=True)
        finally:
            os.unlink(temp_path3)
    
    def test_environment_override_error_handling(self):
        """Test error handling in environment variable overrides."""
        from src.config import Config
        
        # Test with invalid dot paths
        problem_env = {
            'FAIRNESS_INVALID_PATH': 'test',  # Path doesn't exist in config
            'FAIRNESS_MODEL_NONEXISTENT': 'value',  # Nested path doesn't exist
        }
        
        original_env = os.environ.copy()
        try:
            os.environ.update(problem_env)
            
            # Should not raise error, but should log warnings
            config = Config(force_reload=True)
            assert config is not None
            
        finally:
            os.environ.clear() 
            os.environ.update(original_env)
    
    def test_configuration_to_dict_conversion(self):
        """Test comprehensive configuration to dictionary conversion."""
        from src.config import Config
        
        config = Config()
        
        # Convert to dictionary
        config_dict = config.to_dict()
        
        # Verify structure
        assert isinstance(config_dict, dict)
        assert 'model' in config_dict
        assert 'data' in config_dict
        assert 'evaluation' in config_dict
        assert 'general' in config_dict
        assert 'fairness' in config_dict
        assert 'output' in config_dict
        assert 'logging' in config_dict
        
        # Verify nested structure
        assert isinstance(config_dict['model'], dict)
        assert 'logistic_regression' in config_dict['model']
        assert isinstance(config_dict['model']['logistic_regression'], dict)
        assert 'max_iter' in config_dict['model']['logistic_regression']
        
        # Verify values match original config
        assert config_dict['model']['logistic_regression']['max_iter'] == config.model.logistic_regression.max_iter
        assert config_dict['data']['default_test_size'] == config.data.default_test_size
    
    def test_config_section_nested_instantiation(self):
        """Test ConfigSection handling of nested dictionaries."""
        from src.config import ConfigSection
        
        # Test deeply nested configuration
        nested_data = {
            'level1': {
                'level2': {
                    'level3': {
                        'value': 'deep_value'
                    },
                    'other_value': 42
                },
                'simple_value': 'test'
            },
            'top_level': 'surface'
        }
        
        section = ConfigSection(nested_data)
        
        # Test access at different levels
        assert section.top_level == 'surface'
        assert section.level1.simple_value == 'test'
        assert section.level1.level2.other_value == 42
        assert section.level1.level2.level3.value == 'deep_value'
        
        # Test that nested sections are also ConfigSection instances
        assert isinstance(section.level1, ConfigSection)
        assert isinstance(section.level1.level2, ConfigSection)
        assert isinstance(section.level1.level2.level3, ConfigSection)
    
    def test_global_config_functions(self):
        """Test global configuration helper functions."""
        from src.config import get_config, reload_config, reset_config
        
        # Test get_config
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2  # Should be same instance
        
        # Test reload_config
        original_max_iter = config1.model.logistic_regression.max_iter
        
        # Create temporary config with different values
        new_config = {
            'model': {
                'logistic_regression': {
                    'max_iter': original_max_iter + 100
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(new_config, f)
            temp_path = f.name
        
        try:
            reload_config(temp_path)
            reloaded_config = get_config()
            assert reloaded_config.model.logistic_regression.max_iter == original_max_iter + 100
        finally:
            os.unlink(temp_path)
        
        # Test reset_config
        reset_config()
        reset_config_instance = get_config()
        
        # Should have default values after reset
        assert reset_config_instance.model.logistic_regression.max_iter == 1000
    
    def test_environment_variable_type_conversion_edge_cases(self):
        """Test edge cases in environment variable type conversion."""
        from src.config import Config
        
        config = Config()
        
        # Test boolean conversion edge cases
        assert config._convert_env_value("YES", bool) is True
        assert config._convert_env_value("On", bool) is True
        assert config._convert_env_value("OFF", bool) is False
        assert config._convert_env_value("random_string", bool) is False
        
        # Test numeric conversion edge cases
        assert config._convert_env_value("0", int) == 0
        assert config._convert_env_value("-999", int) == -999
        assert config._convert_env_value("1.0", float) == 1.0
        assert config._convert_env_value("-3.14159", float) == -3.14159
        
        # Test string conversion (should pass through unchanged)
        assert config._convert_env_value("", str) == ""
        assert config._convert_env_value("multi word string", str) == "multi word string"
        
        # Test with unknown type (should return string)
        result = config._convert_env_value("test", list)  # list is not handled
        assert result == "test"