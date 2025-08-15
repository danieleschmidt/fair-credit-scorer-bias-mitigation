"""Centralized configuration management system.

This module provides a flexible configuration system that supports:
- Default configuration from YAML files
- Environment variable overrides
- Custom configuration files
- Configuration validation
- Singleton pattern for consistent access
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


class ConfigSection:
    """Base class for configuration sections with dot notation access."""

    def __init__(self, data: Dict[str, Any]):
        """Initialize configuration section from dictionary.
        
        Args:
            data: Dictionary containing configuration values
        """
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigSection(value))
            else:
                setattr(self, key, value)

    def __repr__(self) -> str:
        """Return string representation of configuration section."""
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        return f"ConfigSection({attrs})"


class Config:
    """Centralized configuration management system.
    
    Provides access to all configuration parameters with support for:
    - Default values from YAML configuration files
    - Environment variable overrides
    - Custom configuration file loading
    - Configuration validation
    - Singleton pattern
    """

    _instance: Optional['Config'] = None
    _initialized: bool = False

    def __new__(cls, config_path: Optional[str] = None, force_reload: bool = False) -> 'Config':
        """Implement singleton pattern for consistent configuration access."""
        if cls._instance is None or force_reload:
            cls._instance = super().__new__(cls)
            if force_reload:
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: Optional[str] = None, force_reload: bool = False):
        """Initialize configuration system.
        
        Args:
            config_path: Optional path to custom configuration file
            force_reload: Force reload even for singleton instance
            
        Raises:
            ConfigValidationError: If configuration validation fails
            FileNotFoundError: If specified config file doesn't exist
        """
        # Avoid re-initialization in singleton pattern
        if self._initialized and config_path is None and not force_reload:
            return

        # Reset initialization state if new config path provided or force reload
        if config_path is not None or force_reload:
            self._initialized = False

        if not self._initialized:
            self._load_configuration(config_path)
            self._apply_environment_overrides()
            self._validate_configuration()
            self._initialized = True

    def _load_configuration(self, config_path: Optional[str] = None) -> None:
        """Load configuration from YAML file.
        
        Args:
            config_path: Optional path to custom configuration file
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            ImportError: If PyYAML is not installed
        """
        if yaml is None:
            raise ImportError("PyYAML is required for configuration management. Install with: pip install PyYAML")

        # Determine configuration file path
        if config_path is None:
            # Use default configuration file
            config_dir = Path(__file__).parent.parent / "config"
            config_path = config_dir / "default.yaml"

        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Load configuration from YAML
        try:
            with open(config_path) as f:
                config_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigValidationError(f"Invalid YAML in configuration file {config_path}: {e}")

        if not config_data:
            raise ConfigValidationError(f"Empty configuration file: {config_path}")

        # Create configuration sections
        self._create_sections(config_data)

        logger.info(f"Loaded configuration from: {config_path}")

    def _create_sections(self, config_data: Dict[str, Any]) -> None:
        """Create configuration sections from loaded data.
        
        Args:
            config_data: Dictionary containing configuration data
        """
        # Create main configuration sections
        self.model = ConfigSection(config_data.get('model', {}))
        self.data = ConfigSection(config_data.get('data', {}))
        self.evaluation = ConfigSection(config_data.get('evaluation', {}))
        self.general = ConfigSection(config_data.get('general', {}))
        self.fairness = ConfigSection(config_data.get('fairness', {}))
        self.output = ConfigSection(config_data.get('output', {}))
        self.explainability = ConfigSection(config_data.get('explainability', {}))
        self.logging = ConfigSection(config_data.get('logging', {}))

    def _apply_environment_overrides(self) -> None:
        """Apply environment variable overrides to configuration.
        
        Environment variables should follow the pattern:
        FAIRNESS_<SECTION>_<PARAMETER> = value
        
        Examples:
        - FAIRNESS_MODEL_MAX_ITER=500 (maps to model.logistic_regression.max_iter)
        - FAIRNESS_DATA_TEST_SIZE=0.25 (maps to data.default_test_size)
        - FAIRNESS_GENERAL_RANDOM_STATE=123 (maps to general.default_random_state)
        """
        prefix = "FAIRNESS_"

        # Define mappings for simplified environment variable names
        env_mappings = {
            'model_max_iter': 'model.logistic_regression.max_iter',
            'model_solver': 'model.logistic_regression.solver',
            'data_test_size': 'data.default_test_size',
            'general_random_state': 'general.default_random_state',
            'evaluation_cv_folds': 'evaluation.default_cv_folds',
        }

        for env_var, env_value in os.environ.items():
            if not env_var.startswith(prefix):
                continue

            # Parse environment variable name
            var_key = env_var[len(prefix):].lower()

            # Use mapping if available, otherwise use direct path
            if var_key in env_mappings:
                config_path = env_mappings[var_key]
            else:
                # Try to parse as direct path
                var_parts = var_key.split('_')
                if len(var_parts) < 2:
                    continue
                config_path = '.'.join(var_parts)

            # Apply the override using dot notation
            self._set_nested_value(config_path, env_value)

    def _set_nested_value(self, path: str, value: str) -> None:
        """Set a nested configuration value using dot notation.
        
        Args:
            path: Dot-separated path to configuration value
            value: String value to set (will be converted to appropriate type)
        """
        parts = path.split('.')
        current = self

        try:
            # Navigate to the parent of the target attribute
            for part in parts[:-1]:
                current = getattr(current, part)

            # Set the final value
            param_name = parts[-1]
            if hasattr(current, param_name):
                # Convert string value to appropriate type
                original_value = getattr(current, param_name)
                converted_value = self._convert_env_value(value, type(original_value))
                setattr(current, param_name, converted_value)
                logger.info(f"Applied environment override: {path} = {converted_value}")
            else:
                logger.warning(f"Unknown configuration parameter: {path}")
        except AttributeError as e:
            logger.warning(f"Invalid configuration path {path}: {e}")

    def _convert_env_value(self, value: str, target_type: type) -> Any:
        """Convert environment variable string to target type.
        
        Args:
            value: String value from environment variable
            target_type: Target type for conversion
            
        Returns:
            Converted value
        """
        if target_type is bool:
            return value.lower() in ('true', '1', 'yes', 'on')
        elif target_type is int:
            return int(value)
        elif target_type is float:
            return float(value)
        else:
            return value

    def _validate_configuration(self) -> None:
        """Validate configuration values.
        
        Raises:
            ConfigValidationError: If validation fails
        """
        # Validate data configuration
        if hasattr(self.data, 'default_test_size'):
            test_size = self.data.default_test_size
            if not 0.0 < test_size < 1.0:
                raise ConfigValidationError(f"data.default_test_size must be between 0 and 1, got {test_size}")

        # Validate model configuration
        if hasattr(self.model, 'logistic_regression'):
            if hasattr(self.model.logistic_regression, 'max_iter'):
                max_iter = self.model.logistic_regression.max_iter
                if max_iter <= 0:
                    raise ConfigValidationError(f"model.logistic_regression.max_iter must be positive, got {max_iter}")

        # Validate evaluation configuration
        if hasattr(self.evaluation, 'default_cv_folds'):
            cv_folds = self.evaluation.default_cv_folds
            if cv_folds < 2:
                raise ConfigValidationError(f"evaluation.default_cv_folds must be >= 2, got {cv_folds}")

        # Validate synthetic data parameters
        if hasattr(self.data, 'synthetic'):
            synthetic = self.data.synthetic
            if hasattr(synthetic, 'n_samples') and synthetic.n_samples <= 0:
                raise ConfigValidationError(f"data.synthetic.n_samples must be positive, got {synthetic.n_samples}")
            if hasattr(synthetic, 'n_features') and synthetic.n_features <= 0:
                raise ConfigValidationError(f"data.synthetic.n_features must be positive, got {synthetic.n_features}")
            if hasattr(synthetic, 'n_informative') and hasattr(synthetic, 'n_features'):
                if synthetic.n_informative > synthetic.n_features:
                    raise ConfigValidationError(
                        f"data.synthetic.n_informative ({synthetic.n_informative}) cannot exceed "
                        f"n_features ({synthetic.n_features})"
                    )

    def reload(self, config_path: Optional[str] = None) -> None:
        """Reload configuration from file.
        
        Args:
            config_path: Optional path to new configuration file
        """
        self._initialized = False
        self.__init__(config_path)

    def get_nested_value(self, path: str, default: Any = None) -> Any:
        """Get a nested configuration value using dot notation.
        
        Args:
            path: Dot-separated path to configuration value (e.g., 'model.logistic_regression.max_iter')
            default: Default value if path not found
            
        Returns:
            Configuration value or default
        """
        parts = path.split('.')
        current = self

        try:
            for part in parts:
                current = getattr(current, part)
            return current
        except AttributeError:
            return default

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary representation.
        
        Returns:
            Dictionary containing all configuration values
        """
        def _section_to_dict(section: ConfigSection) -> Dict[str, Any]:
            result = {}
            for key, value in section.__dict__.items():
                if isinstance(value, ConfigSection):
                    result[key] = _section_to_dict(value)
                else:
                    result[key] = value
            return result

        return {
            'model': _section_to_dict(self.model),
            'data': _section_to_dict(self.data),
            'evaluation': _section_to_dict(self.evaluation),
            'general': _section_to_dict(self.general),
            'fairness': _section_to_dict(self.fairness),
            'output': _section_to_dict(self.output),
            'logging': _section_to_dict(self.logging),
        }


# Global configuration instance for easy access
config = Config()


def get_config() -> Config:
    """Get the global configuration instance.
    
    Returns:
        Global configuration instance
    """
    return config


def reload_config(config_path: Optional[str] = None) -> None:
    """Reload the global configuration.
    
    Args:
        config_path: Optional path to new configuration file
    """
    global config
    config.reload(config_path)


def reset_config() -> None:
    """Reset the configuration singleton for testing purposes."""
    global config
    Config._instance = None
    Config._initialized = False
    config = Config()
