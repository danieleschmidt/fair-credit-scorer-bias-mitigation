"""Centralized logging configuration for the fair credit scorer system.

This module provides a unified logging setup that integrates with the configuration
system to ensure consistent logging across all modules.

Features:
- Centralized configuration through config.yaml
- Support for multiple handlers (console, file, rotating file)
- Module-specific log levels
- Structured logging format options
- Environment-specific configurations
- Integration with existing configuration system

Usage:
    >>> from logging_config import setup_logging, get_logger
    >>> 
    >>> # Initialize logging system (call once at application start)
    >>> setup_logging()
    >>> 
    >>> # Get logger for a module
    >>> logger = get_logger(__name__)
    >>> logger.info("Application started")

The logging configuration is controlled by the logging section in config/default.yaml
and can be overridden through environment variables.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from .config import get_config
except ImportError:
    from config import get_config


# Global flag to track if logging has been initialized
_logging_initialized = False


class ContextFilter(logging.Filter):
    """Add contextual information to log records."""
    
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        """Initialize context filter.
        
        Args:
            context: Dictionary of context information to add to log records
        """
        super().__init__()
        self.context = context or {}
    
    def filter(self, record):
        """Add context information to the log record."""
        for key, value in self.context.items():
            setattr(record, key, value)
        return True


def setup_logging(config_override: Optional[Dict[str, Any]] = None,
                  force_reinit: bool = False) -> None:
    """Set up centralized logging configuration.
    
    This function configures the root logger and sets up handlers based on
    the configuration file. It should be called once at application startup.
    
    Args:
        config_override: Optional configuration override dictionary
        force_reinit: Force re-initialization even if already initialized
    """
    global _logging_initialized
    
    if _logging_initialized and not force_reinit:
        return
    
    try:
        config = get_config()
        log_config = config.logging
    except Exception:
        # Fallback configuration if config system fails
        log_config = _get_fallback_config()
    
    # Apply any overrides
    if config_override:
        log_config = {**log_config.__dict__, **config_override}
    
    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set root logger level
    root_logger.setLevel(getattr(logging, log_config.default_level.upper()))
    
    # Set up handlers
    if hasattr(log_config, 'handlers'):
        _setup_console_handler(root_logger, log_config)
        _setup_file_handler(root_logger, log_config)
    
    # Configure module-specific loggers
    if hasattr(log_config, 'modules'):
        _configure_module_loggers(log_config.modules)
    
    # Disable noisy loggers
    if hasattr(log_config, 'disable'):
        _disable_loggers(log_config.disable)
    
    _logging_initialized = True


def _setup_console_handler(root_logger: logging.Logger, log_config) -> None:
    """Set up console logging handler."""
    if not (hasattr(log_config.handlers, 'console') and log_config.handlers.console.enabled):
        return
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_config.handlers.console.level.upper()))
    
    # Get format
    format_name = log_config.handlers.console.format
    format_string = getattr(log_config.format, format_name)
    date_format = log_config.date_format
    
    formatter = logging.Formatter(format_string, datefmt=date_format)
    console_handler.setFormatter(formatter)
    
    root_logger.addHandler(console_handler)


def _setup_file_handler(root_logger: logging.Logger, log_config) -> None:
    """Set up file logging handler with rotation."""
    if not (hasattr(log_config.handlers, 'file') and log_config.handlers.file.enabled):
        return
    
    # Ensure log directory exists
    log_file = Path(log_config.handlers.file.filename)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(log_file),
        maxBytes=log_config.handlers.file.max_bytes,
        backupCount=log_config.handlers.file.backup_count
    )
    file_handler.setLevel(getattr(logging, log_config.handlers.file.level.upper()))
    
    # Get format
    format_name = log_config.handlers.file.format
    format_string = getattr(log_config.format, format_name)
    date_format = log_config.date_format
    
    formatter = logging.Formatter(format_string, datefmt=date_format)
    file_handler.setFormatter(formatter)
    
    root_logger.addHandler(file_handler)


def _configure_module_loggers(module_levels) -> None:
    """Configure module-specific log levels."""
    # Handle both dict and ConfigSection objects
    if hasattr(module_levels, '__dict__'):
        items = module_levels.__dict__.items()
    elif hasattr(module_levels, 'items'):
        items = module_levels.items()
    else:
        return  # Skip if we can't iterate over it
    
    for module_name, level in items:
        if not module_name.startswith('_'):  # Skip private attributes
            logger = logging.getLogger(module_name)
            logger.setLevel(getattr(logging, level.upper()))


def _disable_loggers(loggers_to_disable: list) -> None:
    """Disable specified loggers."""
    for logger_name in loggers_to_disable:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL)


def _get_fallback_config():
    """Get fallback configuration if config system fails."""
    class FallbackConfig:
        def __init__(self):
            self.default_level = "INFO"
            self.format = type('Format', (), {
                'simple': '%(levelname)s - %(name)s - %(message)s',
                'standard': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'detailed': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            })()
            self.date_format = "%Y-%m-%d %H:%M:%S"
            self.handlers = type('Handlers', (), {
                'console': type('Console', (), {'enabled': True, 'level': 'INFO', 'format': 'simple'})(),
                'file': type('File', (), {'enabled': False})()
            })()
    
    return FallbackConfig()


def get_logger(name: str, context: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """Get a logger for the specified module with optional context.
    
    Args:
        name: Logger name (typically __name__)
        context: Optional context dictionary to add to all log messages
        
    Returns:
        Configured logger instance
    """
    # Ensure logging is initialized
    if not _logging_initialized:
        setup_logging()
    
    logger = logging.getLogger(name)
    
    # Add context filter if provided
    if context:
        context_filter = ContextFilter(context)
        logger.addFilter(context_filter)
    
    return logger


def log_function_call(func_name: str, args: tuple = None, kwargs: dict = None,
                     logger: Optional[logging.Logger] = None) -> None:
    """Log function call with parameters.
    
    Utility function for logging function calls with their parameters.
    Useful for debugging and audit trails.
    
    Args:
        func_name: Name of the function being called
        args: Positional arguments
        kwargs: Keyword arguments  
        logger: Logger instance to use (creates one if None)
    """
    if logger is None:
        logger = get_logger(__name__)
    
    args_str = ", ".join(str(arg) for arg in (args or []))
    kwargs_str = ", ".join(f"{k}={v}" for k, v in (kwargs or {}).items())
    
    params = []
    if args_str:
        params.append(args_str)
    if kwargs_str:
        params.append(kwargs_str)
    
    params_str = ", ".join(params)
    logger.debug(f"Calling {func_name}({params_str})")


def log_performance(operation: str, duration: float, 
                   logger: Optional[logging.Logger] = None,
                   **kwargs) -> None:
    """Log performance metrics for an operation.
    
    Args:
        operation: Name of the operation
        duration: Duration in seconds
        logger: Logger instance to use
        **kwargs: Additional metrics to log
    """
    if logger is None:
        logger = get_logger(__name__)
    
    metrics = [f"duration={duration:.3f}s"]
    for key, value in kwargs.items():
        metrics.append(f"{key}={value}")
    
    metrics_str = ", ".join(metrics)
    logger.info(f"Performance - {operation}: {metrics_str}")


# Initialize logging when module is imported if not already done
if __name__ != "__main__":
    try:
        setup_logging()
    except Exception:
        # Silent fallback - logging will be initialized when first used
        pass


if __name__ == "__main__":
    # CLI interface for testing logging configuration
    import argparse
    
    parser = argparse.ArgumentParser(description="Test logging configuration")
    parser.add_argument("--level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Log level for testing")
    parser.add_argument("--file", action="store_true",
                       help="Enable file logging")
    args = parser.parse_args()
    
    # Test configuration
    config_override = {
        'default_level': args.level
    }
    
    if args.file:
        config_override['handlers'] = {
            'console': {'enabled': True, 'level': 'INFO', 'format': 'simple'},
            'file': {'enabled': True, 'level': 'DEBUG', 'format': 'detailed',
                    'filename': 'logs/test.log', 'max_bytes': 1024*1024, 'backup_count': 3}
        }
    
    setup_logging(config_override, force_reinit=True)
    
    # Test logging
    logger = get_logger(__name__)
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test performance logging
    import time
    start_time = time.time()
    time.sleep(0.1)
    log_performance("test_operation", time.time() - start_time, logger, 
                   memory_mb=50, records_processed=1000)
    
    print("Logging test completed!")