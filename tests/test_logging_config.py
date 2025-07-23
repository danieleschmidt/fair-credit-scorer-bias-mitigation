"""Fixed comprehensive tests for logging configuration module."""

import logging
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
from io import StringIO

try:
    from src.logging_config import (
        setup_logging, 
        get_logger, 
        ContextFilter, 
        _setup_console_handler,
        _setup_file_handler,
        _configure_module_loggers,
        _disable_loggers,
        _get_fallback_config
    )
except ImportError:
    from logging_config import (
        setup_logging, 
        get_logger, 
        ContextFilter, 
        _setup_console_handler,
        _setup_file_handler,
        _configure_module_loggers,
        _disable_loggers,
        _get_fallback_config
    )


def create_mock_config(with_modules=False, with_disable=False):
    """Helper to create properly structured mock configuration."""
    mock_config = MagicMock()
    mock_config.logging.default_level = "INFO"
    mock_config.logging.handlers.console.enabled = True
    mock_config.logging.handlers.console.level = "INFO" 
    mock_config.logging.handlers.console.format = "simple"
    mock_config.logging.handlers.file.enabled = False
    mock_config.logging.format.simple = "%(levelname)s - %(message)s"
    mock_config.logging.date_format = "%Y-%m-%d %H:%M:%S"
    
    # Handle optional attributes by using spec to control hasattr behavior
    if not with_modules:
        # Create a mock that doesn't have 'modules' attribute
        mock_logging = MagicMock(spec=['default_level', 'handlers', 'format', 'date_format'])
        for attr in ['default_level', 'handlers', 'format', 'date_format']:
            setattr(mock_logging, attr, getattr(mock_config.logging, attr))
        mock_config.logging = mock_logging
    else:
        mock_config.logging.modules = {"test.module": "DEBUG"}
        
    if not with_disable:
        # Don't add disable attribute if not needed
        pass
    else:
        mock_config.logging.disable = ["noisy.library"]
        
    return mock_config


class TestContextFilter:
    """Test the ContextFilter class."""
    
    def test_context_filter_creation_default(self):
        """Test creating context filter with default context."""
        filter_obj = ContextFilter()
        assert filter_obj.context == {}
    
    def test_context_filter_creation_with_context(self):
        """Test creating context filter with provided context."""
        context = {"user_id": "123", "session": "abc"}
        filter_obj = ContextFilter(context)
        assert filter_obj.context == context
    
    def test_context_filter_adds_context_to_record(self):
        """Test that filter adds context information to log records."""
        context = {"user_id": "123", "request_id": "xyz"}
        filter_obj = ContextFilter(context)
        
        record = MagicMock()
        result = filter_obj.filter(record)
        
        assert result is True
        assert record.user_id == "123" 
        assert record.request_id == "xyz"


class TestSetupLogging:
    """Test the main setup_logging function."""
    
    @pytest.fixture(autouse=True)
    def reset_logging(self):
        """Reset logging state before each test."""
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Reset initialization flag
        import src.logging_config as logging_config
        logging_config._logging_initialized = False
        
        yield
        
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        logging_config._logging_initialized = False
    
    @patch('src.logging_config.get_config')
    def test_setup_logging_success(self, mock_get_config):
        """Test successful logging setup."""
        mock_get_config.return_value = create_mock_config()
        
        setup_logging()
        
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO
        assert len(root_logger.handlers) >= 1
    
    @patch('src.logging_config.get_config')
    def test_setup_logging_fallback_on_config_error(self, mock_get_config):
        """Test fallback configuration when config fails."""
        mock_get_config.side_effect = Exception("Config failed")
        
        setup_logging()
        
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO


class TestConsoleHandler:
    """Test console handler setup."""
    
    def test_setup_console_handler_enabled(self):
        """Test setting up enabled console handler."""
        mock_config = MagicMock()
        mock_config.handlers.console.enabled = True
        mock_config.handlers.console.level = "DEBUG"
        mock_config.handlers.console.format = "simple"
        mock_config.format.simple = "%(levelname)s - %(message)s"
        mock_config.date_format = "%Y-%m-%d %H:%M:%S"
        
        root_logger = logging.getLogger()
        initial_count = len(root_logger.handlers)
        
        _setup_console_handler(root_logger, mock_config)
        
        assert len(root_logger.handlers) == initial_count + 1
        new_handler = root_logger.handlers[-1]
        assert isinstance(new_handler, logging.StreamHandler)
        assert new_handler.level == logging.DEBUG
    
    def test_setup_console_handler_disabled(self):
        """Test console handler when disabled."""
        mock_config = MagicMock()
        mock_config.handlers.console.enabled = False
        
        root_logger = logging.getLogger()
        initial_count = len(root_logger.handlers)
        
        _setup_console_handler(root_logger, mock_config)
        
        assert len(root_logger.handlers) == initial_count


class TestFileHandler:
    """Test file handler setup."""
    
    def test_setup_file_handler_enabled(self):
        """Test setting up enabled file handler."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            
            mock_config = MagicMock()
            mock_config.handlers.file.enabled = True
            mock_config.handlers.file.filename = str(log_file)
            mock_config.handlers.file.level = "INFO"
            mock_config.handlers.file.format = "standard"
            mock_config.handlers.file.max_bytes = 1048576
            mock_config.handlers.file.backup_count = 3
            mock_config.format.standard = "%(asctime)s - %(levelname)s - %(message)s"
            mock_config.date_format = "%Y-%m-%d %H:%M:%S"
            
            root_logger = logging.getLogger()
            initial_count = len(root_logger.handlers)
            
            _setup_file_handler(root_logger, mock_config)
            
            assert len(root_logger.handlers) == initial_count + 1
            new_handler = root_logger.handlers[-1]
            assert isinstance(new_handler, logging.handlers.RotatingFileHandler)
    
    def test_setup_file_handler_disabled(self):
        """Test file handler when disabled."""
        mock_config = MagicMock()
        mock_config.handlers.file.enabled = False
        
        root_logger = logging.getLogger()
        initial_count = len(root_logger.handlers)
        
        _setup_file_handler(root_logger, mock_config)
        
        assert len(root_logger.handlers) == initial_count


class TestModuleLoggerConfiguration:
    """Test module-specific logger configuration."""
    
    def test_configure_module_loggers_dict_style(self):
        """Test configuring module loggers with dictionary."""
        module_levels = {
            "test.module1": "DEBUG",
            "test.module2": "WARNING"
        }
        
        _configure_module_loggers(module_levels)
        
        logger1 = logging.getLogger("test.module1")
        logger2 = logging.getLogger("test.module2")
        
        assert logger1.level == logging.DEBUG
        assert logger2.level == logging.WARNING
    
    def test_configure_module_loggers_config_section_style(self):
        """Test configuring module loggers with ConfigSection-like object."""
        mock_module_levels = MagicMock()
        mock_module_levels.__dict__ = {
            "test.module": "ERROR",
            "_private": "DEBUG"  # Should be ignored
        }
        
        _configure_module_loggers(mock_module_levels)
        
        test_logger = logging.getLogger("test.module")
        assert test_logger.level == logging.ERROR


class TestDisableLoggers:
    """Test logger disabling functionality."""
    
    def test_disable_loggers(self):
        """Test disabling specified loggers."""
        loggers_to_disable = ["noisy.module", "external.library"]
        
        _disable_loggers(loggers_to_disable)
        
        noisy_logger = logging.getLogger("noisy.module")
        external_logger = logging.getLogger("external.library")
        
        assert noisy_logger.level == logging.CRITICAL
        assert external_logger.level == logging.CRITICAL


class TestFallbackConfiguration:
    """Test fallback configuration functionality."""
    
    def test_get_fallback_config_structure(self):
        """Test that fallback config has expected structure."""
        fallback = _get_fallback_config()
        
        assert hasattr(fallback, 'default_level')
        assert hasattr(fallback, 'format') 
        assert hasattr(fallback, 'date_format')
        assert hasattr(fallback, 'handlers')
        
        assert fallback.default_level == "INFO"
        assert fallback.date_format == "%Y-%m-%d %H:%M:%S"
    
    def test_get_fallback_config_formats(self):
        """Test fallback config format options."""
        fallback = _get_fallback_config()
        
        assert hasattr(fallback.format, 'simple')
        assert hasattr(fallback.format, 'standard')
        assert hasattr(fallback.format, 'detailed')
        
        assert '%(levelname)s' in fallback.format.simple
        assert '%(asctime)s' in fallback.format.standard
        assert '%(funcName)s' in fallback.format.detailed


class TestGetLogger:
    """Test the get_logger function."""
    
    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a logger instance."""
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"
    
    def test_get_logger_same_instance(self):
        """Test that same logger name returns same instance."""
        logger1 = get_logger("test.module")
        logger2 = get_logger("test.module")
        assert logger1 is logger2


class TestLoggingIntegration:
    """Integration tests for the logging system."""
    
    @pytest.fixture(autouse=True)
    def reset_logging_integration(self):
        """Reset logging for integration tests."""
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        import src.logging_config as logging_config
        logging_config._logging_initialized = False
        
        yield
        
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        logging_config._logging_initialized = False
    
    @patch('src.logging_config.get_config')  
    def test_initialization_tracking(self, mock_get_config):
        """Test that logging initialization is properly tracked."""
        import src.logging_config as logging_config
        
        assert logging_config._logging_initialized is False
        
        mock_get_config.return_value = create_mock_config()
        setup_logging()
        
        assert logging_config._logging_initialized is True
    
    @patch('src.logging_config.get_config')
    def test_prevent_duplicate_initialization(self, mock_get_config):
        """Test that duplicate initialization is prevented."""
        mock_get_config.return_value = create_mock_config()
        
        # First setup
        setup_logging()
        
        # Mock the internal functions to detect if they're called again
        with patch('src.logging_config._setup_console_handler') as mock_console:
            setup_logging()  # Should not call setup again
            mock_console.assert_not_called()
        
        # But should call with force_reinit=True
        with patch('src.logging_config._setup_console_handler') as mock_console:
            setup_logging(force_reinit=True)
            mock_console.assert_called_once()