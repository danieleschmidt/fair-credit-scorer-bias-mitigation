"""Tests for the test runner module (run_tests.py).

This module tests the development infrastructure and test execution pipeline
to ensure reliable continuous integration and development environment setup.
"""

import os
import sys
import subprocess
import pytest
import inspect
from unittest.mock import patch, MagicMock, call
import tempfile


class TestRunTestsInfrastructure:
    """Test the test runner infrastructure and environment setup."""

    def test_src_directory_path_calculation(self):
        """Test that SRC_DIR is correctly calculated."""
        from src import run_tests
        
        # SRC_DIR should point to the src directory containing run_tests.py
        expected_src_dir = os.path.abspath(os.path.dirname(run_tests.__file__))
        assert run_tests.SRC_DIR == expected_src_dir
        
        # Should be an absolute path
        assert os.path.isabs(run_tests.SRC_DIR)
        
        # Should point to the src directory
        assert run_tests.SRC_DIR.endswith('src')

    def test_pythonpath_environment_setup(self):
        """Test that PYTHONPATH is properly configured."""
        from src import run_tests
        
        # PYTHONPATH should include SRC_DIR
        pythonpath = os.environ.get("PYTHONPATH", "")
        assert run_tests.SRC_DIR in pythonpath.split(os.pathsep)

    def test_sys_path_modification(self):
        """Test that sys.path is modified to include src directory."""
        from src import run_tests
        
        # SRC_DIR should be first in sys.path (inserted at position 0)
        assert run_tests.SRC_DIR in sys.path
        # Should be early in the path for priority
        src_index = sys.path.index(run_tests.SRC_DIR)
        assert src_index <= 1  # Should be at position 0 or 1


class TestRunTestsMainFunction:
    """Test the main function logic and subprocess calls."""

    @patch('subprocess.check_call')
    @patch('os.path.exists')
    @patch('pytest.main')
    def test_main_function_with_requirements_file(self, mock_pytest, mock_exists, mock_subprocess):
        """Test main function when requirements.txt exists."""
        from src.run_tests import main
        
        # Mock requirements.txt exists
        mock_exists.return_value = True
        mock_pytest.return_value = 0
        
        # Calculate project root based on actual run_tests.py location  
        from src import run_tests
        expected_project_root = os.path.abspath(os.path.join(run_tests.SRC_DIR, os.pardir))
        expected_requirements = os.path.join(expected_project_root, "requirements.txt")
        
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        # Should exit with pytest return code
        assert exc_info.value.code == 0
        
        # Verify subprocess calls in order
        expected_calls = [
            # Install requirements
            call([sys.executable, "-m", "pip", "install", "-r", expected_requirements], shell=False),
            # Install package in editable mode
            call([sys.executable, "-m", "pip", "install", "-e", expected_project_root], shell=False),
            # Install dev tools
            call([sys.executable, "-m", "pip", "install", "ruff==0.11.13", "bandit==1.8.5"], shell=False),
            # Run ruff
            call([sys.executable, "-m", "ruff", "check", "src", "--quiet"], shell=False),
            # Run bandit
            call([sys.executable, "-m", "bandit", "-r", "src"], shell=False),
        ]
        
        mock_subprocess.assert_has_calls(expected_calls, any_order=False)
        
        # Verify pytest was called with correct args
        mock_pytest.assert_called_once_with(["-ra", "--cov=src", "--cov-report=term-missing"])

    @patch('subprocess.check_call')
    @patch('os.path.exists')
    @patch('pytest.main')
    def test_main_function_without_requirements_file(self, mock_pytest, mock_exists, mock_subprocess):
        """Test main function when requirements.txt doesn't exist."""
        from src.run_tests import main
        
        # Mock requirements.txt doesn't exist
        mock_exists.return_value = False
        mock_pytest.return_value = 0
        
        from src import run_tests  
        expected_project_root = os.path.abspath(os.path.join(run_tests.SRC_DIR, os.pardir))
        
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        # Should exit with pytest return code
        assert exc_info.value.code == 0
        
        # Should skip requirements installation but do everything else
        expected_calls = [
            # Install package in editable mode (no requirements install call)
            call([sys.executable, "-m", "pip", "install", "-e", expected_project_root], shell=False),
            # Install dev tools
            call([sys.executable, "-m", "pip", "install", "ruff==0.11.13", "bandit==1.8.5"], shell=False),
            # Run ruff
            call([sys.executable, "-m", "ruff", "check", "src", "--quiet"], shell=False),
            # Run bandit
            call([sys.executable, "-m", "bandit", "-r", "src"], shell=False),
        ]
        
        mock_subprocess.assert_has_calls(expected_calls, any_order=False)

    @patch('subprocess.check_call')
    @patch('os.path.exists')
    @patch('pytest.main')
    def test_main_function_pytest_failure(self, mock_pytest, mock_exists, mock_subprocess):
        """Test main function when pytest fails."""
        from src.run_tests import main
        
        mock_exists.return_value = True
        mock_pytest.return_value = 1  # pytest failure
        
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        # Should exit with pytest return code (failure)
        assert exc_info.value.code == 1

    @patch('subprocess.check_call')
    @patch('os.path.exists') 
    @patch('pytest.main')
    def test_main_function_subprocess_failure(self, mock_pytest, mock_exists, mock_subprocess):
        """Test main function when subprocess calls fail."""
        from src.run_tests import main
        
        mock_exists.return_value = True
        # Mock subprocess failure
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, 'cmd')
        
        with pytest.raises(subprocess.CalledProcessError):
            main()
        
        # pytest should not be called if subprocess fails
        mock_pytest.assert_not_called()


class TestRunTestsSecurityAndCompliance:
    """Test security aspects and compliance of the test runner."""

    def test_subprocess_security_settings(self):
        """Test that subprocess calls use secure settings."""
        from src import run_tests
        
        # Verify the source code uses shell=False for security
        import inspect
        source = inspect.getsource(run_tests.main)
        
        # All subprocess.check_call should use shell=False
        assert "shell=False" in source
        # Should not use shell=True anywhere
        assert "shell=True" not in source

    def test_nosec_comments_present(self):
        """Test that appropriate nosec comments are present for bandit."""
        from src import run_tests
        
        source_lines = inspect.getsource(run_tests.main).split('\n')
        
        # Find lines with subprocess.check_call and check nearby lines for nosec
        subprocess_lines = []
        for i, line in enumerate(source_lines):
            if 'subprocess.check_call' in line:
                # Look at the next few lines for shell=False with nosec
                for j in range(i, min(i+4, len(source_lines))):
                    if 'shell=False' in source_lines[j] and 'nosec' in source_lines[j]:
                        subprocess_lines.append(source_lines[j])
                        break
        
        # Should have found nosec comments for all subprocess calls
        assert len(subprocess_lines) >= 4, f"Expected at least 4 nosec comments, found {len(subprocess_lines)}"

    def test_hardcoded_tool_versions(self):
        """Test that development tool versions are pinned for security."""
        from src import run_tests
        
        source = inspect.getsource(run_tests.main)
        
        # Should use pinned versions for security tools
        assert "ruff==0.11.13" in source
        assert "bandit==1.8.5" in source


class TestRunTestsIntegration:
    """Integration tests for the test runner module."""

    def test_module_importable(self):
        """Test that the run_tests module can be imported."""
        import src.run_tests
        assert hasattr(src.run_tests, 'main')
        assert callable(src.run_tests.main)

    def test_module_docstring_quality(self):
        """Test that the module has comprehensive documentation."""
        import src.run_tests
        
        assert src.run_tests.__doc__ is not None
        docstring = src.run_tests.__doc__
        
        # Should contain key information
        required_elements = [
            "Test runner",
            "dependency",
            "security",
            "coverage",
            "Usage",
            "Environment Variables",
            "Exit Codes"
        ]
        
        for element in required_elements:
            assert element in docstring, f"Missing '{element}' in module docstring"

    def test_main_function_docstring(self):
        """Test that the main function has appropriate documentation."""
        from src.run_tests import main
        
        assert main.__doc__ is not None
        assert "test suite" in main.__doc__.lower()
        assert "coverage" in main.__doc__.lower()