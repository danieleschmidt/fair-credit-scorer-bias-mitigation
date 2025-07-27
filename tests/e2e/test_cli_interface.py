"""End-to-end tests for the CLI interface."""

import json
import subprocess
import tempfile
from pathlib import Path

import pytest


class TestCLIInterface:
    """Test the CLI interface end-to-end."""

    def test_cli_help(self):
        """Test CLI help command."""
        result = subprocess.run(
            ["fairness-eval", "--help"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()
        assert "--method" in result.stdout

    def test_cli_baseline_execution(self):
        """Test CLI execution with baseline method."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "cli_test_data.csv"
            
            result = subprocess.run([
                "fairness-eval",
                "--method", "baseline",
                "--data-path", str(data_path),
                "--test-size", "0.3",
                "--random-state", "42"
            ], capture_output=True, text=True)
            
            assert result.returncode == 0
            assert "Accuracy:" in result.stdout
            assert "Demographic Parity Difference:" in result.stdout

    def test_cli_json_output(self):
        """Test CLI with JSON output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "cli_test_data.csv"
            output_path = Path(temp_dir) / "cli_results.json"
            
            result = subprocess.run([
                "fairness-eval",
                "--method", "baseline",
                "--data-path", str(data_path),
                "--output-json", str(output_path),
                "--random-state", "42"
            ], capture_output=True, text=True)
            
            assert result.returncode == 0
            assert output_path.exists()
            
            # Verify JSON content
            with open(output_path) as f:
                data = json.load(f)
            
            assert "accuracy" in data
            assert "demographic_parity_difference" in data

    def test_cli_cross_validation(self):
        """Test CLI with cross-validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "cli_cv_data.csv"
            
            result = subprocess.run([
                "fairness-eval",
                "--method", "baseline",
                "--data-path", str(data_path),
                "--cv", "3",
                "--random-state", "42"
            ], capture_output=True, text=True)
            
            assert result.returncode == 0
            assert "Cross-validation results" in result.stdout or "accuracy" in result.stdout.lower()

    def test_cli_verbose_mode(self):
        """Test CLI verbose mode."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "cli_verbose_data.csv"
            
            result = subprocess.run([
                "fairness-eval",
                "--method", "baseline",
                "--data-path", str(data_path),
                "--verbose",
                "--random-state", "42"
            ], capture_output=True, text=True)
            
            assert result.returncode == 0
            # Verbose mode should produce more output
            assert len(result.stdout) > 100

    def test_cli_invalid_method(self):
        """Test CLI with invalid method."""
        result = subprocess.run([
            "fairness-eval",
            "--method", "invalid_method"
        ], capture_output=True, text=True)
        
        # Should fail with non-zero exit code
        assert result.returncode != 0

    def test_repo_hygiene_cli_help(self):
        """Test repo hygiene bot CLI help."""
        result = subprocess.run([
            "repo-hygiene-bot", "--help"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()

    @pytest.mark.slow
    def test_cli_all_methods(self):
        """Test CLI with all available methods."""
        methods = ["baseline", "reweight", "postprocess"]
        
        for method in methods:
            with tempfile.TemporaryDirectory() as temp_dir:
                data_path = Path(temp_dir) / f"cli_{method}_data.csv"
                
                result = subprocess.run([
                    "fairness-eval",
                    "--method", method,
                    "--data-path", str(data_path),
                    "--test-size", "0.3",
                    "--random-state", "42"
                ], capture_output=True, text=True)
                
                assert result.returncode == 0, f"Method {method} failed: {result.stderr}"
                assert "Accuracy:" in result.stdout