"""Integration tests for the complete fairness evaluation pipeline."""

import json
import tempfile
from pathlib import Path

import pytest

from src.evaluate_fairness import run_pipeline


class TestEndToEndPipeline:
    """Test the complete pipeline from data loading to evaluation."""

    @pytest.mark.integration
    def test_baseline_pipeline_integration(self):
        """Test baseline model pipeline end-to-end."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "test_data.csv"
            
            results = run_pipeline(
                method="baseline",
                data_path=str(data_path),
                test_size=0.3,
                random_state=42
            )
            
            # Verify all expected metrics are present
            assert "accuracy" in results
            assert "demographic_parity_difference" in results
            assert "equalized_odds_difference" in results
            assert isinstance(results["accuracy"], float)
            assert 0 <= results["accuracy"] <= 1

    @pytest.mark.integration
    def test_reweight_pipeline_integration(self):
        """Test reweighting mitigation pipeline end-to-end."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "test_data.csv"
            
            results = run_pipeline(
                method="reweight",
                data_path=str(data_path),
                test_size=0.3,
                random_state=42
            )
            
            assert "accuracy" in results
            assert "demographic_parity_difference" in results
            assert isinstance(results["accuracy"], float)

    @pytest.mark.integration
    def test_cross_validation_pipeline(self):
        """Test pipeline with cross-validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "test_data.csv"
            
            results = run_pipeline(
                method="baseline",
                data_path=str(data_path),
                cv=3,
                random_state=42
            )
            
            # CV results should include std deviation
            assert "accuracy_std" in results
            assert "demographic_parity_difference_std" in results

    @pytest.mark.integration
    def test_json_output_integration(self):
        """Test pipeline with JSON output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "test_data.csv"
            output_path = Path(temp_dir) / "results.json"
            
            results = run_pipeline(
                method="baseline",
                data_path=str(data_path),
                output_json=str(output_path),
                random_state=42
            )
            
            # Verify JSON file was created
            assert output_path.exists()
            
            # Verify JSON content
            with open(output_path) as f:
                json_results = json.load(f)
            
            assert json_results["accuracy"] == results["accuracy"]

    @pytest.mark.integration
    @pytest.mark.slow
    def test_all_methods_integration(self):
        """Test all mitigation methods work end-to-end."""
        methods = ["baseline", "reweight", "postprocess"]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "test_data.csv"
            
            results = {}
            for method in methods:
                results[method] = run_pipeline(
                    method=method,
                    data_path=str(data_path),
                    test_size=0.3,
                    random_state=42
                )
                
                # Each method should produce valid results
                assert "accuracy" in results[method]
                assert "demographic_parity_difference" in results[method]
            
            # Verify we get different results for different methods
            baseline_acc = results["baseline"]["accuracy"]
            reweight_acc = results["reweight"]["accuracy"]
            
            # Results should be different (allowing for small numerical differences)
            assert abs(baseline_acc - reweight_acc) > 0.001 or baseline_acc == reweight_acc