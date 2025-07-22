"""Tests for performance benchmarking functionality.

This module provides comprehensive tests for the performance benchmarking utilities,
ensuring accurate timing measurements, memory tracking, and statistical analysis.
"""

import pytest
import json
import tempfile
import os
import time
from unittest.mock import patch, MagicMock

from src.performance_benchmarking import (
    PerformanceBenchmark,
    BenchmarkResults,
    BenchmarkSuite,
    benchmark_method,
    analyze_performance_trends
)


class TestBenchmarkResults:
    """Test cases for BenchmarkResults data class."""
    
    def test_benchmark_results_creation(self):
        """Test basic creation and attributes of BenchmarkResults."""
        results = BenchmarkResults(
            method_name="baseline",
            training_time_ms=100.5,
            inference_time_ms=10.2,
            memory_peak_mb=50.0,
            n_samples=1000,
            n_runs=5
        )
        
        assert results.method_name == "baseline"
        assert results.training_time_ms == 100.5
        assert results.inference_time_ms == 10.2
        assert results.memory_peak_mb == 50.0
        assert results.n_samples == 1000
        assert results.n_runs == 5
        assert results.success_rate == 100.0
    
    def test_to_dict_conversion(self):
        """Test conversion to dictionary for JSON serialization."""
        results = BenchmarkResults(
            method_name="reweight",
            training_time_ms=150.0,
            inference_time_ms=15.0,
            memory_peak_mb=75.0,
            n_samples=2000,
            n_runs=3,
            metadata={"test_key": "test_value"}
        )
        
        result_dict = results.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict["method_name"] == "reweight"
        assert result_dict["training_time_ms"] == 150.0
        assert result_dict["metadata"]["test_key"] == "test_value"
    
    def test_confidence_interval_calculation(self):
        """Test confidence interval calculation for timing statistics."""
        results = BenchmarkResults(
            method_name="baseline",
            training_time_ms=100.0,
            inference_time_ms=10.0,
            memory_peak_mb=50.0,
            n_samples=1000,
            n_runs=10,
            training_time_std=5.0
        )
        
        ci_lower, ci_upper = results.training_confidence_interval()
        
        # Should have non-zero confidence interval
        assert ci_lower < results.training_time_ms
        assert ci_upper > results.training_time_ms
        assert ci_upper > ci_lower
    
    def test_confidence_interval_single_run(self):
        """Test confidence interval with single run (should return point estimate)."""
        results = BenchmarkResults(
            method_name="baseline",
            training_time_ms=100.0,
            inference_time_ms=10.0,
            memory_peak_mb=50.0,
            n_samples=1000,
            n_runs=1,
            training_time_std=0.0
        )
        
        ci_lower, ci_upper = results.training_confidence_interval()
        
        # Single run should return the mean value for both bounds
        assert ci_lower == results.training_time_ms
        assert ci_upper == results.training_time_ms


class TestPerformanceBenchmark:
    """Test cases for PerformanceBenchmark class."""
    
    def test_benchmark_initialization(self):
        """Test PerformanceBenchmark initialization."""
        benchmark = PerformanceBenchmark(enable_memory_tracking=True)
        assert benchmark.enable_memory_tracking is True
        
        benchmark_no_memory = PerformanceBenchmark(enable_memory_tracking=False)
        assert benchmark_no_memory.enable_memory_tracking is False
    
    def test_measure_time_and_memory_context(self):
        """Test time and memory measurement context manager."""
        benchmark = PerformanceBenchmark()
        
        with benchmark.measure_time_and_memory() as metrics:
            # Simulate some work
            time.sleep(0.01)  # 10ms
        
        # Should measure some positive time
        assert metrics["time_ms"] > 0
        assert metrics["time_ms"] < 100  # Should be reasonable
        
        # Memory should be measured if enabled
        if benchmark.enable_memory_tracking:
            assert metrics["memory_mb"] >= 0
    
    def test_benchmark_function_with_simple_callable(self):
        """Test benchmarking a simple function."""
        benchmark = PerformanceBenchmark()
        
        def simple_function(x, y):
            time.sleep(0.005)  # 5ms
            return x + y
        
        stats = benchmark.benchmark_function(simple_function, 1, 2, n_runs=3)
        
        # Check that statistics are reasonable
        assert stats["mean_time_ms"] > 0
        assert stats["success_rate"] == 100.0
        assert stats["n_successful_runs"] == 3
        assert "std_time_ms" in stats
        assert "min_time_ms" in stats
        assert "max_time_ms" in stats
    
    def test_benchmark_function_with_failures(self):
        """Test benchmarking with some failed runs."""
        benchmark = PerformanceBenchmark()
        
        call_count = 0
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ValueError("Simulated failure")
            time.sleep(0.001)
            return "success"
        
        stats = benchmark.benchmark_function(failing_function, n_runs=5)
        
        # Should handle failures gracefully
        assert stats["success_rate"] == 60.0  # 3/5 successful
        assert stats["n_successful_runs"] == 3
        assert stats["mean_time_ms"] > 0
    
    def test_benchmark_function_all_failures(self):
        """Test benchmarking when all runs fail."""
        benchmark = PerformanceBenchmark()
        
        def always_failing_function():
            raise ValueError("Always fails")
        
        with pytest.raises(RuntimeError, match="All .* benchmark runs failed"):
            benchmark.benchmark_function(always_failing_function, n_runs=3)


class TestBenchmarkMethod:
    """Test cases for benchmark_method function."""
    
    @patch('data_loader_preprocessor.generate_synthetic_credit_data')
    @patch('evaluate_fairness.run_pipeline')
    def test_benchmark_method_success(self, mock_run_pipeline, mock_generate_data):
        """Test successful benchmarking of a method."""
        # Mock data generation
        mock_generate_data.return_value = (
            [[1, 2, 3]] * 100,  # X
            [0, 1] * 50,        # y  
            [0, 1] * 50         # sensitive_features
        )
        
        # Mock pipeline execution
        mock_run_pipeline.return_value = {
            "overall_metrics": {"accuracy": 0.8},
            "group_metrics": {}
        }
        
        result = benchmark_method(
            method_name="baseline",
            n_samples=100,
            n_runs=2
        )
        
        assert isinstance(result, BenchmarkResults)
        assert result.method_name == "baseline"
        assert result.n_samples == 100
        assert result.training_time_ms > 0
        assert result.success_rate > 0
        
        # Verify mocks were called correctly
        mock_generate_data.assert_called_once()
        assert mock_run_pipeline.call_count == 2  # n_runs
    
    @patch('data_loader_preprocessor.generate_synthetic_credit_data')
    @patch('evaluate_fairness.run_pipeline')
    def test_benchmark_method_with_failures(self, mock_run_pipeline, mock_generate_data):
        """Test benchmarking with some pipeline failures."""
        # Mock data generation
        mock_generate_data.return_value = (
            [[1, 2, 3]] * 100,
            [0, 1] * 50,
            [0, 1] * 50
        )
        
        # Mock pipeline with some failures
        side_effects = [
            RuntimeError("Pipeline failed"),
            {"overall_metrics": {"accuracy": 0.8}},
            {"overall_metrics": {"accuracy": 0.85}}
        ]
        mock_run_pipeline.side_effect = side_effects
        
        result = benchmark_method(
            method_name="reweight",
            n_samples=100,
            n_runs=3
        )
        
        # Should handle partial failures
        assert result.success_rate < 100.0
        assert result.n_runs == 2  # Only successful runs counted


class TestBenchmarkSuite:
    """Test cases for BenchmarkSuite class."""
    
    def test_benchmark_suite_initialization(self):
        """Test BenchmarkSuite initialization."""
        # Default methods
        suite = BenchmarkSuite()
        assert "baseline" in suite.methods
        assert "reweight" in suite.methods
        
        # Custom methods
        custom_methods = ["baseline", "reweight"]
        custom_suite = BenchmarkSuite(methods=custom_methods)
        assert custom_suite.methods == custom_methods
    
    @patch('src.performance_benchmarking.benchmark_method')
    def test_run_comprehensive_benchmark(self, mock_benchmark_method):
        """Test comprehensive benchmarking across methods and sizes."""
        # Mock benchmark results
        def create_mock_result(method, n_samples):
            return BenchmarkResults(
                method_name=method,
                training_time_ms=100.0,
                inference_time_ms=10.0,
                memory_peak_mb=50.0,
                n_samples=n_samples,
                n_runs=3
            )
        
        mock_benchmark_method.side_effect = lambda method_name, n_samples, **kwargs: \
            create_mock_result(method_name, n_samples)
        
        suite = BenchmarkSuite(methods=["baseline", "reweight"])
        results = suite.run_comprehensive_benchmark(
            sample_sizes=[1000, 2000],
            n_runs=3
        )
        
        # Check structure of results
        assert "baseline" in results
        assert "reweight" in results
        assert len(results["baseline"]) == 2  # Two sample sizes
        assert len(results["reweight"]) == 2
        
        # Verify benchmark_method was called correctly
        expected_calls = 4  # 2 methods × 2 sample sizes
        assert mock_benchmark_method.call_count == expected_calls
    
    def test_save_results(self):
        """Test saving benchmark results to JSON file."""
        suite = BenchmarkSuite()
        
        # Create sample results
        results = {
            "baseline": [
                BenchmarkResults(
                    method_name="baseline",
                    training_time_ms=100.0,
                    inference_time_ms=10.0,
                    memory_peak_mb=50.0,
                    n_samples=1000,
                    n_runs=3
                )
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            suite.save_results(results, temp_path)
            
            # Verify file was created and contains correct data
            assert os.path.exists(temp_path)
            
            with open(temp_path, 'r') as f:
                loaded_data = json.load(f)
            
            assert "baseline" in loaded_data
            assert loaded_data["baseline"][0]["method_name"] == "baseline"
            assert loaded_data["baseline"][0]["training_time_ms"] == 100.0
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestAnalyzePerformanceTrends:
    """Test cases for analyze_performance_trends function."""
    
    def test_analyze_performance_trends_basic(self):
        """Test basic performance trend analysis."""
        # Create sample benchmark results
        results = {
            "baseline": [
                BenchmarkResults("baseline", 100.0, 10.0, 50.0, 1000, 3),
                BenchmarkResults("baseline", 200.0, 20.0, 100.0, 2000, 3)
            ],
            "reweight": [
                BenchmarkResults("reweight", 150.0, 15.0, 75.0, 1000, 3),
                BenchmarkResults("reweight", 300.0, 30.0, 150.0, 2000, 3)
            ]
        }
        
        analysis = analyze_performance_trends(results)
        
        # Check that analysis contains expected keys
        assert "fastest_method" in analysis
        assert "most_memory_efficient" in analysis
        assert "scalability_ranking" in analysis
        assert "recommendations" in analysis
        
        # Baseline should be faster on average
        assert analysis["fastest_method"] == "baseline"
        assert analysis["most_memory_efficient"] == "baseline"
        
        # Should have recommendations
        assert len(analysis["recommendations"]) > 0
        assert any("baseline" in rec for rec in analysis["recommendations"])
    
    def test_analyze_performance_trends_empty_results(self):
        """Test analysis with empty results."""
        results = {}
        
        analysis = analyze_performance_trends(results)
        
        # Should handle empty results gracefully
        assert analysis["fastest_method"] is None
        assert analysis["most_memory_efficient"] is None
        assert analysis["scalability_ranking"] == []
        assert analysis["recommendations"] == []
    
    def test_analyze_performance_trends_single_method(self):
        """Test analysis with only one method."""
        results = {
            "baseline": [
                BenchmarkResults("baseline", 100.0, 10.0, 50.0, 1000, 3)
            ]
        }
        
        analysis = analyze_performance_trends(results)
        
        # Should identify the single method as best
        assert analysis["fastest_method"] == "baseline"
        assert analysis["most_memory_efficient"] == "baseline"


class TestIntegration:
    """Integration tests for performance benchmarking."""
    
    @pytest.mark.slow
    def test_end_to_end_benchmarking(self):
        """Test complete benchmarking workflow with real data (marked as slow)."""
        # This test uses actual data generation and model training
        # It's marked as slow and may be skipped in fast test runs
        
        try:
            result = benchmark_method(
                method_name="baseline",
                n_samples=100,  # Small sample for speed
                n_runs=2
            )
            
            # Verify we got meaningful results
            assert isinstance(result, BenchmarkResults)
            assert result.method_name == "baseline"
            assert result.training_time_ms > 0
            assert result.n_samples == 100
            assert result.success_rate > 0
            
        except ImportError:
            # Skip if dependencies are not available
            pytest.skip("Required dependencies not available for integration test")
    
    def test_cli_interface_simulation(self):
        """Test CLI interface behavior through direct function calls."""
        import sys
        from unittest.mock import patch
        
        # Test comprehensive benchmarking flag simulation
        with patch('src.performance_benchmarking.BenchmarkSuite') as mock_suite_class:
            mock_suite = MagicMock()
            mock_suite_class.return_value = mock_suite
            mock_suite.run_comprehensive_benchmark.return_value = {
                "baseline": [BenchmarkResults("baseline", 100.0, 10.0, 50.0, 1000, 3)]
            }
            
            # Import and test the main functionality
            from src.performance_benchmarking import analyze_performance_trends
            
            # Simulate running the analysis
            results = {"baseline": [BenchmarkResults("baseline", 100.0, 10.0, 50.0, 1000, 3)]}
            analysis = analyze_performance_trends(results)
            
            assert analysis["fastest_method"] == "baseline"


# Fixtures for common test data
@pytest.fixture
def sample_benchmark_results():
    """Fixture providing sample benchmark results for testing."""
    return {
        "baseline": [
            BenchmarkResults("baseline", 100.0, 10.0, 50.0, 1000, 3),
            BenchmarkResults("baseline", 200.0, 20.0, 100.0, 2000, 3)
        ],
        "reweight": [
            BenchmarkResults("reweight", 150.0, 15.0, 75.0, 1000, 3),
            BenchmarkResults("reweight", 300.0, 30.0, 150.0, 2000, 3)
        ]
    }


@pytest.fixture
def performance_benchmark():
    """Fixture providing a PerformanceBenchmark instance."""
    return PerformanceBenchmark(enable_memory_tracking=True)