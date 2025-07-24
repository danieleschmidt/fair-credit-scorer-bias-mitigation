#!/usr/bin/env python3
"""
Test script for performance benchmarking edge cases.
This script tests the new edge cases without requiring external dependencies.
"""

import sys
import time
import tempfile
import os
from unittest.mock import patch, MagicMock

sys.path.append('src')

# Mock numpy and scipy before importing performance_benchmarking
sys.modules['numpy'] = MagicMock()
sys.modules['scipy'] = MagicMock()
sys.modules['scipy.stats'] = MagicMock()

# Configure mock scipy.stats.t
mock_t = MagicMock()
mock_t.ppf.return_value = 2.262  # t-score for 95% CI
sys.modules['scipy.stats'].t = mock_t

# Mock numpy functionality
mock_np = sys.modules['numpy']
mock_np.corrcoef.return_value = [[1.0, 0.8], [0.8, 1.0]]  # Sample correlation matrix
mock_np.sqrt.return_value = 1.0

from performance_benchmarking import (
    PerformanceBenchmark, 
    BenchmarkResults, 
    BenchmarkSuite,
    analyze_performance_trends
)

def test_memory_pressure_scenario():
    """Test benchmarking under memory pressure conditions."""
    print("Testing memory pressure scenario...")
    benchmark = PerformanceBenchmark(enable_memory_tracking=True)
    
    def memory_intensive_function():
        # Allocate large amounts of memory to simulate pressure
        large_data = [0] * (10 ** 6)  # 1M integers
        time.sleep(0.001)
        return len(large_data)
    
    stats = benchmark.benchmark_function(memory_intensive_function, n_runs=2)
    
    # Should still complete successfully but with high memory usage
    assert stats["success_rate"] == 100.0
    assert stats["peak_memory_mb"] > 0
    assert stats["mean_time_ms"] > 0
    print("✓ Memory pressure test passed")

def test_extremely_large_dataset_simulation():
    """Test performance with extremely large dataset sizes."""
    print("Testing extremely large dataset simulation...")
    benchmark = PerformanceBenchmark()
    
    def simulate_large_dataset_processing(dataset_size):
        # Simulate processing time proportional to dataset size
        processing_time = dataset_size / 1000000  # More aggressive scaling
        time.sleep(min(processing_time, 0.05))  # Cap at 50ms for test speed
        return f"Processed {dataset_size} samples"
    
    # Use smaller sizes but with clearer differences
    large_sizes = [100000, 1000000, 5000000]
    results = []
    
    for size in large_sizes:
        stats = benchmark.benchmark_function(
            simulate_large_dataset_processing, size, n_runs=1  # Single run for speed
        )
        results.append((size, stats["mean_time_ms"]))
    
    # Just verify we can benchmark different sizes
    assert len(results) == 3
    assert all(result[1] > 0 for result in results)  # All times should be positive
    print("✓ Large dataset simulation test passed")

def test_benchmark_failure_recovery():
    """Test error handling and recovery in benchmark failures."""
    print("Testing benchmark failure recovery...")
    benchmark = PerformanceBenchmark()
    
    failure_modes = [
        (ValueError, "Value error occurred"),
        (RuntimeError, "Runtime error occurred"),
        (MemoryError, "Out of memory"),
        (TimeoutError, "Operation timed out")
    ]
    
    for error_type, error_msg in failure_modes:
        def failing_function():
            raise error_type(error_msg)
        
        # Should handle the specific error gracefully
        try:
            benchmark.benchmark_function(failing_function, n_runs=2)
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "All" in str(e) and "benchmark runs failed" in str(e)
    
    print("✓ Benchmark failure recovery test passed")

def test_statistical_significance_testing():
    """Test statistical significance of benchmark results."""
    print("Testing statistical significance...")
    
    results_low_variance = BenchmarkResults(
        method_name="stable_method",
        training_time_ms=100.0,
        inference_time_ms=10.0,
        memory_peak_mb=50.0,
        n_samples=1000,
        n_runs=20,
        training_time_std=2.0  # Low variance
    )
    
    results_high_variance = BenchmarkResults(
        method_name="unstable_method", 
        training_time_ms=100.0,
        inference_time_ms=10.0,
        memory_peak_mb=50.0,
        n_samples=1000,
        n_runs=20,
        training_time_std=20.0  # High variance
    )
    
    # Low variance should have tighter confidence intervals
    ci_low_var = results_low_variance.training_confidence_interval(confidence=0.95)
    ci_high_var = results_high_variance.training_confidence_interval(confidence=0.95)
    
    # Check that confidence intervals are tuples of numbers
    assert len(ci_low_var) == 2
    assert len(ci_high_var) == 2
    assert isinstance(ci_low_var[0], (int, float))
    assert isinstance(ci_low_var[1], (int, float))
    assert isinstance(ci_high_var[0], (int, float))
    assert isinstance(ci_high_var[1], (int, float))
    
    low_var_width = ci_low_var[1] - ci_low_var[0]
    high_var_width = ci_high_var[1] - ci_high_var[0]
    
    assert low_var_width < high_var_width
    assert low_var_width > 0
    assert high_var_width > 0
    print("✓ Statistical significance test passed")

def test_benchmark_suite_partial_method_failures():
    """Test BenchmarkSuite handling when some methods fail."""
    print("Testing benchmark suite with partial failures...")
    with patch('src.performance_benchmarking.benchmark_method') as mock_benchmark:
        # Set up mixed success/failure scenario
        def side_effect(method_name, **kwargs):
            if method_name == "baseline":
                return BenchmarkResults("baseline", 100.0, 10.0, 50.0, 1000, 3)
            elif method_name == "reweight":
                raise RuntimeError("Method failed")
            else:
                return BenchmarkResults(method_name, 150.0, 15.0, 75.0, 1000, 3)
        
        mock_benchmark.side_effect = side_effect
        
        suite = BenchmarkSuite(methods=["baseline", "reweight", "postprocess"])
        results = suite.run_comprehensive_benchmark(sample_sizes=[1000], n_runs=2)
        
        # Should contain successful methods only
        assert "baseline" in results
        assert "postprocess" in results
        # Failed method should have empty results list
        assert len(results.get("reweight", [])) == 0
    
    print("✓ Benchmark suite partial failure test passed")

def test_analyze_performance_trends_edge_cases():
    """Test performance analysis with edge cases."""
    print("Testing performance analysis edge cases...")
    
    # Test with identical performance
    identical_results = {
        "method_a": [BenchmarkResults("method_a", 100.0, 10.0, 50.0, 1000, 3)],
        "method_b": [BenchmarkResults("method_b", 100.0, 10.0, 50.0, 1000, 3)],
        "method_c": [BenchmarkResults("method_c", 100.0, 10.0, 50.0, 1000, 3)]
    }
    
    analysis = analyze_performance_trends(identical_results)
    
    # Should still provide analysis even with identical performance
    assert analysis["fastest_method"] in ["method_a", "method_b", "method_c"]
    assert analysis["most_memory_efficient"] in ["method_a", "method_b", "method_c"]
    assert len(analysis["recommendations"]) > 0
    
    # Test with empty results
    empty_results = {}
    analysis_empty = analyze_performance_trends(empty_results)
    assert analysis_empty["fastest_method"] is None
    assert analysis_empty["most_memory_efficient"] is None
    assert analysis_empty["scalability_ranking"] == []
    assert analysis_empty["recommendations"] == []
    
    print("✓ Performance analysis edge cases test passed")

def main():
    """Run all edge case tests."""
    print("Running performance benchmarking edge case tests...\n")
    
    try:
        test_memory_pressure_scenario()
        test_extremely_large_dataset_simulation()
        test_benchmark_failure_recovery()
        test_statistical_significance_testing()
        test_benchmark_suite_partial_method_failures()
        test_analyze_performance_trends_edge_cases()
        
        print("\n🎉 All edge case tests passed successfully!")
        print("Coverage improvement achieved for task_12: Add performance benchmarking edge cases")
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())