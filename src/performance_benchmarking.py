"""Performance benchmarking utilities for fairness-aware credit scoring models.

This module provides comprehensive performance monitoring capabilities including
timing measurements, memory usage tracking, and statistical analysis of model
performance across different configurations and datasets.

Key Features:
- Training and inference time measurement
- Memory usage monitoring
- Statistical benchmarking with confidence intervals
- Comparative analysis across mitigation methods
- JSON export for automated monitoring

Classes:
    PerformanceBenchmark: Main benchmarking class for timing operations
    BenchmarkResults: Data class for storing benchmark results
    BenchmarkSuite: Comprehensive benchmarking across multiple configurations

Functions:
    benchmark_method: Time a specific fairness method
    benchmark_all_methods: Compare all available methods
    analyze_performance_trends: Statistical analysis of performance data

Usage:
    >>> from performance_benchmarking import PerformanceBenchmark, benchmark_method
    >>> 
    >>> # Benchmark a single method
    >>> results = benchmark_method("baseline", n_samples=10000, n_runs=5)
    >>> 
    >>> # Comprehensive benchmarking
    >>> from performance_benchmarking import BenchmarkSuite
    >>> suite = BenchmarkSuite()
    >>> all_results = suite.run_comprehensive_benchmark()
"""

import time
import logging
import statistics
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable, Any, Tuple
import json
from contextlib import contextmanager
import tracemalloc

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResults:
    """Data class for storing performance benchmark results.
    
    Attributes:
        method_name: Name of the fairness method being benchmarked
        training_time_ms: Average training time in milliseconds
        inference_time_ms: Average inference time in milliseconds
        memory_peak_mb: Peak memory usage in MB
        n_samples: Number of data samples used
        n_runs: Number of benchmark runs performed
        training_time_std: Standard deviation of training times
        inference_time_std: Standard deviation of inference times
        success_rate: Percentage of successful runs
        metadata: Additional benchmark metadata
    """
    method_name: str
    training_time_ms: float
    inference_time_ms: float
    memory_peak_mb: float
    n_samples: int
    n_runs: int
    training_time_std: float = 0.0
    inference_time_std: float = 0.0
    success_rate: float = 100.0
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert benchmark results to dictionary for JSON serialization."""
        return asdict(self)
    
    def training_confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for training time."""
        if self.n_runs < 2:
            return (self.training_time_ms, self.training_time_ms)
        
        # Using t-distribution for small samples
        import scipy.stats as stats
        t_score = stats.t.ppf((1 + confidence) / 2, self.n_runs - 1)
        margin = t_score * (self.training_time_std / np.sqrt(self.n_runs))
        
        return (
            self.training_time_ms - margin,
            self.training_time_ms + margin
        )


class PerformanceBenchmark:
    """Performance benchmarking utilities for fairness methods.
    
    This class provides timing and memory monitoring capabilities for evaluating
    the computational performance of different bias mitigation techniques.
    """
    
    def __init__(self, enable_memory_tracking: bool = True):
        """Initialize performance benchmark.
        
        Args:
            enable_memory_tracking: Whether to enable memory usage tracking
        """
        self.enable_memory_tracking = enable_memory_tracking
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @contextmanager
    def measure_time_and_memory(self):
        """Context manager for measuring execution time and memory usage.
        
        Yields:
            dict: Performance metrics including time_ms and memory_mb
        """
        results = {"time_ms": 0.0, "memory_mb": 0.0}
        
        # Start memory tracking if enabled
        if self.enable_memory_tracking:
            tracemalloc.start()
        
        start_time = time.perf_counter()
        
        try:
            yield results
        finally:
            # Calculate elapsed time
            end_time = time.perf_counter()
            results["time_ms"] = (end_time - start_time) * 1000
            
            # Calculate memory usage if enabled
            if self.enable_memory_tracking:
                current, peak = tracemalloc.get_traced_memory()
                results["memory_mb"] = peak / 1024 / 1024
                tracemalloc.stop()
    
    def benchmark_function(self, func: Callable, *args, n_runs: int = 5, **kwargs) -> Dict[str, float]:
        """Benchmark a function with multiple runs.
        
        Args:
            func: Function to benchmark
            *args: Positional arguments for function
            n_runs: Number of runs for averaging
            **kwargs: Keyword arguments for function
            
        Returns:
            Dictionary with timing statistics
        """
        times = []
        memory_usage = []
        successful_runs = 0
        
        for run in range(n_runs):
            try:
                with self.measure_time_and_memory() as metrics:
                    func(*args, **kwargs)  # Execute function but don't store result
                
                times.append(metrics["time_ms"])
                if self.enable_memory_tracking:
                    memory_usage.append(metrics["memory_mb"])
                
                successful_runs += 1
                
                self.logger.debug(f"Run {run + 1}/{n_runs}: {metrics['time_ms']:.2f}ms")
                
            except Exception as e:
                self.logger.warning(f"Benchmark run {run + 1} failed: {e}")
                continue
        
        if not times:
            raise RuntimeError(f"All {n_runs} benchmark runs failed")
        
        return {
            "mean_time_ms": statistics.mean(times),
            "std_time_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
            "min_time_ms": min(times),
            "max_time_ms": max(times),
            "mean_memory_mb": statistics.mean(memory_usage) if memory_usage else 0.0,
            "peak_memory_mb": max(memory_usage) if memory_usage else 0.0,
            "success_rate": (successful_runs / n_runs) * 100,
            "n_successful_runs": successful_runs
        }


def benchmark_method(method_name: str, n_samples: int = 10000, n_runs: int = 5, 
                    test_size: float = 0.3, random_state: int = 42) -> BenchmarkResults:
    """Benchmark a specific fairness method.
    
    Args:
        method_name: Name of method to benchmark ("baseline", "reweight", etc.)
        n_samples: Number of data samples to generate
        test_size: Proportion of data for testing
        n_runs: Number of benchmark runs
        random_state: Random seed for reproducibility
        
    Returns:
        BenchmarkResults object with performance metrics
    """
    try:
        from data_loader_preprocessor import generate_synthetic_credit_data
        from evaluate_fairness import run_pipeline
    except ImportError as e:
        raise ImportError(f"Required modules not available for benchmarking: {e}")
    
    logger.info(f"Benchmarking method '{method_name}' with {n_samples} samples, {n_runs} runs")
    
    # Generate data once for consistent benchmarking
    X, y, sensitive_features = generate_synthetic_credit_data(
        n_samples=n_samples, 
        random_state=random_state
    )
    
    benchmark = PerformanceBenchmark()
    
    # Benchmark training + inference as a complete pipeline
    def run_method():
        return run_pipeline(
            method=method_name,
            test_size=test_size,
            random_state=random_state,
            data_path=None,  # Use in-memory data
            X=X, y=y, sensitive_features=sensitive_features,
            verbose=False
        )
    
    try:
        perf_stats = benchmark.benchmark_function(run_method, n_runs=n_runs)
        
        return BenchmarkResults(
            method_name=method_name,
            training_time_ms=perf_stats["mean_time_ms"],
            inference_time_ms=0.0,  # Combined in training_time_ms for now
            memory_peak_mb=perf_stats["peak_memory_mb"],
            n_samples=n_samples,
            n_runs=perf_stats["n_successful_runs"],
            training_time_std=perf_stats["std_time_ms"],
            success_rate=perf_stats["success_rate"],
            metadata={
                "test_size": test_size,
                "random_state": random_state,
                "min_time_ms": perf_stats["min_time_ms"],
                "max_time_ms": perf_stats["max_time_ms"],
                "mean_memory_mb": perf_stats["mean_memory_mb"]
            }
        )
        
    except Exception as e:
        logger.error(f"Benchmarking failed for method '{method_name}': {e}")
        raise


class BenchmarkSuite:
    """Comprehensive benchmarking suite for fairness methods.
    
    This class provides systematic benchmarking across multiple methods,
    data sizes, and configurations to provide comprehensive performance insights.
    """
    
    def __init__(self, methods: Optional[List[str]] = None):
        """Initialize benchmark suite.
        
        Args:
            methods: List of methods to benchmark. If None, uses all available methods.
        """
        self.methods = methods or ["baseline", "reweight", "postprocess", "expgrad"]
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def run_comprehensive_benchmark(self, 
                                  sample_sizes: Optional[List[int]] = None,
                                  n_runs: int = 5) -> Dict[str, List[BenchmarkResults]]:
        """Run comprehensive benchmarking across methods and data sizes.
        
        Args:
            sample_sizes: List of sample sizes to test
            n_runs: Number of runs per configuration
            
        Returns:
            Dictionary mapping method names to benchmark results
        """
        if sample_sizes is None:
            sample_sizes = [1000, 5000, 10000]
        
        all_results = {}
        
        for method in self.methods:
            method_results = []
            
            for n_samples in sample_sizes:
                self.logger.info(f"Benchmarking {method} with {n_samples} samples")
                
                try:
                    result = benchmark_method(
                        method_name=method,
                        n_samples=n_samples,
                        n_runs=n_runs
                    )
                    method_results.append(result)
                    
                    self.logger.info(
                        f"{method} ({n_samples} samples): "
                        f"{result.training_time_ms:.1f}±{result.training_time_std:.1f}ms, "
                        f"{result.memory_peak_mb:.1f}MB peak"
                    )
                    
                except Exception as e:
                    self.logger.error(f"Failed to benchmark {method} with {n_samples} samples: {e}")
                    continue
            
            all_results[method] = method_results
        
        return all_results
    
    def save_results(self, results: Dict[str, List[BenchmarkResults]], 
                    output_path: str) -> None:
        """Save benchmark results to JSON file.
        
        Args:
            results: Benchmark results from run_comprehensive_benchmark
            output_path: Path to save JSON results
        """
        serializable_results = {}
        for method, method_results in results.items():
            serializable_results[method] = [r.to_dict() for r in method_results]
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Benchmark results saved to {output_path}")


def analyze_performance_trends(results: Dict[str, List[BenchmarkResults]]) -> Dict[str, Any]:
    """Analyze performance trends across methods and data sizes.
    
    Args:
        results: Benchmark results from BenchmarkSuite
        
    Returns:
        Dictionary with performance analysis and recommendations
    """
    analysis = {
        "fastest_method": None,
        "most_memory_efficient": None,
        "scalability_ranking": [],
        "recommendations": []
    }
    
    # Find fastest method (average across all sample sizes)
    method_avg_times = {}
    method_avg_memory = {}
    
    for method, method_results in results.items():
        if not method_results:
            continue
            
        avg_time = statistics.mean([r.training_time_ms for r in method_results])
        avg_memory = statistics.mean([r.memory_peak_mb for r in method_results])
        
        method_avg_times[method] = avg_time
        method_avg_memory[method] = avg_memory
    
    if method_avg_times:
        analysis["fastest_method"] = min(method_avg_times, key=method_avg_times.get)
        analysis["most_memory_efficient"] = min(method_avg_memory, key=method_avg_memory.get)
    
    # Scalability analysis (time complexity estimation)
    scalability_scores = {}
    for method, method_results in results.items():
        if len(method_results) < 2:
            continue
            
        sample_sizes = [r.n_samples for r in method_results]
        times = [r.training_time_ms for r in method_results]
        
        # Simple linear regression to estimate scaling factor
        if len(sample_sizes) >= 2:
            correlation = np.corrcoef(sample_sizes, times)[0, 1]
            scalability_scores[method] = correlation
    
    # Rank by scalability (lower correlation = better scalability)
    analysis["scalability_ranking"] = sorted(
        scalability_scores.items(), 
        key=lambda x: x[1]
    )
    
    # Generate recommendations
    recommendations = []
    
    if analysis["fastest_method"]:
        recommendations.append(
            f"For speed-critical applications, consider '{analysis['fastest_method']}' method"
        )
    
    if analysis["most_memory_efficient"]:
        recommendations.append(
            f"For memory-constrained environments, use '{analysis['most_memory_efficient']}' method"
        )
    
    if analysis["scalability_ranking"]:
        best_scaling = analysis["scalability_ranking"][0][0]
        recommendations.append(
            f"For large datasets, '{best_scaling}' shows the best scalability"
        )
    
    analysis["recommendations"] = recommendations
    
    return analysis


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark fairness methods")
    parser.add_argument("--method", default="baseline", 
                       choices=["baseline", "reweight", "postprocess", "expgrad"],
                       help="Method to benchmark")
    parser.add_argument("--samples", type=int, default=10000,
                       help="Number of samples to use")
    parser.add_argument("--runs", type=int, default=5,
                       help="Number of benchmark runs")
    parser.add_argument("--output", type=str,
                       help="Output file for benchmark results")
    parser.add_argument("--comprehensive", action="store_true",
                       help="Run comprehensive benchmark across all methods")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if args.comprehensive:
        suite = BenchmarkSuite()
        results = suite.run_comprehensive_benchmark(n_runs=args.runs)
        
        if args.output:
            suite.save_results(results, args.output)
        
        # Print analysis
        analysis = analyze_performance_trends(results)
        print("\n=== Performance Analysis ===")
        print(f"Fastest method: {analysis['fastest_method']}")
        print(f"Most memory efficient: {analysis['most_memory_efficient']}")
        print("\nRecommendations:")
        for rec in analysis['recommendations']:
            print(f"- {rec}")
            
    else:
        result = benchmark_method(
            method_name=args.method,
            n_samples=args.samples,
            n_runs=args.runs
        )
        
        print(f"\n=== Benchmark Results for {args.method} ===")
        print(f"Training time: {result.training_time_ms:.1f} ± {result.training_time_std:.1f} ms")
        print(f"Peak memory: {result.memory_peak_mb:.1f} MB")
        print(f"Success rate: {result.success_rate:.1f}%")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"Results saved to {args.output}")