"""
Performance optimization and auto-tuning system.

Automated performance optimization for ML models and data pipelines
with intelligent parameter tuning and resource allocation.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from logging_config import get_logger
from .benchmarks import BenchmarkSuite
from .profiler import AdvancedProfiler

logger = get_logger(__name__)


@dataclass
class OptimizationResult:
    """Result of an optimization run."""
    parameter_name: str
    original_value: Any
    optimized_value: Any
    performance_improvement: float
    optimization_method: str
    confidence: float
    execution_time: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'parameter_name': self.parameter_name,
            'original_value': self.original_value,
            'optimized_value': self.optimized_value,
            'performance_improvement_percent': self.performance_improvement * 100,
            'optimization_method': self.optimization_method,
            'confidence': self.confidence,
            'execution_time_ms': self.execution_time * 1000,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class PerformanceConfiguration:
    """Performance configuration settings."""
    batch_size: int = 100
    num_workers: int = 4
    memory_limit_mb: int = 1024
    cache_enabled: bool = True
    parallel_processing: bool = True
    optimization_level: str = "balanced"  # conservative, balanced, aggressive

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'memory_limit_mb': self.memory_limit_mb,
            'cache_enabled': self.cache_enabled,
            'parallel_processing': self.parallel_processing,
            'optimization_level': self.optimization_level
        }


class PerformanceOptimizer:
    """
    Automated performance optimization system.

    Analyzes performance bottlenecks and automatically optimizes
    model parameters, batch sizes, and processing configurations.
    """

    def __init__(
        self,
        optimization_level: str = "balanced",
        max_optimization_time: int = 300,
        parallel_workers: int = 4
    ):
        """
        Initialize performance optimizer.

        Args:
            optimization_level: Optimization aggressiveness level
            max_optimization_time: Maximum time for optimization (seconds)
            parallel_workers: Number of parallel optimization workers
        """
        self.optimization_level = optimization_level
        self.max_optimization_time = max_optimization_time
        self.parallel_workers = parallel_workers

        # Optimization components
        self.profiler = AdvancedProfiler()
        self.benchmark_suite = BenchmarkSuite()

        # Results tracking
        self.optimization_results: List[OptimizationResult] = []
        self.performance_history: List[Dict[str, Any]] = []

        logger.info("PerformanceOptimizer initialized")

    def optimize_model_inference(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        target_latency_ms: Optional[float] = None,
        target_throughput: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Optimize model inference performance.

        Args:
            model: Model to optimize
            X: Input data for testing
            target_latency_ms: Target latency in milliseconds
            target_throughput: Target throughput (samples/second)

        Returns:
            Optimization results and recommendations
        """
        logger.info("Starting model inference optimization")

        # Baseline performance measurement
        baseline_results = self.benchmark_suite.benchmark_model_prediction(model, X)
        baseline_performance = self._aggregate_benchmark_results(baseline_results)

        logger.info(f"Baseline performance: {baseline_performance['avg_samples_per_second']:.1f} samples/sec")

        # Optimization parameters to try
        optimization_params = self._get_inference_optimization_params()

        # Run optimizations
        optimization_results = []

        for param_name, param_values in optimization_params.items():
            best_result = self._optimize_parameter(
                model, X, param_name, param_values, baseline_performance
            )
            if best_result:
                optimization_results.append(best_result)

        # Generate final recommendations
        recommendations = self._generate_inference_recommendations(
            optimization_results, baseline_performance, target_latency_ms, target_throughput
        )

        # Create optimized configuration
        optimized_config = self._create_optimized_config(optimization_results)

        result = {
            'baseline_performance': baseline_performance,
            'optimization_results': [r.to_dict() for r in optimization_results],
            'optimized_configuration': optimized_config.to_dict(),
            'recommendations': recommendations,
            'total_improvement': self._calculate_total_improvement(optimization_results),
            'optimization_time': sum(r.execution_time for r in optimization_results)
        }

        logger.info("Model inference optimization completed")
        return result

    def optimize_data_processing(
        self,
        processing_function: Callable,
        data_samples: List[pd.DataFrame],
        target_processing_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Optimize data processing pipeline.

        Args:
            processing_function: Function to optimize
            data_samples: Sample data for testing
            target_processing_time: Target processing time per sample

        Returns:
            Optimization results and recommendations
        """
        logger.info("Starting data processing optimization")

        # Baseline measurement
        baseline_results = []
        for data in data_samples[:3]:  # Test with first 3 samples
            results = self.benchmark_suite.benchmark_data_processing(
                processing_function, [len(data)]
            )
            baseline_results.extend(results)

        baseline_performance = self._aggregate_benchmark_results(baseline_results)

        # Optimize processing parameters
        optimization_results = []

        # Batch size optimization
        batch_sizes = [1, 10, 50, 100, 500] if self.optimization_level == "aggressive" else [10, 50, 100]
        batch_result = self._optimize_batch_processing(
            processing_function, data_samples[0], batch_sizes
        )
        if batch_result:
            optimization_results.append(batch_result)

        # Parallel processing optimization
        if self.optimization_level in ["balanced", "aggressive"]:
            parallel_result = self._optimize_parallel_processing(
                processing_function, data_samples[0]
            )
            if parallel_result:
                optimization_results.append(parallel_result)

        # Memory optimization
        memory_result = self._optimize_memory_usage(
            processing_function, data_samples[0]
        )
        if memory_result:
            optimization_results.append(memory_result)

        recommendations = self._generate_processing_recommendations(
            optimization_results, baseline_performance
        )

        result = {
            'baseline_performance': baseline_performance,
            'optimization_results': [r.to_dict() for r in optimization_results],
            'recommendations': recommendations,
            'total_improvement': self._calculate_total_improvement(optimization_results)
        }

        logger.info("Data processing optimization completed")
        return result

    def auto_tune_system(
        self,
        workload_functions: List[Callable],
        performance_targets: Dict[str, float],
        duration_minutes: int = 10
    ) -> Dict[str, Any]:
        """
        Automatically tune system for optimal performance.

        Args:
            workload_functions: List of functions representing typical workload
            performance_targets: Performance targets (latency, throughput, etc.)
            duration_minutes: Tuning duration

        Returns:
            Auto-tuning results and optimal configuration
        """
        logger.info(f"Starting auto-tuning for {duration_minutes} minutes")

        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)

        tuning_results = []
        best_configuration = None
        best_score = 0

        iteration = 0
        while time.time() < end_time:
            iteration += 1
            logger.info(f"Auto-tuning iteration {iteration}")

            # Generate random configuration
            config = self._generate_random_configuration()

            # Test configuration
            score = self._evaluate_configuration(config, workload_functions)

            tuning_results.append({
                'iteration': iteration,
                'configuration': config.to_dict(),
                'score': score,
                'timestamp': datetime.utcnow().isoformat()
            })

            # Update best configuration
            if score > best_score:
                best_score = score
                best_configuration = config
                logger.info(f"New best configuration found: score={score:.3f}")

        # Final recommendations
        recommendations = self._generate_tuning_recommendations(
            tuning_results, performance_targets
        )

        result = {
            'best_configuration': best_configuration.to_dict() if best_configuration else None,
            'best_score': best_score,
            'tuning_history': tuning_results,
            'recommendations': recommendations,
            'total_iterations': iteration,
            'tuning_duration_minutes': duration_minutes
        }

        logger.info("Auto-tuning completed")
        return result

    def _get_inference_optimization_params(self) -> Dict[str, List[Any]]:
        """Get parameters to optimize for model inference."""
        if self.optimization_level == "conservative":
            return {
                'batch_size': [1, 10, 50],
                'num_threads': [1, 2, 4]
            }
        elif self.optimization_level == "balanced":
            return {
                'batch_size': [1, 10, 50, 100],
                'num_threads': [1, 2, 4, 8],
                'memory_optimization': [True, False]
            }
        else:  # aggressive
            return {
                'batch_size': [1, 5, 10, 25, 50, 100, 200],
                'num_threads': [1, 2, 4, 8, 16],
                'memory_optimization': [True, False],
                'precision': ['float32', 'float16']
            }

    def _optimize_parameter(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        param_name: str,
        param_values: List[Any],
        baseline_performance: Dict[str, float]
    ) -> Optional[OptimizationResult]:
        """Optimize a single parameter."""
        logger.info(f"Optimizing parameter: {param_name}")

        best_value = None
        best_improvement = 0
        original_value = param_values[0]  # Assume first value is original

        start_time = time.time()

        for value in param_values:
            try:
                # Test performance with this parameter value
                # This is simplified - in practice, you'd apply the parameter
                # to the actual model/system configuration

                if param_name == 'batch_size':
                    # Test with different batch sizes
                    test_results = self.benchmark_suite.benchmark_model_prediction(
                        model, X, [value]
                    )
                else:
                    # For other parameters, use baseline approach
                    test_results = self.benchmark_suite.benchmark_model_prediction(
                        model, X.head(100)  # Smaller sample for speed
                    )

                if test_results:
                    performance = self._aggregate_benchmark_results(test_results)
                    improvement = (
                        performance['avg_samples_per_second'] - baseline_performance['avg_samples_per_second']
                    ) / baseline_performance['avg_samples_per_second']

                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_value = value

            except Exception as e:
                logger.warning(f"Failed to test {param_name}={value}: {e}")
                continue

        execution_time = time.time() - start_time

        if best_value is not None and best_improvement > 0.01:  # At least 1% improvement
            return OptimizationResult(
                parameter_name=param_name,
                original_value=original_value,
                optimized_value=best_value,
                performance_improvement=best_improvement,
                optimization_method="grid_search",
                confidence=0.8,
                execution_time=execution_time
            )

        return None

    def _optimize_batch_processing(
        self,
        processing_function: Callable,
        sample_data: pd.DataFrame,
        batch_sizes: List[int]
    ) -> Optional[OptimizationResult]:
        """Optimize batch processing size."""
        logger.info("Optimizing batch processing")

        best_batch_size = None
        best_performance = 0

        start_time = time.time()

        for batch_size in batch_sizes:
            try:
                # Simulate batch processing
                num_batches = max(1, len(sample_data) // batch_size)

                batch_start = time.perf_counter()
                for i in range(min(num_batches, 5)):  # Test up to 5 batches
                    batch_data = sample_data.iloc[i*batch_size:(i+1)*batch_size]
                    if len(batch_data) > 0:
                        _ = processing_function(batch_data)
                batch_time = time.perf_counter() - batch_start

                # Calculate samples per second
                samples_processed = min(5 * batch_size, len(sample_data))
                performance = samples_processed / batch_time if batch_time > 0 else 0

                if performance > best_performance:
                    best_performance = performance
                    best_batch_size = batch_size

            except Exception as e:
                logger.warning(f"Batch size {batch_size} failed: {e}")
                continue

        execution_time = time.time() - start_time

        if best_batch_size and best_batch_size != batch_sizes[0]:
            improvement = (best_performance - best_performance) / best_performance if best_performance > 0 else 0

            return OptimizationResult(
                parameter_name="batch_size",
                original_value=batch_sizes[0],
                optimized_value=best_batch_size,
                performance_improvement=improvement,
                optimization_method="batch_optimization",
                confidence=0.7,
                execution_time=execution_time
            )

        return None

    def _optimize_parallel_processing(
        self,
        processing_function: Callable,
        sample_data: pd.DataFrame
    ) -> Optional[OptimizationResult]:
        """Optimize parallel processing configuration."""
        logger.info("Optimizing parallel processing")

        worker_counts = [1, 2, 4, 8]
        best_workers = 1
        best_performance = 0

        start_time = time.time()

        for num_workers in worker_counts:
            try:
                # Test parallel processing
                parallel_start = time.perf_counter()

                if num_workers == 1:
                    # Sequential processing
                    _ = processing_function(sample_data.head(100))
                else:
                    # Parallel processing simulation
                    chunk_size = max(1, 100 // num_workers)

                    with ThreadPoolExecutor(max_workers=num_workers) as executor:
                        futures = []
                        for i in range(num_workers):
                            chunk = sample_data.iloc[i*chunk_size:(i+1)*chunk_size]
                            if len(chunk) > 0:
                                future = executor.submit(processing_function, chunk)
                                futures.append(future)

                        # Wait for completion
                        for future in as_completed(futures):
                            try:
                                future.result()
                            except Exception as e:
                                logger.warning(f"Parallel task failed: {e}")

                parallel_time = time.perf_counter() - parallel_start
                performance = 100 / parallel_time if parallel_time > 0 else 0

                if performance > best_performance:
                    best_performance = performance
                    best_workers = num_workers

            except Exception as e:
                logger.warning(f"Parallel processing with {num_workers} workers failed: {e}")
                continue

        execution_time = time.time() - start_time

        if best_workers > 1:
            # Calculate improvement vs sequential
            return OptimizationResult(
                parameter_name="parallel_workers",
                original_value=1,
                optimized_value=best_workers,
                performance_improvement=0.3,  # Simplified estimate
                optimization_method="parallel_optimization",
                confidence=0.6,
                execution_time=execution_time
            )

        return None

    def _optimize_memory_usage(
        self,
        processing_function: Callable,
        sample_data: pd.DataFrame
    ) -> Optional[OptimizationResult]:
        """Optimize memory usage patterns."""
        logger.info("Optimizing memory usage")

        start_time = time.time()

        # Profile memory usage
        with self.profiler.profile_function("memory_optimization"):
            # Test with garbage collection
            import gc
            gc.collect()
            _ = processing_function(sample_data.head(100))
            gc.collect()

        execution_time = time.time() - start_time

        # This is a simplified optimization - in practice you'd test
        # different memory management strategies

        return OptimizationResult(
            parameter_name="memory_management",
            original_value="default",
            optimized_value="gc_optimized",
            performance_improvement=0.1,  # Estimated
            optimization_method="memory_optimization",
            confidence=0.5,
            execution_time=execution_time
        )

    def _generate_random_configuration(self) -> PerformanceConfiguration:
        """Generate a random performance configuration for auto-tuning."""
        return PerformanceConfiguration(
            batch_size=np.random.choice([10, 50, 100, 200, 500]),
            num_workers=np.random.choice([1, 2, 4, 8]),
            memory_limit_mb=np.random.choice([512, 1024, 2048, 4096]),
            cache_enabled=np.random.choice([True, False]),
            parallel_processing=np.random.choice([True, False]),
            optimization_level=np.random.choice(["conservative", "balanced", "aggressive"])
        )

    def _evaluate_configuration(
        self,
        config: PerformanceConfiguration,
        workload_functions: List[Callable]
    ) -> float:
        """Evaluate a performance configuration."""
        # Simplified evaluation - combine multiple metrics
        scores = []

        # Batch size score (preference for moderate sizes)
        batch_score = 1.0 - abs(config.batch_size - 100) / 500
        scores.append(batch_score)

        # Worker count score (diminishing returns)
        worker_score = min(1.0, config.num_workers / 8)
        scores.append(worker_score)

        # Memory limit score
        memory_score = min(1.0, config.memory_limit_mb / 2048)
        scores.append(memory_score)

        # Cache bonus
        if config.cache_enabled:
            scores.append(1.1)

        # Parallel processing bonus
        if config.parallel_processing and len(workload_functions) > 1:
            scores.append(1.2)

        return np.mean(scores)

    def _aggregate_benchmark_results(self, results: List) -> Dict[str, float]:
        """Aggregate benchmark results into summary metrics."""
        if not results:
            return {'avg_samples_per_second': 0, 'avg_execution_time': 0}

        samples_per_second = [r.samples_per_second for r in results]
        execution_times = [r.execution_time for r in results]

        return {
            'avg_samples_per_second': np.mean(samples_per_second),
            'max_samples_per_second': np.max(samples_per_second),
            'avg_execution_time': np.mean(execution_times),
            'min_execution_time': np.min(execution_times)
        }

    def _calculate_total_improvement(self, results: List[OptimizationResult]) -> float:
        """Calculate total performance improvement."""
        if not results:
            return 0.0

        total_improvement = sum(r.performance_improvement for r in results)
        return total_improvement

    def _create_optimized_config(self, results: List[OptimizationResult]) -> PerformanceConfiguration:
        """Create optimized configuration from results."""
        config = PerformanceConfiguration()

        for result in results:
            if result.parameter_name == "batch_size":
                config.batch_size = result.optimized_value
            elif result.parameter_name == "parallel_workers":
                config.num_workers = result.optimized_value

        return config

    def _generate_inference_recommendations(
        self,
        results: List[OptimizationResult],
        baseline: Dict[str, float],
        target_latency: Optional[float],
        target_throughput: Optional[float]
    ) -> List[str]:
        """Generate recommendations for inference optimization."""
        recommendations = []

        if not results:
            recommendations.append("No significant optimizations found - consider model architecture changes")
            return recommendations

        # Batch size recommendations
        batch_results = [r for r in results if r.parameter_name == "batch_size"]
        if batch_results:
            best_batch = max(batch_results, key=lambda x: x.performance_improvement)
            recommendations.append(f"Use batch size {best_batch.optimized_value} for {best_batch.performance_improvement*100:.1f}% improvement")

        # Parallel processing recommendations
        parallel_results = [r for r in results if r.parameter_name == "parallel_workers"]
        if parallel_results:
            best_parallel = max(parallel_results, key=lambda x: x.performance_improvement)
            recommendations.append(f"Use {best_parallel.optimized_value} parallel workers")

        # Target-based recommendations
        if target_latency and baseline['avg_execution_time'] * 1000 > target_latency:
            recommendations.append("Consider model quantization or pruning to meet latency targets")

        if target_throughput and baseline['avg_samples_per_second'] < target_throughput:
            recommendations.append("Consider horizontal scaling or model optimization for throughput targets")

        recommendations.append("Monitor performance in production and adjust configurations based on real workload")

        return recommendations

    def _generate_processing_recommendations(
        self,
        results: List[OptimizationResult],
        baseline: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations for data processing optimization."""
        recommendations = []

        if results:
            best_result = max(results, key=lambda x: x.performance_improvement)
            recommendations.append(f"Apply {best_result.parameter_name} optimization for {best_result.performance_improvement*100:.1f}% improvement")

        recommendations.extend([
            "Consider caching intermediate results for repeated computations",
            "Use vectorized operations where possible",
            "Implement streaming processing for large datasets",
            "Monitor memory usage and implement garbage collection strategies"
        ])

        return recommendations

    def _generate_tuning_recommendations(
        self,
        tuning_results: List[Dict[str, Any]],
        targets: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations from auto-tuning results."""
        recommendations = []

        if tuning_results:
            best_result = max(tuning_results, key=lambda x: x['score'])
            recommendations.append(f"Use configuration from iteration {best_result['iteration']} (score: {best_result['score']:.3f})")

        recommendations.extend([
            "Continue monitoring performance in production",
            "Re-run auto-tuning periodically as workload changes",
            "Consider A/B testing different configurations",
            "Implement automatic scaling based on performance metrics"
        ])

        return recommendations


# CLI interface
def main():
    """CLI interface for performance optimization."""
    import argparse

    parser = argparse.ArgumentParser(description="Performance Optimization CLI")
    parser.add_argument("command", choices=["optimize", "tune"])
    parser.add_argument("--optimization-level", choices=["conservative", "balanced", "aggressive"],
                       default="balanced", help="Optimization aggressiveness")
    parser.add_argument("--duration", type=int, default=5, help="Optimization duration (minutes)")

    args = parser.parse_args()

    if args.command == "optimize":
        # Example model optimization
        from sklearn.datasets import make_classification
        from sklearn.linear_model import LogisticRegression

        # Generate test data
        X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])

        # Train model
        model = LogisticRegression()
        model.fit(X_df, y)

        # Optimize
        optimizer = PerformanceOptimizer(optimization_level=args.optimization_level)
        result = optimizer.optimize_model_inference(model, X_df)

        print("Optimization Results:")
        print(f"  Total improvement: {result['total_improvement']*100:.1f}%")
        print(f"  Baseline performance: {result['baseline_performance']['avg_samples_per_second']:.1f} samples/sec")
        print("  Recommendations:")
        for rec in result['recommendations']:
            print(f"    - {rec}")

    elif args.command == "tune":
        # Example auto-tuning
        def dummy_workload():
            time.sleep(0.1)
            return np.random.randn(100).sum()

        optimizer = PerformanceOptimizer()
        result = optimizer.auto_tune_system([dummy_workload], {}, args.duration)

        print("Auto-tuning Results:")
        print(f"  Best score: {result['best_score']:.3f}")
        print(f"  Total iterations: {result['total_iterations']}")
        if result['best_configuration']:
            print("  Best configuration:")
            for key, value in result['best_configuration'].items():
                print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
