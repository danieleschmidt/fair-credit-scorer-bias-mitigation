"""
Comprehensive benchmarking and load testing suite.

Provides tools for performance testing, load testing, and benchmarking
of machine learning models and API endpoints.
"""

import asyncio
import concurrent.futures
import gc
import logging
import psutil
import resource
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from ..logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a benchmark test."""
    test_name: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    iterations: int
    samples_per_second: float
    error_rate: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'test_name': self.test_name,
            'execution_time_ms': self.execution_time * 1000,
            'memory_usage_mb': self.memory_usage / 1024 / 1024,
            'cpu_usage_percent': self.cpu_usage,
            'iterations': self.iterations,
            'samples_per_second': self.samples_per_second,
            'error_rate': self.error_rate,
            'metadata': self.metadata
        }


@dataclass
class LoadTestResult:
    """Result of a load test."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_duration: float
    average_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    error_rate: float
    peak_memory_usage: float
    peak_cpu_usage: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'total_duration_seconds': self.total_duration,
            'average_response_time_ms': self.average_response_time * 1000,
            'p50_response_time_ms': self.p50_response_time * 1000,
            'p95_response_time_ms': self.p95_response_time * 1000,
            'p99_response_time_ms': self.p99_response_time * 1000,
            'requests_per_second': self.requests_per_second,
            'error_rate_percent': self.error_rate * 100,
            'peak_memory_usage_mb': self.peak_memory_usage / 1024 / 1024,
            'peak_cpu_usage_percent': self.peak_cpu_usage
        }


class BenchmarkSuite:
    """
    Comprehensive benchmarking suite for ML models and systems.
    
    Provides standardized performance testing for models, data processing,
    and API endpoints with detailed metrics collection.
    """
    
    def __init__(self, warmup_iterations: int = 10, measurement_iterations: int = 100):
        """
        Initialize benchmark suite.
        
        Args:
            warmup_iterations: Number of warmup iterations
            measurement_iterations: Number of measurement iterations
        """
        self.warmup_iterations = warmup_iterations
        self.measurement_iterations = measurement_iterations
        self.results: List[BenchmarkResult] = []
        
        logger.info("BenchmarkSuite initialized")
    
    def benchmark_model_prediction(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        batch_sizes: Optional[List[int]] = None
    ) -> List[BenchmarkResult]:
        """
        Benchmark model prediction performance.
        
        Args:
            model: Trained model to benchmark
            X: Input data for predictions
            batch_sizes: List of batch sizes to test
            
        Returns:
            List of benchmark results
        """
        batch_sizes = batch_sizes or [1, 10, 100, 1000]
        results = []
        
        logger.info("Benchmarking model prediction performance")
        
        for batch_size in batch_sizes:
            if batch_size > len(X):
                continue
            
            # Prepare batch data
            batch_X = X.head(batch_size)
            
            # Warmup
            for _ in range(self.warmup_iterations):
                try:
                    _ = model.predict(batch_X)
                except Exception:
                    break
            
            # Benchmark
            execution_times = []
            memory_usages = []
            cpu_usages = []
            errors = 0
            
            process = psutil.Process()
            
            for i in range(self.measurement_iterations):
                # Measure memory and CPU before
                memory_before = process.memory_info().rss
                cpu_before = process.cpu_percent()
                
                # Time prediction
                start_time = time.perf_counter()
                try:
                    _ = model.predict(batch_X)
                    execution_time = time.perf_counter() - start_time
                    execution_times.append(execution_time)
                except Exception as e:
                    logger.warning(f"Prediction failed in iteration {i}: {e}")
                    errors += 1
                    continue
                
                # Measure memory and CPU after
                memory_after = process.memory_info().rss
                cpu_after = process.cpu_percent()
                
                memory_usages.append(memory_after - memory_before)
                cpu_usages.append(cpu_after - cpu_before)
            
            if execution_times:
                # Calculate statistics
                avg_execution_time = statistics.mean(execution_times)
                avg_memory_usage = statistics.mean(memory_usages) if memory_usages else 0
                avg_cpu_usage = statistics.mean(cpu_usages) if cpu_usages else 0
                samples_per_second = batch_size / avg_execution_time if avg_execution_time > 0 else 0
                error_rate = errors / self.measurement_iterations
                
                result = BenchmarkResult(
                    test_name=f"model_prediction_batch_{batch_size}",
                    execution_time=avg_execution_time,
                    memory_usage=avg_memory_usage,
                    cpu_usage=avg_cpu_usage,
                    iterations=len(execution_times),
                    samples_per_second=samples_per_second,
                    error_rate=error_rate,
                    metadata={
                        'batch_size': batch_size,
                        'model_type': type(model).__name__,
                        'data_shape': X.shape,
                        'execution_time_std': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
                        'min_execution_time': min(execution_times),
                        'max_execution_time': max(execution_times)
                    }
                )
                
                results.append(result)
                self.results.append(result)
                
                logger.info(f"Batch size {batch_size}: {samples_per_second:.1f} samples/sec")
        
        return results
    
    def benchmark_data_processing(
        self,
        processing_function: Callable,
        data_sizes: Optional[List[int]] = None,
        **kwargs
    ) -> List[BenchmarkResult]:
        """
        Benchmark data processing functions.
        
        Args:
            processing_function: Function to benchmark
            data_sizes: List of data sizes to test
            **kwargs: Additional arguments for processing function
            
        Returns:
            List of benchmark results
        """
        data_sizes = data_sizes or [100, 1000, 10000]
        results = []
        
        logger.info("Benchmarking data processing performance")
        
        for data_size in data_sizes:
            # Generate test data
            test_data = self._generate_test_data(data_size)
            
            # Warmup
            for _ in range(min(self.warmup_iterations, 5)):
                try:
                    _ = processing_function(test_data, **kwargs)
                except Exception:
                    break
            
            # Benchmark
            execution_times = []
            memory_usages = []
            errors = 0
            
            process = psutil.Process()
            
            for i in range(self.measurement_iterations):
                # Force garbage collection
                gc.collect()
                
                memory_before = process.memory_info().rss
                
                start_time = time.perf_counter()
                try:
                    result = processing_function(test_data, **kwargs)
                    execution_time = time.perf_counter() - start_time
                    execution_times.append(execution_time)
                    
                    # Clean up result to avoid memory leak
                    del result
                except Exception as e:
                    logger.warning(f"Processing failed in iteration {i}: {e}")
                    errors += 1
                    continue
                
                memory_after = process.memory_info().rss
                memory_usages.append(memory_after - memory_before)
            
            if execution_times:
                avg_execution_time = statistics.mean(execution_times)
                avg_memory_usage = statistics.mean(memory_usages) if memory_usages else 0
                samples_per_second = data_size / avg_execution_time if avg_execution_time > 0 else 0
                error_rate = errors / self.measurement_iterations
                
                result = BenchmarkResult(
                    test_name=f"data_processing_{processing_function.__name__}_size_{data_size}",
                    execution_time=avg_execution_time,
                    memory_usage=avg_memory_usage,
                    cpu_usage=0.0,  # CPU usage not measured for data processing
                    iterations=len(execution_times),
                    samples_per_second=samples_per_second,
                    error_rate=error_rate,
                    metadata={
                        'data_size': data_size,
                        'function_name': processing_function.__name__,
                        'execution_time_std': statistics.stdev(execution_times) if len(execution_times) > 1 else 0
                    }
                )
                
                results.append(result)
                self.results.append(result)
                
                logger.info(f"Data size {data_size}: {samples_per_second:.1f} samples/sec")
        
        return results
    
    def benchmark_memory_usage(
        self,
        test_function: Callable,
        test_name: str,
        *args,
        **kwargs
    ) -> BenchmarkResult:
        """
        Benchmark memory usage of a function.
        
        Args:
            test_function: Function to benchmark
            test_name: Name of the test
            *args: Arguments for test function
            **kwargs: Keyword arguments for test function
            
        Returns:
            Benchmark result
        """
        logger.info(f"Benchmarking memory usage: {test_name}")
        
        process = psutil.Process()
        
        # Baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss
        
        # Peak memory tracking
        peak_memory = baseline_memory
        memory_samples = []
        
        # Execute function and track memory
        start_time = time.perf_counter()
        
        try:
            # Sample memory during execution
            def memory_tracker():
                nonlocal peak_memory
                while True:
                    current_memory = process.memory_info().rss
                    memory_samples.append(current_memory)
                    peak_memory = max(peak_memory, current_memory)
                    time.sleep(0.01)  # Sample every 10ms
            
            # Start memory tracking in background
            import threading
            tracker_thread = threading.Thread(target=memory_tracker, daemon=True)
            tracker_thread.start()
            
            # Execute function
            result = test_function(*args, **kwargs)
            execution_time = time.perf_counter() - start_time
            
            # Stop tracking
            time.sleep(0.1)  # Allow final samples
            
            # Calculate memory usage
            memory_used = peak_memory - baseline_memory
            avg_memory = statistics.mean(memory_samples) - baseline_memory if memory_samples else 0
            
            benchmark_result = BenchmarkResult(
                test_name=f"memory_{test_name}",
                execution_time=execution_time,
                memory_usage=memory_used,
                cpu_usage=0.0,
                iterations=1,
                samples_per_second=0.0,
                metadata={
                    'baseline_memory_mb': baseline_memory / 1024 / 1024,
                    'peak_memory_mb': peak_memory / 1024 / 1024,
                    'average_memory_mb': avg_memory / 1024 / 1024,
                    'memory_samples': len(memory_samples)
                }
            )
            
            self.results.append(benchmark_result)
            
            logger.info(f"Memory usage: {memory_used / 1024 / 1024:.2f} MB")
            return benchmark_result
            
        except Exception as e:
            logger.error(f"Memory benchmark failed: {e}")
            raise
    
    def _generate_test_data(self, size: int) -> pd.DataFrame:
        """Generate test data for benchmarking."""
        np.random.seed(42)  # Reproducible data
        
        data = {
            'feature_1': np.random.normal(0, 1, size),
            'feature_2': np.random.uniform(0, 100, size),
            'feature_3': np.random.choice(['A', 'B', 'C'], size),
            'feature_4': np.random.exponential(2, size)
        }
        
        return pd.DataFrame(data)
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Get summary report of all benchmark results."""
        if not self.results:
            return {"message": "No benchmark results available"}
        
        # Group results by test type
        model_results = [r for r in self.results if 'model_prediction' in r.test_name]
        processing_results = [r for r in self.results if 'data_processing' in r.test_name]
        memory_results = [r for r in self.results if 'memory' in r.test_name]
        
        summary = {
            'total_tests': len(self.results),
            'model_prediction_tests': len(model_results),
            'data_processing_tests': len(processing_results),
            'memory_tests': len(memory_results),
            'performance_summary': {},
            'recommendations': []
        }
        
        # Model prediction summary
        if model_results:
            max_throughput = max(r.samples_per_second for r in model_results)
            avg_throughput = statistics.mean(r.samples_per_second for r in model_results)
            
            summary['performance_summary']['model_prediction'] = {
                'max_throughput_samples_per_sec': max_throughput,
                'average_throughput_samples_per_sec': avg_throughput,
                'fastest_test': max(model_results, key=lambda x: x.samples_per_second).test_name
            }
            
            # Recommendations
            if max_throughput < 100:
                summary['recommendations'].append("Consider model optimization for better throughput")
            if avg_throughput < 50:
                summary['recommendations'].append("Model performance may be insufficient for production")
        
        # Memory usage summary
        if memory_results:
            max_memory = max(r.memory_usage for r in memory_results)
            avg_memory = statistics.mean(r.memory_usage for r in memory_results)
            
            summary['performance_summary']['memory_usage'] = {
                'max_memory_usage_mb': max_memory / 1024 / 1024,
                'average_memory_usage_mb': avg_memory / 1024 / 1024,
                'memory_efficient_test': min(memory_results, key=lambda x: x.memory_usage).test_name
            }
            
            # Memory recommendations
            if max_memory > 1024 * 1024 * 1024:  # > 1GB
                summary['recommendations'].append("High memory usage detected - consider optimization")
        
        return summary


class LoadTester:
    """
    Load testing framework for API endpoints and services.
    
    Provides comprehensive load testing with configurable concurrency,
    request patterns, and detailed performance analytics.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize load tester.
        
        Args:
            base_url: Base URL for load testing
        """
        self.base_url = base_url
        self.results: List[LoadTestResult] = []
        
        logger.info(f"LoadTester initialized for {base_url}")
    
    async def run_load_test(
        self,
        endpoint: str,
        concurrent_users: int = 10,
        duration_seconds: int = 60,
        request_data: Optional[Dict[str, Any]] = None,
        method: str = "GET"
    ) -> LoadTestResult:
        """
        Run load test against an endpoint.
        
        Args:
            endpoint: API endpoint to test
            concurrent_users: Number of concurrent users
            duration_seconds: Test duration in seconds
            request_data: Request payload for POST/PUT requests
            method: HTTP method
            
        Returns:
            Load test results
        """
        import aiohttp
        
        logger.info(f"Starting load test: {concurrent_users} users for {duration_seconds}s")
        
        url = f"{self.base_url}{endpoint}"
        response_times = []
        successful_requests = 0
        failed_requests = 0
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        # Performance monitoring
        process = psutil.Process()
        peak_memory = 0
        peak_cpu = 0
        
        async def make_request(session: aiohttp.ClientSession) -> float:
            """Make a single request and return response time."""
            nonlocal successful_requests, failed_requests, peak_memory, peak_cpu
            
            request_start = time.perf_counter()
            try:
                if method.upper() == "GET":
                    async with session.get(url) as response:
                        await response.text()
                        if response.status < 400:
                            successful_requests += 1
                        else:
                            failed_requests += 1
                elif method.upper() == "POST":
                    async with session.post(url, json=request_data) as response:
                        await response.text()
                        if response.status < 400:
                            successful_requests += 1
                        else:
                            failed_requests += 1
                
                # Monitor system resources
                current_memory = process.memory_info().rss
                current_cpu = process.cpu_percent()
                peak_memory = max(peak_memory, current_memory)
                peak_cpu = max(peak_cpu, current_cpu)
                
                return time.perf_counter() - request_start
                
            except Exception as e:
                failed_requests += 1
                logger.debug(f"Request failed: {e}")
                return time.perf_counter() - request_start
        
        async def user_simulation(user_id: int):
            """Simulate a single user making requests."""
            connector = aiohttp.TCPConnector(limit=100)
            timeout = aiohttp.ClientTimeout(total=30)
            
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                while time.time() < end_time:
                    response_time = await make_request(session)
                    response_times.append(response_time)
                    
                    # Small delay between requests (simulate realistic usage)
                    await asyncio.sleep(0.1)
        
        # Run concurrent users
        tasks = [user_simulation(i) for i in range(concurrent_users)]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Calculate results
        total_duration = time.time() - start_time
        total_requests = successful_requests + failed_requests
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            p50_response_time = statistics.median(response_times)
            
            # Calculate percentiles
            sorted_times = sorted(response_times)
            p95_idx = int(len(sorted_times) * 0.95)
            p99_idx = int(len(sorted_times) * 0.99)
            
            p95_response_time = sorted_times[p95_idx] if p95_idx < len(sorted_times) else sorted_times[-1]
            p99_response_time = sorted_times[p99_idx] if p99_idx < len(sorted_times) else sorted_times[-1]
        else:
            avg_response_time = p50_response_time = p95_response_time = p99_response_time = 0
        
        requests_per_second = total_requests / total_duration if total_duration > 0 else 0
        error_rate = failed_requests / total_requests if total_requests > 0 else 0
        
        result = LoadTestResult(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            total_duration=total_duration,
            average_response_time=avg_response_time,
            p50_response_time=p50_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            requests_per_second=requests_per_second,
            error_rate=error_rate,
            peak_memory_usage=peak_memory,
            peak_cpu_usage=peak_cpu
        )
        
        self.results.append(result)
        
        logger.info(f"Load test completed: {requests_per_second:.1f} req/s, {error_rate*100:.1f}% errors")
        return result
    
    def run_stress_test(
        self,
        endpoint: str,
        max_users: int = 100,
        ramp_up_duration: int = 300,
        request_data: Optional[Dict[str, Any]] = None
    ) -> List[LoadTestResult]:
        """
        Run stress test with gradually increasing load.
        
        Args:
            endpoint: API endpoint to test
            max_users: Maximum number of concurrent users
            ramp_up_duration: Duration to ramp up to max users
            request_data: Request payload
            
        Returns:
            List of load test results at different load levels
        """
        logger.info(f"Starting stress test: ramping up to {max_users} users over {ramp_up_duration}s")
        
        results = []
        user_increments = [5, 10, 20, 50, max_users]
        test_duration = 60  # Test each level for 60 seconds
        
        for num_users in user_increments:
            if num_users > max_users:
                break
            
            logger.info(f"Testing with {num_users} concurrent users")
            
            # Run load test at this level
            result = asyncio.run(self.run_load_test(
                endpoint=endpoint,
                concurrent_users=num_users,
                duration_seconds=test_duration,
                request_data=request_data
            ))
            
            results.append(result)
            
            # Check if system is under stress
            if result.error_rate > 0.1 or result.p95_response_time > 5.0:
                logger.warning(f"System stress detected at {num_users} users")
                break
            
            # Brief pause between tests
            time.sleep(10)
        
        return results


# CLI interface
def main():
    """CLI interface for benchmarking and load testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance Benchmarking CLI")
    parser.add_argument("command", choices=["benchmark", "load-test"])
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 10, 100], help="Batch sizes to test")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL for load testing")
    parser.add_argument("--endpoint", default="/health", help="Endpoint to test")
    parser.add_argument("--users", type=int, default=10, help="Concurrent users")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
    
    args = parser.parse_args()
    
    if args.command == "benchmark":
        # Example model benchmark
        from sklearn.linear_model import LogisticRegression
        from sklearn.datasets import make_classification
        
        # Generate test data
        X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        
        # Train model
        model = LogisticRegression()
        model.fit(X_df, y)
        
        # Run benchmark
        suite = BenchmarkSuite(measurement_iterations=args.iterations)
        results = suite.benchmark_model_prediction(model, X_df, args.batch_sizes)
        
        print("Benchmark Results:")
        for result in results:
            print(f"  {result.test_name}: {result.samples_per_second:.1f} samples/sec")
        
        # Summary
        summary = suite.get_summary_report()
        print(f"\nSummary: {summary['total_tests']} tests completed")
        if summary['recommendations']:
            print("Recommendations:")
            for rec in summary['recommendations']:
                print(f"  - {rec}")
    
    elif args.command == "load-test":
        # Run load test
        tester = LoadTester(args.url)
        
        async def run_test():
            result = await tester.run_load_test(
                endpoint=args.endpoint,
                concurrent_users=args.users,
                duration_seconds=args.duration
            )
            
            print("Load Test Results:")
            print(f"  Total requests: {result.total_requests}")
            print(f"  Requests/second: {result.requests_per_second:.1f}")
            print(f"  Average response time: {result.average_response_time*1000:.1f}ms")
            print(f"  95th percentile: {result.p95_response_time*1000:.1f}ms")
            print(f"  Error rate: {result.error_rate*100:.1f}%")
        
        asyncio.run(run_test())


if __name__ == "__main__":
    main()