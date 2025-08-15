"""
Performance monitoring and optimization package.

This package provides comprehensive performance monitoring, profiling,
benchmarking, and optimization tools for machine learning systems.

Modules:
    - benchmarks: Performance benchmarking and load testing
    - profiler: Advanced profiling and optimization analysis
    - optimizer: Automated performance optimization and tuning
    - metrics: Real-time metrics collection and monitoring
"""

__version__ = "0.2.0"

from .benchmarks import BenchmarkResult, BenchmarkSuite, LoadTester, LoadTestResult
from .metrics import (
    Alert,
    MetricPoint,
    MetricsCollector,
    MetricSummary,
    MLMetricsCollector,
    SystemMetricsCollector,
)
from .optimizer import (
    OptimizationResult,
    PerformanceConfiguration,
    PerformanceOptimizer,
)
from .profiler import AdvancedProfiler, MemorySnapshot, ProfileResult, ResourceMonitor

__all__ = [
    # Benchmarking
    "BenchmarkSuite",
    "LoadTester",
    "BenchmarkResult",
    "LoadTestResult",

    # Profiling
    "AdvancedProfiler",
    "ResourceMonitor",
    "ProfileResult",
    "MemorySnapshot",

    # Optimization
    "PerformanceOptimizer",
    "PerformanceConfiguration",
    "OptimizationResult",

    # Metrics
    "MetricsCollector",
    "SystemMetricsCollector",
    "MLMetricsCollector",
    "MetricPoint",
    "MetricSummary",
    "Alert"
]
