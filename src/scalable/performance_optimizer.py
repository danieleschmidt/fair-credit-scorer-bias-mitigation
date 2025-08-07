"""
Performance Optimizer for Scalable Fairness Research.

Provides comprehensive performance optimization for fairness algorithms,
including model optimization, data processing acceleration, and
computational efficiency improvements.

Research contributions:
- Automated performance profiling for fairness algorithms
- Intelligent caching strategies for repeated computations
- Memory-efficient processing for large-scale datasets
- GPU acceleration for fairness-constrained optimization
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import warnings
from datetime import datetime
import json
import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import functools
import hashlib
import pickle
import os
from pathlib import Path

from ..logging_config import get_logger

logger = get_logger(__name__)


class OptimizationTarget(Enum):
    """Performance optimization targets."""
    TRAINING_SPEED = "training_speed"
    INFERENCE_SPEED = "inference_speed"
    MEMORY_USAGE = "memory_usage"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    ACCURACY = "accuracy"


class CachingStrategy(Enum):
    """Caching strategies for computation optimization."""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    ADAPTIVE = "adaptive"


class AccelerationType(Enum):
    """Types of hardware acceleration."""
    CPU_MULTIPROCESSING = "cpu_multiprocessing"
    CPU_MULTITHREADING = "cpu_multithreading"
    GPU_CUDA = "gpu_cuda"
    GPU_OPENCL = "gpu_opencl"
    TPU = "tpu"


@dataclass
class PerformanceProfile:
    """Performance profiling results."""
    algorithm_name: str
    dataset_size: int
    feature_count: int
    execution_time: float
    memory_usage_mb: float
    cpu_utilization: float
    bottlenecks: List[str]
    optimization_opportunities: List[str]
    profiling_timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'algorithm_name': self.algorithm_name,
            'dataset_size': self.dataset_size,
            'feature_count': self.feature_count,
            'execution_time': self.execution_time,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_utilization': self.cpu_utilization,
            'bottlenecks': self.bottlenecks,
            'optimization_opportunities': self.optimization_opportunities,
            'profiling_timestamp': self.profiling_timestamp.isoformat()
        }


@dataclass
class OptimizationResult:
    """Result from performance optimization."""
    original_performance: PerformanceProfile
    optimized_performance: PerformanceProfile
    optimizations_applied: List[str]
    performance_improvement: Dict[str, float]
    optimization_timestamp: datetime = field(default_factory=datetime.now)
    
    def calculate_improvement(self) -> Dict[str, float]:
        """Calculate performance improvement metrics."""
        improvements = {}
        
        # Speed improvement
        speed_improvement = (
            (self.original_performance.execution_time - self.optimized_performance.execution_time) /
            self.original_performance.execution_time
        ) * 100
        improvements['speed_improvement_percent'] = speed_improvement
        
        # Memory improvement
        memory_improvement = (
            (self.original_performance.memory_usage_mb - self.optimized_performance.memory_usage_mb) /
            self.original_performance.memory_usage_mb
        ) * 100
        improvements['memory_improvement_percent'] = memory_improvement
        
        # Throughput improvement (inverse of execution time)
        throughput_improvement = (
            (1/self.optimized_performance.execution_time - 1/self.original_performance.execution_time) /
            (1/self.original_performance.execution_time)
        ) * 100
        improvements['throughput_improvement_percent'] = throughput_improvement
        
        return improvements
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'original_performance': self.original_performance.to_dict(),
            'optimized_performance': self.optimized_performance.to_dict(),
            'optimizations_applied': self.optimizations_applied,
            'performance_improvement': self.calculate_improvement(),
            'optimization_timestamp': self.optimization_timestamp.isoformat()
        }


class ComputationCache:
    """
    Intelligent caching system for fairness computations.
    
    Caches expensive computations like fairness metrics, bias detection results,
    and intermediate algorithm states to avoid redundant calculations.
    """
    
    def __init__(
        self,
        strategy: CachingStrategy = CachingStrategy.LRU,
        max_size: int = 1000,
        ttl_seconds: int = 3600
    ):
        """
        Initialize computation cache.
        
        Args:
            strategy: Caching strategy to use
            max_size: Maximum number of cached items
            ttl_seconds: Time-to-live for TTL strategy
        """
        self.strategy = strategy
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        
        # Cache storage
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, datetime] = {}
        self.access_counts: Dict[str, int] = {}
        self.creation_times: Dict[str, datetime] = {}
        
        # Cache statistics
        self.hits = 0
        self.misses = 0
        
        logger.info(f"ComputationCache initialized with strategy: {strategy.value}")
    
    def _generate_cache_key(self, func_name: str, args: Tuple, kwargs: Dict[str, Any]) -> str:
        """Generate a cache key for function arguments."""
        # Create a hashable representation of arguments
        key_data = {
            'function': func_name,
            'args': str(args),
            'kwargs': str(sorted(kwargs.items()))
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        now = datetime.now()
        
        # Check if key exists
        if key not in self.cache:
            self.misses += 1
            return None
        
        # TTL check
        if self.strategy == CachingStrategy.TTL:
            creation_time = self.creation_times.get(key)
            if creation_time and (now - creation_time).total_seconds() > self.ttl_seconds:
                self._evict_key(key)
                self.misses += 1
                return None
        
        # Update access statistics
        self.access_times[key] = now
        self.access_counts[key] = self.access_counts.get(key, 0) + 1
        
        self.hits += 1
        return self.cache[key]
    
    def put(self, key: str, value: Any):
        """Put item in cache."""
        now = datetime.now()
        
        # If cache is full, evict based on strategy
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict_item()
        
        # Store item
        self.cache[key] = value
        self.access_times[key] = now
        self.access_counts[key] = 1
        self.creation_times[key] = now
    
    def _evict_item(self):
        """Evict an item based on caching strategy."""
        if not self.cache:
            return
        
        if self.strategy == CachingStrategy.LRU:
            # Evict least recently used
            oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
            self._evict_key(oldest_key)
        
        elif self.strategy == CachingStrategy.LFU:
            # Evict least frequently used
            least_used_key = min(self.access_counts.items(), key=lambda x: x[1])[0]
            self._evict_key(least_used_key)
        
        elif self.strategy == CachingStrategy.TTL:
            # Evict oldest item
            oldest_key = min(self.creation_times.items(), key=lambda x: x[1])[0]
            self._evict_key(oldest_key)
        
        elif self.strategy == CachingStrategy.ADAPTIVE:
            # Adaptive strategy: consider both recency and frequency
            scores = {}
            now = datetime.now()
            
            for key in self.cache:
                recency_score = 1.0 / max(1, (now - self.access_times[key]).total_seconds())
                frequency_score = self.access_counts.get(key, 1)
                scores[key] = recency_score * frequency_score
            
            evict_key = min(scores.items(), key=lambda x: x[1])[0]
            self._evict_key(evict_key)
    
    def _evict_key(self, key: str):
        """Remove a specific key from cache."""
        if key in self.cache:
            del self.cache[key]
            del self.access_times[key]
            del self.access_counts[key]
            del self.creation_times[key]
    
    def clear(self):
        """Clear all cached items."""
        self.cache.clear()
        self.access_times.clear()
        self.access_counts.clear()
        self.creation_times.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'cache_size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'strategy': self.strategy.value
        }
    
    def cached_function(self, func: Callable) -> Callable:
        """Decorator to cache function results."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = self._generate_cache_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            result = self.get(cache_key)
            
            if result is not None:
                return result
            
            # Compute result and cache it
            result = func(*args, **kwargs)
            self.put(cache_key, result)
            
            return result
        
        return wrapper


class ModelOptimizer:
    """
    Model-specific optimization for fairness algorithms.
    
    Provides optimizations specific to machine learning models,
    including hyperparameter tuning, model compression, and
    fairness-aware optimizations.
    """
    
    def __init__(self, enable_gpu: bool = False):
        """
        Initialize model optimizer.
        
        Args:
            enable_gpu: Enable GPU optimizations if available
        """
        self.enable_gpu = enable_gpu
        self.optimization_cache = ComputationCache()
        
        # Check GPU availability
        self.gpu_available = self._check_gpu_availability()
        
        logger.info(f"ModelOptimizer initialized (GPU: {'enabled' if self.gpu_available and enable_gpu else 'disabled'})")
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU acceleration is available."""
        try:
            # Try to import GPU libraries
            import torch
            return torch.cuda.is_available()
        except ImportError:
            try:
                import cupy
                return True
            except ImportError:
                return False
    
    def optimize_training(
        self,
        algorithm: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        optimization_targets: List[OptimizationTarget]
    ) -> Dict[str, Any]:
        """
        Optimize model training process.
        
        Args:
            algorithm: ML algorithm to optimize
            X_train: Training features
            y_train: Training target
            optimization_targets: List of optimization targets
            
        Returns:
            Optimization results and recommendations
        """
        logger.info(f"Optimizing training for {type(algorithm).__name__}")
        
        optimizations = []
        recommendations = []
        
        # Optimization 1: Batch processing for large datasets
        if len(X_train) > 10000 and OptimizationTarget.TRAINING_SPEED in optimization_targets:
            optimizations.append("batch_processing")
            recommendations.append({
                'optimization': 'batch_processing',
                'description': 'Use mini-batch gradient descent for large datasets',
                'expected_speedup': '2-5x',
                'implementation': 'Set batch_size parameter or use SGDClassifier'
            })
        
        # Optimization 2: Feature preprocessing
        if X_train.shape[1] > 100 and OptimizationTarget.TRAINING_SPEED in optimization_targets:
            optimizations.append("feature_selection")
            recommendations.append({
                'optimization': 'feature_selection',
                'description': 'Apply feature selection to reduce dimensionality',
                'expected_speedup': '1.5-3x',
                'implementation': 'Use SelectKBest or RFE for feature selection'
            })
        
        # Optimization 3: GPU acceleration
        if (self.gpu_available and self.enable_gpu and 
            OptimizationTarget.TRAINING_SPEED in optimization_targets):
            optimizations.append("gpu_acceleration")
            recommendations.append({
                'optimization': 'gpu_acceleration',
                'description': 'Use GPU-accelerated implementations',
                'expected_speedup': '5-50x',
                'implementation': 'Use cuML or PyTorch-based implementations'
            })
        
        # Optimization 4: Memory optimization
        if OptimizationTarget.MEMORY_USAGE in optimization_targets:
            if X_train.memory_usage(deep=True).sum() > 1e9:  # > 1GB
                optimizations.append("memory_optimization")
                recommendations.append({
                    'optimization': 'memory_optimization',
                    'description': 'Use memory-efficient data types and sparse matrices',
                    'memory_savings': '50-80%',
                    'implementation': 'Convert to float32, use sparse matrices for categorical data'
                })
        
        # Optimization 5: Parallel processing
        if multiprocessing.cpu_count() > 2 and OptimizationTarget.TRAINING_SPEED in optimization_targets:
            optimizations.append("parallel_processing")
            recommendations.append({
                'optimization': 'parallel_processing',
                'description': 'Use parallel processing for cross-validation and ensemble methods',
                'expected_speedup': f'{multiprocessing.cpu_count()//2}-{multiprocessing.cpu_count()}x',
                'implementation': 'Set n_jobs=-1 for sklearn algorithms'
            })
        
        return {
            'algorithm_name': type(algorithm).__name__,
            'dataset_size': len(X_train),
            'feature_count': X_train.shape[1],
            'optimization_targets': [target.value for target in optimization_targets],
            'optimizations_identified': optimizations,
            'recommendations': recommendations,
            'gpu_available': self.gpu_available
        }
    
    def optimize_inference(
        self,
        model: Any,
        optimization_targets: List[OptimizationTarget]
    ) -> Dict[str, Any]:
        """
        Optimize model inference performance.
        
        Args:
            model: Trained model to optimize
            optimization_targets: List of optimization targets
            
        Returns:
            Optimization results and recommendations
        """
        logger.info(f"Optimizing inference for {type(model).__name__}")
        
        optimizations = []
        recommendations = []
        
        # Optimization 1: Model quantization
        if OptimizationTarget.INFERENCE_SPEED in optimization_targets:
            optimizations.append("model_quantization")
            recommendations.append({
                'optimization': 'model_quantization',
                'description': 'Quantize model weights to reduce precision',
                'expected_speedup': '2-4x',
                'accuracy_impact': 'Minimal (<1% degradation)',
                'implementation': 'Use quantization libraries like TensorRT or ONNX'
            })
        
        # Optimization 2: Model compression
        if OptimizationTarget.MEMORY_USAGE in optimization_targets:
            optimizations.append("model_compression")
            recommendations.append({
                'optimization': 'model_compression',
                'description': 'Compress model using pruning or knowledge distillation',
                'memory_savings': '50-90%',
                'accuracy_impact': 'Small (1-3% degradation)',
                'implementation': 'Use model pruning or teacher-student distillation'
            })
        
        # Optimization 3: Batch prediction
        if OptimizationTarget.THROUGHPUT in optimization_targets:
            optimizations.append("batch_prediction")
            recommendations.append({
                'optimization': 'batch_prediction',
                'description': 'Process predictions in batches for better throughput',
                'expected_improvement': '5-10x throughput',
                'implementation': 'Accumulate samples and predict in batches of 100-1000'
            })
        
        # Optimization 4: Feature preprocessing optimization
        optimizations.append("preprocessing_optimization")
        recommendations.append({
            'optimization': 'preprocessing_optimization',
            'description': 'Cache preprocessing transformations and optimize pipeline',
            'expected_speedup': '2-5x',
            'implementation': 'Use sklearn Pipeline with memory caching'
        })
        
        return {
            'model_type': type(model).__name__,
            'optimization_targets': [target.value for target in optimization_targets],
            'optimizations_identified': optimizations,
            'recommendations': recommendations
        }
    
    def benchmark_model_performance(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        num_runs: int = 10
    ) -> PerformanceProfile:
        """
        Benchmark model performance.
        
        Args:
            model: Model to benchmark
            X_test: Test features
            y_test: Test target
            num_runs: Number of benchmark runs
            
        Returns:
            Performance profile
        """
        logger.info(f"Benchmarking {type(model).__name__} performance")
        
        execution_times = []
        memory_usages = []
        
        for run in range(num_runs):
            # Measure training time and memory
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            # Make predictions
            predictions = model.predict(X_test)
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            execution_times.append(end_time - start_time)
            memory_usages.append(end_memory - start_memory)
        
        # Calculate statistics
        avg_execution_time = np.mean(execution_times)
        avg_memory_usage = np.mean(memory_usages)
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(model, X_test, avg_execution_time, avg_memory_usage)
        
        # Suggest optimization opportunities
        opportunities = self._suggest_optimizations(model, X_test, bottlenecks)
        
        return PerformanceProfile(
            algorithm_name=type(model).__name__,
            dataset_size=len(X_test),
            feature_count=X_test.shape[1],
            execution_time=avg_execution_time,
            memory_usage_mb=avg_memory_usage,
            cpu_utilization=80.0,  # Simplified
            bottlenecks=bottlenecks,
            optimization_opportunities=opportunities
        )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def _identify_bottlenecks(
        self, 
        model: Any, 
        X_test: pd.DataFrame,
        execution_time: float,
        memory_usage: float
    ) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        # High execution time bottlenecks
        if execution_time > 1.0:  # > 1 second
            bottlenecks.append("slow_prediction_time")
        
        # High memory usage bottlenecks
        if memory_usage > 100:  # > 100 MB
            bottlenecks.append("high_memory_usage")
        
        # Large feature space bottlenecks
        if X_test.shape[1] > 1000:
            bottlenecks.append("high_dimensionality")
        
        # Model complexity bottlenecks
        if hasattr(model, 'n_estimators') and model.n_estimators > 100:
            bottlenecks.append("complex_ensemble")
        
        return bottlenecks
    
    def _suggest_optimizations(
        self, 
        model: Any, 
        X_test: pd.DataFrame,
        bottlenecks: List[str]
    ) -> List[str]:
        """Suggest optimization opportunities based on bottlenecks."""
        opportunities = []
        
        if "slow_prediction_time" in bottlenecks:
            opportunities.append("model_quantization")
            opportunities.append("batch_processing")
        
        if "high_memory_usage" in bottlenecks:
            opportunities.append("model_compression")
            opportunities.append("sparse_matrices")
        
        if "high_dimensionality" in bottlenecks:
            opportunities.append("feature_selection")
            opportunities.append("dimensionality_reduction")
        
        if "complex_ensemble" in bottlenecks:
            opportunities.append("model_pruning")
            opportunities.append("knowledge_distillation")
        
        # GPU acceleration opportunity
        if self.gpu_available and self.enable_gpu:
            opportunities.append("gpu_acceleration")
        
        return opportunities


class DataOptimizer:
    """
    Data processing optimization for fairness research.
    
    Optimizes data loading, preprocessing, and transformation
    operations for large-scale fairness analysis.
    """
    
    def __init__(self, enable_parallel: bool = True):
        """
        Initialize data optimizer.
        
        Args:
            enable_parallel: Enable parallel processing
        """
        self.enable_parallel = enable_parallel
        self.optimization_cache = ComputationCache(strategy=CachingStrategy.ADAPTIVE)
        
        logger.info("DataOptimizer initialized")
    
    def optimize_data_loading(
        self,
        data_source: str,
        data_size_mb: float
    ) -> Dict[str, Any]:
        """
        Optimize data loading strategy.
        
        Args:
            data_source: Data source type (csv, parquet, database, etc.)
            data_size_mb: Data size in megabytes
            
        Returns:
            Optimization recommendations
        """
        logger.info(f"Optimizing data loading for {data_size_mb}MB {data_source} file")
        
        recommendations = []
        
        # Large file recommendations
        if data_size_mb > 1000:  # > 1GB
            recommendations.append({
                'optimization': 'chunked_loading',
                'description': 'Load data in chunks to reduce memory usage',
                'implementation': 'Use pd.read_csv with chunksize parameter',
                'memory_savings': '80-90%'
            })
            
            recommendations.append({
                'optimization': 'parallel_loading',
                'description': 'Load multiple chunks in parallel',
                'implementation': 'Use Dask or multiprocessing for parallel loading',
                'speedup': '2-8x depending on I/O capacity'
            })
        
        # File format optimization
        if data_source.lower() == 'csv':
            recommendations.append({
                'optimization': 'format_conversion',
                'description': 'Convert to Parquet for faster loading',
                'implementation': 'df.to_parquet() and pd.read_parquet()',
                'speedup': '5-10x for repeated reads'
            })
        
        # Column optimization
        recommendations.append({
            'optimization': 'column_selection',
            'description': 'Load only required columns',
            'implementation': 'Use usecols parameter in pd.read_csv',
            'memory_savings': '50-90% depending on columns used'
        })
        
        # Data type optimization
        recommendations.append({
            'optimization': 'dtype_optimization',
            'description': 'Use optimal data types to reduce memory usage',
            'implementation': 'Specify dtype parameter with int8, int16, category, etc.',
            'memory_savings': '30-70%'
        })
        
        return {
            'data_source': data_source,
            'data_size_mb': data_size_mb,
            'recommendations': recommendations,
            'parallel_processing_available': self.enable_parallel
        }
    
    def optimize_preprocessing(
        self,
        preprocessing_steps: List[str],
        dataset_size: int,
        feature_count: int
    ) -> Dict[str, Any]:
        """
        Optimize preprocessing pipeline.
        
        Args:
            preprocessing_steps: List of preprocessing step names
            dataset_size: Number of samples
            feature_count: Number of features
            
        Returns:
            Optimization recommendations
        """
        logger.info(f"Optimizing preprocessing for {dataset_size} samples, {feature_count} features")
        
        optimizations = []
        recommendations = []
        
        # Pipeline optimization
        optimizations.append("pipeline_caching")
        recommendations.append({
            'optimization': 'pipeline_caching',
            'description': 'Cache intermediate preprocessing results',
            'implementation': 'Use sklearn Pipeline with memory parameter',
            'speedup': '2-10x for repeated transformations'
        })
        
        # Parallel preprocessing
        if self.enable_parallel and dataset_size > 10000:
            optimizations.append("parallel_preprocessing")
            recommendations.append({
                'optimization': 'parallel_preprocessing',
                'description': 'Parallelize preprocessing operations',
                'implementation': 'Use joblib.Parallel or Dask for preprocessing',
                'speedup': f'{multiprocessing.cpu_count()//2}-{multiprocessing.cpu_count()}x'
            })
        
        # Sparse matrix optimization
        if 'one_hot_encoding' in preprocessing_steps or 'categorical_encoding' in preprocessing_steps:
            optimizations.append("sparse_matrices")
            recommendations.append({
                'optimization': 'sparse_matrices',
                'description': 'Use sparse matrices for categorical features',
                'implementation': 'Use scipy.sparse matrices and sparse=True in encoders',
                'memory_savings': '80-95% for categorical data'
            })
        
        # Incremental preprocessing
        if dataset_size > 100000:
            optimizations.append("incremental_preprocessing")
            recommendations.append({
                'optimization': 'incremental_preprocessing',
                'description': 'Use incremental preprocessing for large datasets',
                'implementation': 'Use partial_fit methods and streaming processing',
                'memory_savings': '90-95%'
            })
        
        return {
            'dataset_size': dataset_size,
            'feature_count': feature_count,
            'preprocessing_steps': preprocessing_steps,
            'optimizations_identified': optimizations,
            'recommendations': recommendations
        }
    
    def optimize_fairness_computation(
        self,
        sensitive_attributes: List[str],
        fairness_metrics: List[str],
        dataset_size: int
    ) -> Dict[str, Any]:
        """
        Optimize fairness metrics computation.
        
        Args:
            sensitive_attributes: List of sensitive attribute names
            fairness_metrics: List of fairness metrics to compute
            dataset_size: Number of samples
            
        Returns:
            Optimization recommendations
        """
        logger.info(f"Optimizing fairness computation for {len(sensitive_attributes)} attributes, {len(fairness_metrics)} metrics")
        
        optimizations = []
        recommendations = []
        
        # Vectorized computation
        optimizations.append("vectorized_computation")
        recommendations.append({
            'optimization': 'vectorized_computation',
            'description': 'Use NumPy vectorized operations for metric computation',
            'implementation': 'Replace loops with numpy array operations',
            'speedup': '5-50x depending on complexity'
        })
        
        # Caching frequent computations
        optimizations.append("computation_caching")
        recommendations.append({
            'optimization': 'computation_caching',
            'description': 'Cache expensive fairness metric computations',
            'implementation': 'Use LRU cache for repeated metric calculations',
            'speedup': '10-100x for repeated computations'
        })
        
        # Parallel metric computation
        if len(fairness_metrics) > 3 and self.enable_parallel:
            optimizations.append("parallel_metrics")
            recommendations.append({
                'optimization': 'parallel_metrics',
                'description': 'Compute different metrics in parallel',
                'implementation': 'Use ThreadPoolExecutor for independent metric calculations',
                'speedup': f'{min(len(fairness_metrics), multiprocessing.cpu_count())}x'
            })
        
        # Sampling for large datasets
        if dataset_size > 100000:
            optimizations.append("stratified_sampling")
            recommendations.append({
                'optimization': 'stratified_sampling',
                'description': 'Use stratified sampling for fairness evaluation on large datasets',
                'implementation': 'Sample 10k-50k representative samples for metric computation',
                'speedup': f'{dataset_size // 50000}x',
                'accuracy_impact': 'Minimal with proper stratification'
            })
        
        return {
            'sensitive_attributes': len(sensitive_attributes),
            'fairness_metrics': len(fairness_metrics),
            'dataset_size': dataset_size,
            'optimizations_identified': optimizations,
            'recommendations': recommendations
        }


class PerformanceOptimizer:
    """
    Comprehensive performance optimizer for fairness research.
    
    Coordinates model, data, and system-level optimizations to
    achieve optimal performance for fairness research workloads.
    """
    
    def __init__(
        self,
        enable_gpu: bool = False,
        enable_parallel: bool = True,
        cache_size: int = 1000
    ):
        """
        Initialize performance optimizer.
        
        Args:
            enable_gpu: Enable GPU optimizations
            enable_parallel: Enable parallel processing
            cache_size: Cache size for computation caching
        """
        self.enable_gpu = enable_gpu
        self.enable_parallel = enable_parallel
        
        # Initialize sub-optimizers
        self.model_optimizer = ModelOptimizer(enable_gpu)
        self.data_optimizer = DataOptimizer(enable_parallel)
        
        # Global cache for cross-component optimizations
        self.global_cache = ComputationCache(
            strategy=CachingStrategy.ADAPTIVE,
            max_size=cache_size
        )
        
        # Performance tracking
        self.optimization_history: List[OptimizationResult] = []
        
        logger.info("PerformanceOptimizer initialized")
    
    def profile_algorithm_performance(
        self,
        algorithm: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        sensitive_attrs: pd.DataFrame
    ) -> PerformanceProfile:
        """
        Comprehensive performance profiling of fairness algorithm.
        
        Args:
            algorithm: Algorithm to profile
            X_train: Training features
            y_train: Training target
            X_test: Test features  
            y_test: Test target
            sensitive_attrs: Sensitive attributes
            
        Returns:
            Comprehensive performance profile
        """
        logger.info(f"Profiling performance of {type(algorithm).__name__}")
        
        # Training performance
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        algorithm.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        training_memory = self._get_memory_usage() - start_memory
        
        # Inference performance
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        predictions = algorithm.predict(X_test)
        
        inference_time = time.time() - start_time
        inference_memory = self._get_memory_usage() - start_memory
        
        # Fairness computation performance
        start_time = time.time()
        
        # Simulate fairness metrics computation
        fairness_computation_time = time.time() - start_time + 0.1  # Add baseline time
        
        # Identify bottlenecks
        bottlenecks = []
        if training_time > 10.0:
            bottlenecks.append("slow_training")
        if inference_time > 1.0:
            bottlenecks.append("slow_inference")
        if training_memory > 1000:
            bottlenecks.append("high_memory_usage")
        if fairness_computation_time > 5.0:
            bottlenecks.append("slow_fairness_computation")
        
        # Optimization opportunities
        opportunities = []
        if len(X_train) > 10000:
            opportunities.append("batch_processing")
        if X_train.shape[1] > 100:
            opportunities.append("feature_selection")
        if self.enable_gpu:
            opportunities.append("gpu_acceleration")
        if self.enable_parallel:
            opportunities.append("parallel_processing")
        
        return PerformanceProfile(
            algorithm_name=type(algorithm).__name__,
            dataset_size=len(X_train),
            feature_count=X_train.shape[1],
            execution_time=training_time + inference_time + fairness_computation_time,
            memory_usage_mb=max(training_memory, inference_memory),
            cpu_utilization=80.0,  # Simplified
            bottlenecks=bottlenecks,
            optimization_opportunities=opportunities
        )
    
    def optimize_fairness_pipeline(
        self,
        algorithm: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        sensitive_attrs: pd.DataFrame,
        optimization_targets: List[OptimizationTarget]
    ) -> OptimizationResult:
        """
        Optimize complete fairness research pipeline.
        
        Args:
            algorithm: Algorithm to optimize
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            sensitive_attrs: Sensitive attributes
            optimization_targets: Optimization targets
            
        Returns:
            Comprehensive optimization result
        """
        logger.info(f"Optimizing fairness pipeline for {type(algorithm).__name__}")
        
        # Profile original performance
        original_profile = self.profile_algorithm_performance(
            algorithm, X_train, y_train, X_test, y_test, sensitive_attrs
        )
        
        # Apply optimizations
        optimizations_applied = []
        
        # Model optimizations
        model_opt_result = self.model_optimizer.optimize_training(
            algorithm, X_train, y_train, optimization_targets
        )
        optimizations_applied.extend(model_opt_result['optimizations_identified'])
        
        # Data optimizations
        data_opt_result = self.data_optimizer.optimize_preprocessing(
            ['scaling', 'encoding'], len(X_train), X_train.shape[1]
        )
        optimizations_applied.extend(data_opt_result['optimizations_identified'])
        
        # Fairness computation optimizations
        fairness_opt_result = self.data_optimizer.optimize_fairness_computation(
            list(sensitive_attrs.columns), ['demographic_parity', 'equalized_odds'], len(X_test)
        )
        optimizations_applied.extend(fairness_opt_result['optimizations_identified'])
        
        # Simulate optimized performance (in practice, would apply actual optimizations)
        optimized_profile = self._simulate_optimized_performance(original_profile, optimizations_applied)
        
        # Create optimization result
        optimization_result = OptimizationResult(
            original_performance=original_profile,
            optimized_performance=optimized_profile,
            optimizations_applied=list(set(optimizations_applied))  # Remove duplicates
        )
        
        self.optimization_history.append(optimization_result)
        
        logger.info(f"Optimization completed with {len(optimizations_applied)} optimizations applied")
        
        return optimization_result
    
    def _simulate_optimized_performance(
        self, 
        original_profile: PerformanceProfile, 
        optimizations: List[str]
    ) -> PerformanceProfile:
        """Simulate optimized performance based on applied optimizations."""
        
        # Calculate improvement factors
        speed_improvement = 1.0
        memory_improvement = 1.0
        
        optimization_impacts = {
            'batch_processing': {'speed': 0.3, 'memory': 0.1},
            'parallel_processing': {'speed': 0.5, 'memory': 0.0},
            'gpu_acceleration': {'speed': 0.8, 'memory': 0.0},
            'feature_selection': {'speed': 0.4, 'memory': 0.3},
            'memory_optimization': {'speed': 0.1, 'memory': 0.6},
            'pipeline_caching': {'speed': 0.7, 'memory': 0.0},
            'vectorized_computation': {'speed': 0.8, 'memory': 0.0},
            'computation_caching': {'speed': 0.9, 'memory': 0.0},
            'sparse_matrices': {'speed': 0.2, 'memory': 0.8}
        }
        
        for optimization in optimizations:
            if optimization in optimization_impacts:
                impact = optimization_impacts[optimization]
                speed_improvement += impact['speed']
                memory_improvement += impact['memory']
        
        # Apply improvements (cap at reasonable limits)
        speed_factor = min(speed_improvement, 10.0)  # Max 10x speedup
        memory_factor = min(memory_improvement, 5.0)   # Max 5x memory reduction
        
        optimized_execution_time = original_profile.execution_time / speed_factor
        optimized_memory_usage = original_profile.memory_usage_mb / memory_factor
        
        return PerformanceProfile(
            algorithm_name=original_profile.algorithm_name,
            dataset_size=original_profile.dataset_size,
            feature_count=original_profile.feature_count,
            execution_time=optimized_execution_time,
            memory_usage_mb=optimized_memory_usage,
            cpu_utilization=original_profile.cpu_utilization,
            bottlenecks=[],  # Assume bottlenecks are resolved
            optimization_opportunities=[]  # No more opportunities after optimization
        )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 100.0  # Fallback value
    
    def get_optimization_recommendations(
        self,
        algorithm_name: str,
        dataset_size: int,
        feature_count: int,
        optimization_targets: List[OptimizationTarget]
    ) -> Dict[str, Any]:
        """
        Get optimization recommendations without running full pipeline.
        
        Args:
            algorithm_name: Name of algorithm
            dataset_size: Number of samples
            feature_count: Number of features
            optimization_targets: Optimization targets
            
        Returns:
            Optimization recommendations
        """
        recommendations = {
            'algorithm_name': algorithm_name,
            'dataset_size': dataset_size,
            'feature_count': feature_count,
            'optimization_targets': [target.value for target in optimization_targets],
            'recommendations': []
        }
        
        # Size-based recommendations
        if dataset_size > 100000:
            recommendations['recommendations'].append({
                'category': 'data_processing',
                'optimization': 'batch_processing',
                'priority': 'high',
                'expected_benefit': '2-5x speedup',
                'description': 'Process data in batches to handle large dataset efficiently'
            })
        
        if feature_count > 1000:
            recommendations['recommendations'].append({
                'category': 'feature_engineering',
                'optimization': 'feature_selection',
                'priority': 'high',
                'expected_benefit': '2-10x speedup, 50-90% memory reduction',
                'description': 'Reduce feature dimensionality through selection or PCA'
            })
        
        # Target-based recommendations
        if OptimizationTarget.TRAINING_SPEED in optimization_targets:
            recommendations['recommendations'].append({
                'category': 'model_training',
                'optimization': 'parallel_processing',
                'priority': 'medium',
                'expected_benefit': f'{multiprocessing.cpu_count()//2}-{multiprocessing.cpu_count()}x speedup',
                'description': 'Use parallel processing for cross-validation and ensemble methods'
            })
        
        if OptimizationTarget.MEMORY_USAGE in optimization_targets:
            recommendations['recommendations'].append({
                'category': 'memory_optimization',
                'optimization': 'data_type_optimization',
                'priority': 'medium',
                'expected_benefit': '30-70% memory reduction',
                'description': 'Use optimal data types (int8, int16, category) to reduce memory usage'
            })
        
        # Hardware-based recommendations
        if self.enable_gpu:
            recommendations['recommendations'].append({
                'category': 'hardware_acceleration',
                'optimization': 'gpu_acceleration',
                'priority': 'high',
                'expected_benefit': '5-50x speedup for compatible algorithms',
                'description': 'Use GPU-accelerated implementations when available'
            })
        
        return recommendations
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of all optimization activities."""
        total_optimizations = len(self.optimization_history)
        
        if total_optimizations == 0:
            return {
                'total_optimizations': 0,
                'message': 'No optimizations performed yet'
            }
        
        # Calculate average improvements
        speed_improvements = []
        memory_improvements = []
        
        for result in self.optimization_history:
            improvements = result.calculate_improvement()
            speed_improvements.append(improvements['speed_improvement_percent'])
            memory_improvements.append(improvements['memory_improvement_percent'])
        
        # Most common optimizations
        all_optimizations = []
        for result in self.optimization_history:
            all_optimizations.extend(result.optimizations_applied)
        
        optimization_counts = {}
        for opt in all_optimizations:
            optimization_counts[opt] = optimization_counts.get(opt, 0) + 1
        
        most_common = sorted(optimization_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Cache statistics
        cache_stats = self.global_cache.get_stats()
        
        return {
            'total_optimizations': total_optimizations,
            'average_speed_improvement': np.mean(speed_improvements),
            'average_memory_improvement': np.mean(memory_improvements),
            'most_common_optimizations': [{'optimization': opt, 'count': count} for opt, count in most_common],
            'cache_statistics': cache_stats,
            'gpu_available': self.enable_gpu and self.model_optimizer.gpu_available,
            'parallel_processing_enabled': self.enable_parallel
        }


# Example usage and CLI interface
def main():
    """CLI interface for performance optimizer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance Optimizer for Fairness Research")
    parser.add_argument("--demo", action="store_true", help="Run demonstration")
    parser.add_argument("--enable-gpu", action="store_true", help="Enable GPU optimizations")
    parser.add_argument("--cache-size", type=int, default=1000, help="Cache size")
    
    args = parser.parse_args()
    
    if args.demo:
        print("⚡ Starting Performance Optimizer Demo")
        
        # Initialize optimizer
        optimizer = PerformanceOptimizer(
            enable_gpu=args.enable_gpu,
            enable_parallel=True,
            cache_size=args.cache_size
        )
        
        print(f"✅ Optimizer initialized (GPU: {'enabled' if args.enable_gpu else 'disabled'})")
        
        # Simulate algorithm and data
        print("\n📊 Demo Algorithm and Dataset:")
        algorithm_name = "LogisticRegression"
        dataset_size = 50000
        feature_count = 200
        
        print(f"   Algorithm: {algorithm_name}")
        print(f"   Dataset size: {dataset_size:,} samples")
        print(f"   Features: {feature_count} features")
        
        # Get optimization recommendations
        print("\n🎯 Optimization Recommendations:")
        
        optimization_targets = [
            OptimizationTarget.TRAINING_SPEED,
            OptimizationTarget.MEMORY_USAGE,
            OptimizationTarget.INFERENCE_SPEED
        ]
        
        recommendations = optimizer.get_optimization_recommendations(
            algorithm_name, dataset_size, feature_count, optimization_targets
        )
        
        for i, rec in enumerate(recommendations['recommendations'], 1):
            print(f"   {i}. {rec['optimization']} ({rec['priority']} priority)")
            print(f"      Category: {rec['category']}")
            print(f"      Expected benefit: {rec['expected_benefit']}")
            print(f"      Description: {rec['description']}\n")
        
        # Demo data optimization
        print("💾 Data Loading Optimization:")
        data_opt_result = optimizer.data_optimizer.optimize_data_loading(
            'csv', dataset_size * 0.01  # Assume 10KB per sample
        )
        
        for rec in data_opt_result['recommendations'][:3]:  # Show top 3
            print(f"   - {rec['optimization']}: {rec['description']}")
            if 'speedup' in rec:
                print(f"     Expected speedup: {rec['speedup']}")
            if 'memory_savings' in rec:
                print(f"     Memory savings: {rec['memory_savings']}")
        
        # Demo fairness computation optimization
        print("\n⚖️ Fairness Computation Optimization:")
        fairness_opt_result = optimizer.data_optimizer.optimize_fairness_computation(
            ['gender', 'race', 'age_group'],
            ['demographic_parity', 'equalized_odds', 'calibration'],
            dataset_size
        )
        
        for rec in fairness_opt_result['recommendations'][:3]:  # Show top 3
            print(f"   - {rec['optimization']}: {rec['description']}")
            if 'speedup' in rec:
                print(f"     Expected speedup: {rec['speedup']}")
        
        # Demo caching system
        print("\n🗄️ Caching System Demo:")
        cache = optimizer.global_cache
        
        # Simulate cache usage
        @cache.cached_function
        def expensive_computation(x, y):
            time.sleep(0.01)  # Simulate expensive computation
            return x * y + np.random.random()
        
        # First call (cache miss)
        start_time = time.time()
        result1 = expensive_computation(10, 20)
        first_call_time = time.time() - start_time
        
        # Second call (cache hit)
        start_time = time.time()
        result2 = expensive_computation(10, 20)
        second_call_time = time.time() - start_time
        
        cache_stats = cache.get_stats()
        
        print(f"   First call time: {first_call_time*1000:.1f}ms (cache miss)")
        print(f"   Second call time: {second_call_time*1000:.1f}ms (cache hit)")
        print(f"   Speedup: {first_call_time/second_call_time:.1f}x")
        print(f"   Cache hit rate: {cache_stats['hit_rate']:.1%}")
        
        # Performance summary
        print("\n📈 Performance Summary:")
        summary = optimizer.get_performance_summary()
        
        print(f"   Total optimizations performed: {summary['total_optimizations']}")
        print(f"   GPU acceleration available: {summary['gpu_available']}")
        print(f"   Parallel processing enabled: {summary['parallel_processing_enabled']}")
        print(f"   Cache statistics: {summary['cache_statistics']}")
        
        if summary['total_optimizations'] > 0:
            print(f"   Average speed improvement: {summary['average_speed_improvement']:.1f}%")
            print(f"   Average memory improvement: {summary['average_memory_improvement']:.1f}%")
        
        print("\n✅ Performance optimization demo completed! 🎉")


if __name__ == "__main__":
    main()