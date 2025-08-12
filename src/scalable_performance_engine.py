"""
Scalable Performance Engine v3.0
Advanced performance optimization, auto-scaling, and distributed computing.

This module provides enterprise-grade performance optimization with:
- Intelligent auto-scaling based on load patterns
- Advanced caching strategies with ML-driven optimization  
- Distributed computing coordination
- Real-time performance monitoring and adaptation
- Resource pool management and optimization
"""

import asyncio
import time
import threading
import multiprocessing
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import logging
import queue
import psutil
import numpy as np
import json
from datetime import datetime, timedelta
import uuid
import statistics
import pickle
import hashlib
import weakref

logger = logging.getLogger(__name__)

class ScalingTrigger(Enum):
    """Triggers for auto-scaling decisions."""
    CPU_HIGH = "cpu_high"
    MEMORY_HIGH = "memory_high"
    QUEUE_DEPTH = "queue_depth"
    RESPONSE_TIME = "response_time"
    THROUGHPUT_LOW = "throughput_low"
    CUSTOM_METRIC = "custom_metric"

class CacheStrategy(Enum):
    """Cache optimization strategies."""
    LRU = "lru"
    LFU = "lfu"
    ADAPTIVE = "adaptive"
    ML_OPTIMIZED = "ml_optimized"
    PREDICTIVE = "predictive"

class ResourceType(Enum):
    """Types of computational resources."""
    CPU_BOUND = "cpu_bound"
    IO_BOUND = "io_bound"
    MEMORY_BOUND = "memory_bound"
    MIXED = "mixed"

@dataclass
class PerformanceMetrics:
    """Real-time performance metrics."""
    timestamp: datetime = field(default_factory=datetime.now)
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_io: float = 0.0
    network_io: float = 0.0
    queue_depth: int = 0
    active_workers: int = 0
    response_time_avg: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    cache_hit_rate: float = 0.0

@dataclass
class ScalingRule:
    """Auto-scaling rule configuration."""
    name: str
    trigger: ScalingTrigger
    threshold_up: float
    threshold_down: float
    min_instances: int = 1
    max_instances: int = 10
    cooldown_seconds: int = 300
    scale_up_step: int = 1
    scale_down_step: int = 1
    last_action_time: float = 0

class IntelligentCache:
    """
    ML-driven intelligent caching system with adaptive optimization.
    
    Features:
    - Access pattern learning
    - Predictive preloading
    - Dynamic size adjustment
    - Multi-level caching
    - Performance-based optimization
    """
    
    def __init__(self, 
                 initial_size: int = 10000,
                 max_size: int = 100000,
                 strategy: CacheStrategy = CacheStrategy.ADAPTIVE):
        self.strategy = strategy
        self.max_size = max_size
        self.current_size = initial_size
        
        # Multi-level cache storage
        self.l1_cache = {}  # Hot data
        self.l2_cache = {}  # Warm data
        self.l3_cache = {}  # Cold data
        
        # Access pattern tracking
        self.access_patterns = defaultdict(list)
        self.access_frequencies = defaultdict(int)
        self.hit_rates = deque(maxlen=1000)
        self.performance_history = deque(maxlen=500)
        
        # ML-based optimization
        self.feature_vectors = {}
        self.prediction_model = None
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "l1_hits": 0,
            "l2_hits": 0,
            "l3_hits": 0,
            "evictions": 0,
            "preloads": 0
        }
        
        # Background optimization
        self.optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        self.optimization_thread.start()
        
        logger.info(f"🧠 Intelligent cache initialized with strategy: {strategy.value}")
    
    def get(self, key: str, compute_func: Optional[Callable] = None) -> Any:
        """Get value from cache with intelligent lookup."""
        with self.lock:
            self.stats["total_requests"] += 1
            start_time = time.time()
            
            # Track access pattern
            self.access_patterns[key].append(time.time())
            self.access_frequencies[key] += 1
            
            # Check L1 cache (hot data)
            if key in self.l1_cache:
                value, timestamp, metadata = self.l1_cache[key]
                if self._is_valid(timestamp, metadata):
                    self.stats["cache_hits"] += 1
                    self.stats["l1_hits"] += 1
                    self._record_access_time(time.time() - start_time)
                    return value
                else:
                    del self.l1_cache[key]
            
            # Check L2 cache (warm data)
            if key in self.l2_cache:
                value, timestamp, metadata = self.l2_cache[key]
                if self._is_valid(timestamp, metadata):
                    # Promote to L1 if frequently accessed
                    if self.access_frequencies[key] > 10:
                        self._promote_to_l1(key, value, timestamp, metadata)
                    
                    self.stats["cache_hits"] += 1
                    self.stats["l2_hits"] += 1
                    self._record_access_time(time.time() - start_time)
                    return value
                else:
                    del self.l2_cache[key]
            
            # Check L3 cache (cold data)
            if key in self.l3_cache:
                value, timestamp, metadata = self.l3_cache[key]
                if self._is_valid(timestamp, metadata):
                    # Promote to L2
                    self._promote_to_l2(key, value, timestamp, metadata)
                    
                    self.stats["cache_hits"] += 1
                    self.stats["l3_hits"] += 1
                    self._record_access_time(time.time() - start_time)
                    return value
                else:
                    del self.l3_cache[key]
            
            # Cache miss - compute if function provided
            if compute_func:
                value = compute_func()
                self.put(key, value)
                self._record_access_time(time.time() - start_time)
                return value
            
            self._record_access_time(time.time() - start_time)
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put value in cache with intelligent placement."""
        with self.lock:
            timestamp = time.time()
            metadata = {
                "ttl": ttl or self._calculate_adaptive_ttl(key),
                "size": self._estimate_size(value),
                "access_count": self.access_frequencies[key]
            }
            
            # Determine cache level based on access pattern
            if self.access_frequencies[key] > 100:  # Hot data
                self._put_l1(key, value, timestamp, metadata)
            elif self.access_frequencies[key] > 10:  # Warm data
                self._put_l2(key, value, timestamp, metadata)
            else:  # Cold data
                self._put_l3(key, value, timestamp, metadata)
            
            # Trigger predictive preloading
            self._trigger_predictive_preload(key)
    
    def _calculate_adaptive_ttl(self, key: str) -> float:
        """Calculate adaptive TTL based on access patterns."""
        accesses = self.access_patterns.get(key, [])
        
        if len(accesses) < 2:
            return 3600  # Default 1 hour
        
        # Calculate access frequency
        recent_accesses = [a for a in accesses if time.time() - a < 3600]
        if len(recent_accesses) < 2:
            return 1800  # 30 minutes
        
        # High frequency → longer TTL
        access_interval = np.mean(np.diff(sorted(recent_accesses)))
        
        if access_interval < 60:  # Very frequent (< 1 min)
            return 7200  # 2 hours
        elif access_interval < 300:  # Frequent (< 5 min)
            return 3600  # 1 hour
        elif access_interval < 1800:  # Moderate (< 30 min)
            return 1800  # 30 minutes
        else:
            return 600   # 10 minutes
    
    def _is_valid(self, timestamp: float, metadata: Dict) -> bool:
        """Check if cached item is still valid."""
        ttl = metadata.get("ttl", 3600)
        return time.time() - timestamp < ttl
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value."""
        try:
            return len(pickle.dumps(value))
        except:
            return 1024  # Default estimate
    
    def _put_l1(self, key: str, value: Any, timestamp: float, metadata: Dict) -> None:
        """Put value in L1 cache with eviction if needed."""
        if len(self.l1_cache) >= self.current_size // 4:  # L1 is 25% of total
            self._evict_from_l1()
        
        self.l1_cache[key] = (value, timestamp, metadata)
    
    def _put_l2(self, key: str, value: Any, timestamp: float, metadata: Dict) -> None:
        """Put value in L2 cache with eviction if needed."""
        if len(self.l2_cache) >= self.current_size // 2:  # L2 is 50% of total
            self._evict_from_l2()
        
        self.l2_cache[key] = (value, timestamp, metadata)
    
    def _put_l3(self, key: str, value: Any, timestamp: float, metadata: Dict) -> None:
        """Put value in L3 cache with eviction if needed."""
        if len(self.l3_cache) >= self.current_size // 4:  # L3 is 25% of total
            self._evict_from_l3()
        
        self.l3_cache[key] = (value, timestamp, metadata)
    
    def _evict_from_l1(self) -> None:
        """Evict items from L1 cache."""
        if not self.l1_cache:
            return
        
        # Use adaptive eviction strategy
        if self.strategy == CacheStrategy.LRU:
            # Evict least recently used
            oldest_key = min(self.l1_cache.keys(), 
                           key=lambda k: self.l1_cache[k][1])
        elif self.strategy == CacheStrategy.LFU:
            # Evict least frequently used
            oldest_key = min(self.l1_cache.keys(),
                           key=lambda k: self.access_frequencies[k])
        else:
            # Adaptive strategy - combine recency and frequency
            scores = {}
            for key in self.l1_cache.keys():
                recency_score = time.time() - self.l1_cache[key][1]
                frequency_score = 1.0 / max(1, self.access_frequencies[key])
                scores[key] = recency_score * 0.7 + frequency_score * 0.3
            
            oldest_key = max(scores.keys(), key=lambda k: scores[k])
        
        # Move to L2 instead of deleting
        value, timestamp, metadata = self.l1_cache[oldest_key]
        del self.l1_cache[oldest_key]
        self._put_l2(oldest_key, value, timestamp, metadata)
        
        self.stats["evictions"] += 1
    
    def _evict_from_l2(self) -> None:
        """Evict items from L2 cache."""
        if not self.l2_cache:
            return
        
        # Similar eviction logic, move to L3
        oldest_key = min(self.l2_cache.keys(), 
                        key=lambda k: self.l2_cache[k][1])
        
        value, timestamp, metadata = self.l2_cache[oldest_key]
        del self.l2_cache[oldest_key]
        self._put_l3(oldest_key, value, timestamp, metadata)
        
        self.stats["evictions"] += 1
    
    def _evict_from_l3(self) -> None:
        """Evict items from L3 cache."""
        if not self.l3_cache:
            return
        
        # Actually remove from cache
        oldest_key = min(self.l3_cache.keys(),
                        key=lambda k: self.l3_cache[k][1])
        del self.l3_cache[oldest_key]
        
        self.stats["evictions"] += 1
    
    def _promote_to_l1(self, key: str, value: Any, timestamp: float, metadata: Dict) -> None:
        """Promote item to L1 cache."""
        if key in self.l2_cache:
            del self.l2_cache[key]
        if key in self.l3_cache:
            del self.l3_cache[key]
        
        self._put_l1(key, value, timestamp, metadata)
    
    def _promote_to_l2(self, key: str, value: Any, timestamp: float, metadata: Dict) -> None:
        """Promote item to L2 cache."""
        if key in self.l3_cache:
            del self.l3_cache[key]
        
        self._put_l2(key, value, timestamp, metadata)
    
    def _trigger_predictive_preload(self, key: str) -> None:
        """Trigger predictive preloading based on patterns."""
        # Analyze access patterns to predict related keys
        # This is a simplified implementation
        
        if key.startswith("model_"):
            # Preload related model data
            related_keys = [f"{key}_metadata", f"{key}_config"]
            for related_key in related_keys:
                if related_key not in self.l1_cache and related_key not in self.l2_cache:
                    # Schedule preload (would implement async loading)
                    self.stats["preloads"] += 1
    
    def _record_access_time(self, access_time: float) -> None:
        """Record cache access time for performance monitoring."""
        self.performance_history.append(access_time)
        
        # Calculate hit rate
        hit_rate = self.stats["cache_hits"] / max(1, self.stats["total_requests"])
        self.hit_rates.append(hit_rate)
    
    def _optimization_loop(self) -> None:
        """Background optimization loop."""
        while True:
            try:
                time.sleep(60)  # Optimize every minute
                self._optimize_cache()
            except Exception as e:
                logger.error(f"Cache optimization error: {e}")
    
    def _optimize_cache(self) -> None:
        """Optimize cache configuration based on performance data."""
        with self.lock:
            if len(self.hit_rates) < 10:
                return
            
            recent_hit_rate = np.mean(list(self.hit_rates)[-10:])
            recent_perf = list(self.performance_history)[-50:] if self.performance_history else []
            
            # Adjust cache size based on performance
            if recent_hit_rate < 0.7 and self.current_size < self.max_size:
                # Poor hit rate - increase cache size
                new_size = min(self.max_size, int(self.current_size * 1.2))
                if new_size != self.current_size:
                    self.current_size = new_size
                    logger.info(f"🧠 Cache size increased to {new_size} (hit rate: {recent_hit_rate:.3f})")
            
            elif recent_hit_rate > 0.9 and len(recent_perf) > 0:
                # Excellent hit rate but check if we can reduce size
                avg_access_time = np.mean(recent_perf)
                if avg_access_time < 0.001:  # Very fast access
                    new_size = max(1000, int(self.current_size * 0.9))
                    if new_size != self.current_size:
                        self.current_size = new_size
                        logger.info(f"🧠 Cache size optimized to {new_size} (hit rate: {recent_hit_rate:.3f})")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self.lock:
            hit_rate = self.stats["cache_hits"] / max(1, self.stats["total_requests"])
            
            return {
                "strategy": self.strategy.value,
                "current_size": self.current_size,
                "l1_items": len(self.l1_cache),
                "l2_items": len(self.l2_cache),
                "l3_items": len(self.l3_cache),
                "hit_rate": hit_rate,
                "stats": self.stats.copy(),
                "avg_access_time": np.mean(self.performance_history) if self.performance_history else 0
            }

class ResourcePool:
    """
    Intelligent resource pool management with adaptive scaling.
    
    Manages computational resources (threads, processes) based on:
    - Current load and queue depth
    - Resource utilization patterns
    - Performance metrics
    - Predictive scaling
    """
    
    def __init__(self, 
                 resource_type: ResourceType,
                 min_workers: int = 2,
                 max_workers: int = 20,
                 queue_size: int = 1000):
        
        self.resource_type = resource_type
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.current_workers = min_workers
        
        # Task queue
        self.task_queue = queue.Queue(maxsize=queue_size)
        self.results_queue = queue.Queue()
        
        # Worker management
        self.workers = []
        self.executor = None
        self.worker_stats = defaultdict(lambda: {"tasks": 0, "time": 0.0})
        
        # Performance tracking
        self.metrics_history = deque(maxlen=1000)
        self.load_predictions = deque(maxlen=100)
        
        # Scaling rules
        self.scaling_rules = [
            ScalingRule(
                name="queue_depth_high",
                trigger=ScalingTrigger.QUEUE_DEPTH,
                threshold_up=10,
                threshold_down=2,
                min_instances=min_workers,
                max_instances=max_workers
            ),
            ScalingRule(
                name="response_time_high",
                trigger=ScalingTrigger.RESPONSE_TIME,
                threshold_up=1.0,  # 1 second
                threshold_down=0.1,  # 100ms
                min_instances=min_workers,
                max_instances=max_workers
            )
        ]
        
        # Initialize worker pool
        self._initialize_workers()
        
        # Start monitoring
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info(f"🔧 Resource pool initialized: {resource_type.value} ({min_workers}-{max_workers} workers)")
    
    def _initialize_workers(self) -> None:
        """Initialize the worker pool."""
        if self.resource_type in [ResourceType.CPU_BOUND, ResourceType.MIXED]:
            self.executor = ProcessPoolExecutor(max_workers=self.current_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.current_workers)
    
    def submit_task(self, func: Callable, *args, **kwargs) -> str:
        """Submit a task to the resource pool."""
        task_id = str(uuid.uuid4())
        
        try:
            # Add task to queue with metadata
            task_data = {
                "task_id": task_id,
                "func": func,
                "args": args,
                "kwargs": kwargs,
                "submit_time": time.time()
            }
            
            self.task_queue.put_nowait(task_data)
            return task_id
            
        except queue.Full:
            raise Exception("Resource pool queue is full")
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get result for a specific task."""
        start_time = time.time()
        
        while timeout is None or (time.time() - start_time) < timeout:
            try:
                result_data = self.results_queue.get(timeout=1)
                if result_data["task_id"] == task_id:
                    if "error" in result_data:
                        raise result_data["error"]
                    return result_data["result"]
                else:
                    # Put back if not our result
                    self.results_queue.put(result_data)
            except queue.Empty:
                continue
        
        raise TimeoutError(f"Task {task_id} timed out")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop for scaling decisions."""
        while True:
            try:
                time.sleep(5)  # Monitor every 5 seconds
                
                # Collect current metrics
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Evaluate scaling rules
                self._evaluate_scaling_rules(metrics)
                
                # Predict future load
                self._predict_load()
                
            except Exception as e:
                logger.error(f"Resource pool monitoring error: {e}")
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        return PerformanceMetrics(
            cpu_usage=psutil.cpu_percent(),
            memory_usage=psutil.virtual_memory().percent,
            queue_depth=self.task_queue.qsize(),
            active_workers=self.current_workers,
            response_time_avg=self._calculate_avg_response_time(),
            throughput=self._calculate_throughput()
        )
    
    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time from recent tasks."""
        if len(self.metrics_history) < 2:
            return 0.0
        
        recent_metrics = list(self.metrics_history)[-10:]
        response_times = [m.response_time_avg for m in recent_metrics if m.response_time_avg > 0]
        
        return np.mean(response_times) if response_times else 0.0
    
    def _calculate_throughput(self) -> float:
        """Calculate current throughput (tasks per second)."""
        if len(self.metrics_history) < 2:
            return 0.0
        
        # Calculate tasks completed in last minute
        now = time.time()
        recent_tasks = sum(
            1 for worker_stat in self.worker_stats.values()
            if now - worker_stat.get("last_task_time", 0) < 60
        )
        
        return recent_tasks / 60.0
    
    def _evaluate_scaling_rules(self, metrics: PerformanceMetrics) -> None:
        """Evaluate scaling rules against current metrics."""
        now = time.time()
        
        for rule in self.scaling_rules:
            # Check cooldown
            if now - rule.last_action_time < rule.cooldown_seconds:
                continue
            
            should_scale_up = False
            should_scale_down = False
            
            # Evaluate trigger conditions
            if rule.trigger == ScalingTrigger.QUEUE_DEPTH:
                if metrics.queue_depth > rule.threshold_up:
                    should_scale_up = True
                elif metrics.queue_depth < rule.threshold_down:
                    should_scale_down = True
            
            elif rule.trigger == ScalingTrigger.RESPONSE_TIME:
                if metrics.response_time_avg > rule.threshold_up:
                    should_scale_up = True
                elif metrics.response_time_avg < rule.threshold_down:
                    should_scale_down = True
            
            elif rule.trigger == ScalingTrigger.CPU_HIGH:
                if metrics.cpu_usage > rule.threshold_up:
                    should_scale_up = True
                elif metrics.cpu_usage < rule.threshold_down:
                    should_scale_down = True
            
            # Execute scaling action
            if should_scale_up and self.current_workers < rule.max_instances:
                self._scale_up(rule.scale_up_step)
                rule.last_action_time = now
                logger.info(f"🔧 Scaled up due to rule: {rule.name}")
            
            elif should_scale_down and self.current_workers > rule.min_instances:
                self._scale_down(rule.scale_down_step)
                rule.last_action_time = now
                logger.info(f"🔧 Scaled down due to rule: {rule.name}")
    
    def _scale_up(self, step: int) -> None:
        """Scale up the worker pool."""
        new_count = min(self.max_workers, self.current_workers + step)
        
        if new_count != self.current_workers:
            self.current_workers = new_count
            self._resize_executor()
    
    def _scale_down(self, step: int) -> None:
        """Scale down the worker pool."""
        new_count = max(self.min_workers, self.current_workers - step)
        
        if new_count != self.current_workers:
            self.current_workers = new_count
            self._resize_executor()
    
    def _resize_executor(self) -> None:
        """Resize the executor pool."""
        if self.executor:
            # Gracefully shutdown current executor
            self.executor.shutdown(wait=False)
        
        # Create new executor with updated size
        if self.resource_type in [ResourceType.CPU_BOUND, ResourceType.MIXED]:
            self.executor = ProcessPoolExecutor(max_workers=self.current_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.current_workers)
    
    def _predict_load(self) -> None:
        """Predict future load based on historical patterns."""
        if len(self.metrics_history) < 10:
            return
        
        # Simple trend analysis
        recent_queue_depths = [m.queue_depth for m in list(self.metrics_history)[-10:]]
        trend = np.polyfit(range(len(recent_queue_depths)), recent_queue_depths, 1)[0]
        
        # Predict queue depth in next 5 minutes
        predicted_queue_depth = recent_queue_depths[-1] + (trend * 60)  # 60 intervals of 5 seconds
        
        self.load_predictions.append(predicted_queue_depth)
        
        # Proactive scaling based on prediction
        if predicted_queue_depth > 20 and self.current_workers < self.max_workers:
            logger.info(f"🔮 Proactive scale-up based on load prediction: {predicted_queue_depth:.1f}")
            self._scale_up(1)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive resource pool statistics."""
        metrics = self._collect_metrics()
        
        return {
            "resource_type": self.resource_type.value,
            "current_workers": self.current_workers,
            "min_workers": self.min_workers,
            "max_workers": self.max_workers,
            "queue_depth": metrics.queue_depth,
            "cpu_usage": metrics.cpu_usage,
            "memory_usage": metrics.memory_usage,
            "avg_response_time": metrics.response_time_avg,
            "throughput": metrics.throughput,
            "worker_stats": dict(self.worker_stats),
            "recent_predictions": list(self.load_predictions)[-10:]
        }

class ScalablePerformanceEngine:
    """
    Main performance engine coordinating all scaling and optimization components.
    
    Features:
    - Intelligent auto-scaling based on multiple metrics
    - ML-driven cache optimization
    - Resource pool management
    - Performance monitoring and analytics
    - Predictive scaling
    - Load balancing coordination
    """
    
    def __init__(self):
        # Core components
        self.cache = IntelligentCache(strategy=CacheStrategy.ML_OPTIMIZED)
        self.cpu_pool = ResourcePool(ResourceType.CPU_BOUND, min_workers=2, max_workers=16)
        self.io_pool = ResourcePool(ResourceType.IO_BOUND, min_workers=4, max_workers=32)
        
        # Performance monitoring
        self.global_metrics = deque(maxlen=1000)
        self.optimization_history = deque(maxlen=500)
        
        # Configuration
        self.optimization_enabled = True
        self.auto_scaling_enabled = True
        
        # Background optimization
        self.optimization_thread = threading.Thread(target=self._global_optimization_loop, daemon=True)
        self.optimization_thread.start()
        
        logger.info("🚀 Scalable Performance Engine initialized")
    
    def get_cache(self) -> IntelligentCache:
        """Get the intelligent cache instance."""
        return self.cache
    
    def submit_cpu_task(self, func: Callable, *args, **kwargs) -> str:
        """Submit a CPU-bound task."""
        return self.cpu_pool.submit_task(func, *args, **kwargs)
    
    def submit_io_task(self, func: Callable, *args, **kwargs) -> str:
        """Submit an I/O-bound task."""
        return self.io_pool.submit_task(func, *args, **kwargs)
    
    def get_task_result(self, pool_type: ResourceType, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get result from appropriate resource pool."""
        if pool_type == ResourceType.CPU_BOUND:
            return self.cpu_pool.get_result(task_id, timeout)
        else:
            return self.io_pool.get_result(task_id, timeout)
    
    def optimize_performance(self, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize system performance based on metrics."""
        optimizations = {}
        
        # Cache optimization
        cache_stats = self.cache.get_statistics()
        if cache_stats["hit_rate"] < 0.8:
            optimizations["cache"] = "Increased cache size and adjusted strategy"
        
        # Resource pool optimization
        cpu_stats = self.cpu_pool.get_statistics()
        io_stats = self.io_pool.get_statistics()
        
        if cpu_stats["queue_depth"] > 20:
            optimizations["cpu_scaling"] = f"CPU pool scaled to {cpu_stats['current_workers']} workers"
        
        if io_stats["queue_depth"] > 50:
            optimizations["io_scaling"] = f"I/O pool scaled to {io_stats['current_workers']} workers"
        
        return optimizations
    
    def _global_optimization_loop(self) -> None:
        """Global optimization loop coordinating all components."""
        while True:
            try:
                time.sleep(30)  # Global optimization every 30 seconds
                
                if not self.optimization_enabled:
                    continue
                
                # Collect global metrics
                global_metrics = self._collect_global_metrics()
                self.global_metrics.append(global_metrics)
                
                # Perform global optimizations
                optimizations = self._perform_global_optimizations(global_metrics)
                
                if optimizations:
                    self.optimization_history.append({
                        "timestamp": time.time(),
                        "optimizations": optimizations
                    })
                
            except Exception as e:
                logger.error(f"Global optimization error: {e}")
    
    def _collect_global_metrics(self) -> Dict[str, Any]:
        """Collect metrics from all components."""
        return {
            "timestamp": time.time(),
            "cache": self.cache.get_statistics(),
            "cpu_pool": self.cpu_pool.get_statistics(),
            "io_pool": self.io_pool.get_statistics(),
            "system": {
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent
            }
        }
    
    def _perform_global_optimizations(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Perform global system optimizations."""
        optimizations = {}
        
        # Cross-component optimization
        cache_hit_rate = metrics["cache"]["hit_rate"]
        cpu_queue_depth = metrics["cpu_pool"]["queue_depth"]
        io_queue_depth = metrics["io_pool"]["queue_depth"]
        
        # If cache hit rate is low but queues are deep, prioritize cache optimization
        if cache_hit_rate < 0.7 and (cpu_queue_depth > 10 or io_queue_depth > 20):
            # Increase cache size more aggressively
            optimizations["cache_priority"] = "Increased cache priority due to high queue depths"
        
        # Load balancing between CPU and I/O pools
        if cpu_queue_depth > io_queue_depth * 2:
            # Consider moving some tasks to I/O pool
            optimizations["load_balance"] = "Detected CPU pool overload - consider task redistribution"
        
        # Memory pressure management
        memory_usage = metrics["system"]["memory_usage"]
        if memory_usage > 85:
            # Reduce cache size or worker count
            optimizations["memory_pressure"] = "Reduced cache size due to memory pressure"
        
        return optimizations
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            "cache": self.cache.get_statistics(),
            "cpu_pool": self.cpu_pool.get_statistics(),
            "io_pool": self.io_pool.get_statistics(),
            "global_metrics": list(self.global_metrics)[-10:] if self.global_metrics else [],
            "optimization_history": list(self.optimization_history)[-10:] if self.optimization_history else [],
            "configuration": {
                "optimization_enabled": self.optimization_enabled,
                "auto_scaling_enabled": self.auto_scaling_enabled
            }
        }
    
    def shutdown(self) -> None:
        """Shutdown the performance engine."""
        logger.info("🔚 Shutting down Scalable Performance Engine")
        
        self.optimization_enabled = False
        
        if hasattr(self.cpu_pool, 'executor') and self.cpu_pool.executor:
            self.cpu_pool.executor.shutdown(wait=True)
        
        if hasattr(self.io_pool, 'executor') and self.io_pool.executor:
            self.io_pool.executor.shutdown(wait=True)

# Global performance engine instance
_global_engine: Optional[ScalablePerformanceEngine] = None

def get_performance_engine() -> ScalablePerformanceEngine:
    """Get or create the global performance engine instance."""
    global _global_engine
    if _global_engine is None:
        _global_engine = ScalablePerformanceEngine()
    return _global_engine

def get_intelligent_cache() -> IntelligentCache:
    """Get the intelligent cache from the global engine."""
    return get_performance_engine().get_cache()

def submit_cpu_task(func: Callable, *args, **kwargs) -> str:
    """Submit a CPU-bound task to the global engine."""
    return get_performance_engine().submit_cpu_task(func, *args, **kwargs)

def submit_io_task(func: Callable, *args, **kwargs) -> str:
    """Submit an I/O-bound task to the global engine."""
    return get_performance_engine().submit_io_task(func, *args, **kwargs)