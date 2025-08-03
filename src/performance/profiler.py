"""
Advanced performance profiling and optimization analysis.

Provides detailed profiling capabilities for ML models, data pipelines,
and API endpoints with actionable optimization recommendations.
"""

import cProfile
import gc
import logging
import pstats
import psutil
import sys
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from io import StringIO
from typing import Dict, List, Optional, Any, Callable, Tuple
import warnings

import numpy as np
import pandas as pd
from memory_profiler import profile as memory_profile

from ..logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ProfileResult:
    """Result of a profiling session."""
    function_name: str
    total_time: float
    cpu_time: float
    memory_peak: float
    memory_increment: float
    call_count: int
    hotspots: List[Dict[str, Any]]
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'function_name': self.function_name,
            'total_time_ms': self.total_time * 1000,
            'cpu_time_ms': self.cpu_time * 1000,
            'memory_peak_mb': self.memory_peak / 1024 / 1024,
            'memory_increment_mb': self.memory_increment / 1024 / 1024,
            'call_count': self.call_count,
            'hotspots': self.hotspots,
            'recommendations': self.recommendations,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""
    timestamp: datetime
    rss_memory: int
    vms_memory: int
    peak_memory: int
    available_memory: int
    memory_percent: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'rss_memory_mb': self.rss_memory / 1024 / 1024,
            'vms_memory_mb': self.vms_memory / 1024 / 1024,
            'peak_memory_mb': self.peak_memory / 1024 / 1024,
            'available_memory_mb': self.available_memory / 1024 / 1024,
            'memory_percent': self.memory_percent
        }


class AdvancedProfiler:
    """
    Advanced profiling system for ML models and data pipelines.
    
    Provides comprehensive performance analysis including CPU profiling,
    memory tracking, and optimization recommendations.
    """
    
    def __init__(self, enable_memory_tracking: bool = True):
        """
        Initialize profiler.
        
        Args:
            enable_memory_tracking: Enable detailed memory tracking
        """
        self.enable_memory_tracking = enable_memory_tracking
        self.profile_results: List[ProfileResult] = []
        self.memory_snapshots: List[MemorySnapshot] = []
        
        # Profiling state
        self._profiler: Optional[cProfile.Profile] = None
        self._memory_start: Optional[int] = None
        self._start_time: Optional[float] = None
        
        logger.info("AdvancedProfiler initialized")
    
    @contextmanager
    def profile_function(self, function_name: str):
        """
        Context manager for profiling a function.
        
        Args:
            function_name: Name of the function being profiled
        """
        logger.info(f"Starting profiling: {function_name}")
        
        # Initialize profiler
        self._profiler = cProfile.Profile()
        
        # Start memory tracking
        if self.enable_memory_tracking:
            tracemalloc.start()
            gc.collect()
            process = psutil.Process()
            self._memory_start = process.memory_info().rss
        
        # Start timing
        self._start_time = time.perf_counter()
        
        # Start CPU profiling
        self._profiler.enable()
        
        try:
            yield self
        finally:
            # Stop profiling
            self._profiler.disable()
            
            # Calculate timing
            total_time = time.perf_counter() - self._start_time
            
            # Get memory info
            memory_peak = 0
            memory_increment = 0
            
            if self.enable_memory_tracking:
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                process = psutil.Process()
                memory_end = process.memory_info().rss
                memory_increment = memory_end - self._memory_start
                memory_peak = peak
            
            # Analyze profiling results
            hotspots = self._analyze_hotspots()
            recommendations = self._generate_recommendations(hotspots, total_time, memory_increment)
            
            # Create result
            result = ProfileResult(
                function_name=function_name,
                total_time=total_time,
                cpu_time=self._get_cpu_time(),
                memory_peak=memory_peak,
                memory_increment=memory_increment,
                call_count=self._get_call_count(),
                hotspots=hotspots,
                recommendations=recommendations
            )
            
            self.profile_results.append(result)
            
            logger.info(f"Profiling completed: {function_name} ({total_time:.3f}s)")
    
    def profile_model_inference(
        self,
        model,
        X: pd.DataFrame,
        batch_sizes: Optional[List[int]] = None
    ) -> List[ProfileResult]:
        """
        Profile model inference performance.
        
        Args:
            model: Trained model to profile
            X: Input data
            batch_sizes: List of batch sizes to test
            
        Returns:
            List of profiling results
        """
        batch_sizes = batch_sizes or [1, 10, 100, 1000]
        results = []
        
        logger.info("Profiling model inference")
        
        for batch_size in batch_sizes:
            if batch_size > len(X):
                continue
            
            batch_X = X.head(batch_size)
            
            with self.profile_function(f"model_inference_batch_{batch_size}"):
                # Warmup
                for _ in range(5):
                    try:
                        _ = model.predict(batch_X)
                    except Exception:
                        break
                
                # Actual profiling
                _ = model.predict(batch_X)
            
            results.append(self.profile_results[-1])
        
        return results
    
    def profile_data_processing(
        self,
        processing_function: Callable,
        data_sizes: Optional[List[int]] = None,
        **kwargs
    ) -> List[ProfileResult]:
        """
        Profile data processing functions.
        
        Args:
            processing_function: Function to profile
            data_sizes: List of data sizes to test
            **kwargs: Arguments for processing function
            
        Returns:
            List of profiling results
        """
        data_sizes = data_sizes or [100, 1000, 10000]
        results = []
        
        logger.info("Profiling data processing")
        
        for data_size in data_sizes:
            # Generate test data
            test_data = self._generate_test_data(data_size)
            
            with self.profile_function(f"data_processing_{processing_function.__name__}_size_{data_size}"):
                # Warmup
                for _ in range(3):
                    try:
                        _ = processing_function(test_data, **kwargs)
                    except Exception:
                        break
                
                # Actual profiling
                _ = processing_function(test_data, **kwargs)
            
            results.append(self.profile_results[-1])
        
        return results
    
    def take_memory_snapshot(self) -> MemorySnapshot:
        """Take a memory usage snapshot."""
        process = psutil.Process()
        memory_info = process.memory_info()
        virtual_memory = psutil.virtual_memory()
        
        snapshot = MemorySnapshot(
            timestamp=datetime.utcnow(),
            rss_memory=memory_info.rss,
            vms_memory=memory_info.vms,
            peak_memory=memory_info.rss,  # Simplified
            available_memory=virtual_memory.available,
            memory_percent=process.memory_percent()
        )
        
        self.memory_snapshots.append(snapshot)
        return snapshot
    
    def monitor_memory_usage(self, duration_seconds: int = 60, interval_seconds: int = 1):
        """
        Monitor memory usage over time.
        
        Args:
            duration_seconds: Monitoring duration
            interval_seconds: Sampling interval
        """
        logger.info(f"Starting memory monitoring for {duration_seconds}s")
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        while time.time() < end_time:
            self.take_memory_snapshot()
            time.sleep(interval_seconds)
        
        logger.info("Memory monitoring completed")
    
    def _analyze_hotspots(self) -> List[Dict[str, Any]]:
        """Analyze CPU profiling results to find hotspots."""
        if not self._profiler:
            return []
        
        # Get stats
        stream = StringIO()
        stats = pstats.Stats(self._profiler, stream=stream)
        stats.sort_stats('cumulative')
        
        # Extract top functions
        hotspots = []
        
        try:
            # Get function statistics
            for func, (cc, nc, tt, ct, callers) in stats.stats.items():
                if tt > 0.001:  # Only include functions with significant time
                    filename, line, func_name = func
                    
                    hotspot = {
                        'function': func_name,
                        'filename': filename,
                        'line': line,
                        'call_count': cc,
                        'total_time': tt,
                        'cumulative_time': ct,
                        'time_per_call': tt / cc if cc > 0 else 0,
                        'percent_time': (tt / sum(s[2] for s in stats.stats.values())) * 100
                    }
                    
                    hotspots.append(hotspot)
            
            # Sort by total time
            hotspots.sort(key=lambda x: x['total_time'], reverse=True)
            
            # Return top 10
            return hotspots[:10]
            
        except Exception as e:
            logger.error(f"Hotspot analysis failed: {e}")
            return []
    
    def _generate_recommendations(
        self,
        hotspots: List[Dict[str, Any]],
        total_time: float,
        memory_increment: int
    ) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Performance recommendations
        if total_time > 5.0:
            recommendations.append("Consider optimizing algorithm complexity - execution time is high")
        
        if total_time > 1.0:
            recommendations.append("Profile individual components to identify bottlenecks")
        
        # Memory recommendations
        if memory_increment > 1024 * 1024 * 100:  # > 100MB
            recommendations.append("High memory usage detected - consider memory optimization")
            recommendations.append("Review data structures and consider streaming processing")
        
        if memory_increment > 1024 * 1024 * 1024:  # > 1GB
            recommendations.append("CRITICAL: Very high memory usage - immediate optimization required")
        
        # Hotspot recommendations
        if hotspots:
            top_hotspot = hotspots[0]
            if top_hotspot['percent_time'] > 50:
                recommendations.append(f"Focus optimization on {top_hotspot['function']} - consuming {top_hotspot['percent_time']:.1f}% of execution time")
            
            # Check for excessive function calls
            for hotspot in hotspots[:3]:
                if hotspot['call_count'] > 10000:
                    recommendations.append(f"Reduce calls to {hotspot['function']} - called {hotspot['call_count']} times")
        
        # General recommendations
        if len(hotspots) > 5:
            recommendations.append("Consider caching results for frequently called functions")
        
        recommendations.append("Use vectorized operations where possible")
        recommendations.append("Consider parallel processing for independent operations")
        
        return recommendations
    
    def _get_cpu_time(self) -> float:
        """Get CPU time from profiler."""
        if not self._profiler:
            return 0.0
        
        try:
            stats = pstats.Stats(self._profiler)
            total_calls, total_time = stats.total_tt, stats.total_tt
            return total_time
        except Exception:
            return 0.0
    
    def _get_call_count(self) -> int:
        """Get total call count from profiler."""
        if not self._profiler:
            return 0
        
        try:
            stats = pstats.Stats(self._profiler)
            return sum(cc for cc, nc, tt, ct, callers in stats.stats.values())
        except Exception:
            return 0
    
    def _generate_test_data(self, size: int) -> pd.DataFrame:
        """Generate test data for profiling."""
        np.random.seed(42)  # Reproducible data
        
        data = {
            'feature_1': np.random.normal(0, 1, size),
            'feature_2': np.random.uniform(0, 100, size),
            'feature_3': np.random.choice(['A', 'B', 'C'], size),
            'feature_4': np.random.exponential(2, size)
        }
        
        return pd.DataFrame(data)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance profiling summary."""
        if not self.profile_results:
            return {"message": "No profiling results available"}
        
        # Calculate aggregated metrics
        total_profiles = len(self.profile_results)
        avg_execution_time = np.mean([r.total_time for r in self.profile_results])
        avg_memory_usage = np.mean([r.memory_increment for r in self.profile_results])
        
        # Find most expensive operations
        slowest_operation = max(self.profile_results, key=lambda x: x.total_time)
        memory_intensive_operation = max(self.profile_results, key=lambda x: x.memory_increment)
        
        # Common recommendations
        all_recommendations = []
        for result in self.profile_results:
            all_recommendations.extend(result.recommendations)
        
        # Count recommendation frequency
        recommendation_counts = {}
        for rec in all_recommendations:
            recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1
        
        top_recommendations = sorted(
            recommendation_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        summary = {
            'total_profiles': total_profiles,
            'average_execution_time_ms': avg_execution_time * 1000,
            'average_memory_usage_mb': avg_memory_usage / 1024 / 1024,
            'slowest_operation': {
                'function': slowest_operation.function_name,
                'time_ms': slowest_operation.total_time * 1000
            },
            'memory_intensive_operation': {
                'function': memory_intensive_operation.function_name,
                'memory_mb': memory_intensive_operation.memory_increment / 1024 / 1024
            },
            'top_recommendations': [rec for rec, count in top_recommendations],
            'memory_snapshots': len(self.memory_snapshots)
        }
        
        return summary
    
    def export_profile_data(self, filepath: str):
        """Export profiling data to file."""
        export_data = {
            'profile_results': [result.to_dict() for result in self.profile_results],
            'memory_snapshots': [snapshot.to_dict() for snapshot in self.memory_snapshots],
            'summary': self.get_performance_summary(),
            'export_timestamp': datetime.utcnow().isoformat()
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Profile data exported to {filepath}")


class ResourceMonitor:
    """
    Real-time resource monitoring for production systems.
    
    Monitors CPU, memory, disk, and network usage with alerting.
    """
    
    def __init__(
        self,
        monitoring_interval: int = 1,
        alert_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize resource monitor.
        
        Args:
            monitoring_interval: Monitoring interval in seconds
            alert_thresholds: Thresholds for resource alerts
        """
        self.monitoring_interval = monitoring_interval
        self.alert_thresholds = alert_thresholds or {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0
        }
        
        # Monitoring state
        self.is_monitoring = False
        self.resource_history: List[Dict[str, Any]] = []
        self.alerts: List[Dict[str, Any]] = []
        
        logger.info("ResourceMonitor initialized")
    
    def start_monitoring(self, duration_seconds: Optional[int] = None):
        """
        Start resource monitoring.
        
        Args:
            duration_seconds: Monitoring duration (None for infinite)
        """
        logger.info("Starting resource monitoring")
        self.is_monitoring = True
        
        start_time = time.time()
        end_time = start_time + duration_seconds if duration_seconds else float('inf')
        
        while self.is_monitoring and time.time() < end_time:
            try:
                # Collect resource metrics
                metrics = self._collect_metrics()
                self.resource_history.append(metrics)
                
                # Check for alerts
                self._check_alerts(metrics)
                
                time.sleep(self.monitoring_interval)
                
            except KeyboardInterrupt:
                logger.info("Monitoring interrupted by user")
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(self.monitoring_interval)
        
        self.is_monitoring = False
        logger.info("Resource monitoring stopped")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.is_monitoring = False
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect current resource metrics."""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        
        # Network metrics (basic)
        network = psutil.net_io_counters()
        
        # Process metrics
        process = psutil.Process()
        process_memory = process.memory_info()
        
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'cpu': {
                'percent': cpu_percent,
                'count': cpu_count
            },
            'memory': {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent,
                'used': memory.used
            },
            'disk': {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': (disk.used / disk.total) * 100
            },
            'network': {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            },
            'process': {
                'memory_rss': process_memory.rss,
                'memory_vms': process_memory.vms,
                'cpu_percent': process.cpu_percent()
            }
        }
        
        return metrics
    
    def _check_alerts(self, metrics: Dict[str, Any]):
        """Check for resource alerts."""
        alerts = []
        
        # CPU alert
        if metrics['cpu']['percent'] > self.alert_thresholds['cpu_percent']:
            alerts.append({
                'type': 'cpu_high',
                'value': metrics['cpu']['percent'],
                'threshold': self.alert_thresholds['cpu_percent'],
                'message': f"High CPU usage: {metrics['cpu']['percent']:.1f}%"
            })
        
        # Memory alert
        if metrics['memory']['percent'] > self.alert_thresholds['memory_percent']:
            alerts.append({
                'type': 'memory_high',
                'value': metrics['memory']['percent'],
                'threshold': self.alert_thresholds['memory_percent'],
                'message': f"High memory usage: {metrics['memory']['percent']:.1f}%"
            })
        
        # Disk alert
        if metrics['disk']['percent'] > self.alert_thresholds['disk_percent']:
            alerts.append({
                'type': 'disk_high',
                'value': metrics['disk']['percent'],
                'threshold': self.alert_thresholds['disk_percent'],
                'message': f"High disk usage: {metrics['disk']['percent']:.1f}%"
            })
        
        # Add alerts with timestamp
        for alert in alerts:
            alert['timestamp'] = metrics['timestamp']
            self.alerts.append(alert)
            logger.warning(f"Resource alert: {alert['message']}")
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get resource usage summary."""
        if not self.resource_history:
            return {"message": "No monitoring data available"}
        
        # Calculate averages
        cpu_values = [m['cpu']['percent'] for m in self.resource_history]
        memory_values = [m['memory']['percent'] for m in self.resource_history]
        disk_values = [m['disk']['percent'] for m in self.resource_history]
        
        summary = {
            'monitoring_duration_minutes': len(self.resource_history) * self.monitoring_interval / 60,
            'cpu_usage': {
                'average': np.mean(cpu_values),
                'peak': np.max(cpu_values),
                'minimum': np.min(cpu_values)
            },
            'memory_usage': {
                'average': np.mean(memory_values),
                'peak': np.max(memory_values),
                'minimum': np.min(memory_values)
            },
            'disk_usage': {
                'average': np.mean(disk_values),
                'peak': np.max(disk_values),
                'minimum': np.min(disk_values)
            },
            'total_alerts': len(self.alerts),
            'alerts_by_type': self._count_alerts_by_type(),
            'data_points': len(self.resource_history)
        }
        
        return summary
    
    def _count_alerts_by_type(self) -> Dict[str, int]:
        """Count alerts by type."""
        counts = {}
        for alert in self.alerts:
            alert_type = alert['type']
            counts[alert_type] = counts.get(alert_type, 0) + 1
        return counts


# CLI interface
def main():
    """CLI interface for profiling."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance Profiling CLI")
    parser.add_argument("command", choices=["profile", "monitor"])
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds")
    parser.add_argument("--output", help="Output file path")
    
    args = parser.parse_args()
    
    if args.command == "profile":
        # Example profiling
        profiler = AdvancedProfiler()
        
        def example_function():
            # Simulate some work
            data = np.random.randn(10000, 100)
            result = np.dot(data, data.T)
            return result.sum()
        
        with profiler.profile_function("example_function"):
            result = example_function()
        
        print("Profiling Results:")
        summary = profiler.get_performance_summary()
        print(f"  Execution time: {summary['average_execution_time_ms']:.2f}ms")
        print(f"  Memory usage: {summary['average_memory_usage_mb']:.2f}MB")
        print("  Recommendations:")
        for rec in summary['top_recommendations']:
            print(f"    - {rec}")
        
        if args.output:
            profiler.export_profile_data(args.output)
            print(f"Results exported to {args.output}")
    
    elif args.command == "monitor":
        # Resource monitoring
        monitor = ResourceMonitor()
        
        try:
            monitor.start_monitoring(args.duration)
        except KeyboardInterrupt:
            pass
        
        print("Monitoring Results:")
        summary = monitor.get_resource_summary()
        print(f"  Duration: {summary['monitoring_duration_minutes']:.1f} minutes")
        print(f"  Average CPU: {summary['cpu_usage']['average']:.1f}%")
        print(f"  Peak CPU: {summary['cpu_usage']['peak']:.1f}%")
        print(f"  Average Memory: {summary['memory_usage']['average']:.1f}%")
        print(f"  Peak Memory: {summary['memory_usage']['peak']:.1f}%")
        print(f"  Total alerts: {summary['total_alerts']}")


if __name__ == "__main__":
    main()