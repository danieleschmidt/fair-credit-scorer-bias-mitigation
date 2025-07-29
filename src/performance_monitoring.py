"""
Performance monitoring and metrics collection for the fair credit scorer.

This module provides comprehensive performance monitoring capabilities including
resource usage tracking, custom metrics, and health checks.
"""

import time
import psutil
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from contextlib import contextmanager
import yaml
from pathlib import Path


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    execution_time: float = 0.0
    timestamp: float = field(default_factory=time.time)
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """Advanced performance monitoring system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize performance monitor with configuration."""
        self.config = self._load_config(config_path)
        self.metrics_history: List[PerformanceMetrics] = []
        self.logger = logging.getLogger(__name__)
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load monitoring configuration."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "monitoring.yaml"
            
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"Config file not found: {config_path}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default monitoring configuration."""
        return {
            "monitoring": {
                "enabled": True,
                "benchmarks": {
                    "baseline_model": {
                        "target_accuracy": 0.80,
                        "max_training_time": 30,
                        "memory_limit": 500
                    }
                },
                "resources": {
                    "cpu_threshold": 80,
                    "memory_threshold": 85
                },
                "alerts": {
                    "performance_degradation": 20
                }
            }
        }
    
    @contextmanager
    def monitor_execution(self, operation_name: str = "operation"):
        """Context manager for monitoring operation performance."""
        if not self.config.get("monitoring", {}).get("enabled", True):
            yield
            return
            
        start_time = time.time()
        start_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        start_cpu = psutil.cpu_percent()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
            end_cpu = psutil.cpu_percent()
            
            metrics = PerformanceMetrics(
                cpu_percent=end_cpu,
                memory_mb=end_memory - start_memory,
                execution_time=end_time - start_time,
                custom_metrics={"operation": operation_name}
            )
            
            self.metrics_history.append(metrics)
            self._check_thresholds(metrics, operation_name)
    
    def _check_thresholds(self, metrics: PerformanceMetrics, operation: str):
        """Check if performance metrics exceed configured thresholds."""
        thresholds = self.config.get("monitoring", {}).get("resources", {})
        
        # CPU threshold check
        cpu_threshold = thresholds.get("cpu_threshold", 80)
        if metrics.cpu_percent > cpu_threshold:
            self.logger.warning(
                f"CPU usage {metrics.cpu_percent:.1f}% exceeds threshold "
                f"{cpu_threshold}% for operation: {operation}"
            )
        
        # Memory threshold check
        memory_threshold = thresholds.get("memory_threshold", 85)
        current_memory_percent = psutil.virtual_memory().percent
        if current_memory_percent > memory_threshold:
            self.logger.warning(
                f"Memory usage {current_memory_percent:.1f}% exceeds threshold "
                f"{memory_threshold}% for operation: {operation}"
            )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics."""
        if not self.metrics_history:
            return {"status": "no_data", "total_operations": 0}
        
        total_operations = len(self.metrics_history)
        avg_cpu = sum(m.cpu_percent for m in self.metrics_history) / total_operations
        avg_memory = sum(m.memory_mb for m in self.metrics_history) / total_operations
        avg_time = sum(m.execution_time for m in self.metrics_history) / total_operations
        
        return {
            "status": "active",
            "total_operations": total_operations,
            "average_metrics": {
                "cpu_percent": round(avg_cpu, 2),
                "memory_mb": round(avg_memory, 2),
                "execution_time": round(avg_time, 4)
            },
            "current_system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent
            }
        }
    
    def check_system_health(self) -> Dict[str, Any]:
        """Perform comprehensive system health check."""
        health_status = {
            "overall_status": "healthy",
            "checks": {},
            "timestamp": time.time()
        }
        
        # CPU check
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_threshold = self.config.get("monitoring", {}).get("resources", {}).get("cpu_threshold", 80)
        health_status["checks"]["cpu"] = {
            "status": "healthy" if cpu_percent < cpu_threshold else "warning",
            "value": cpu_percent,
            "threshold": cpu_threshold
        }
        
        # Memory check
        memory = psutil.virtual_memory()
        memory_threshold = self.config.get("monitoring", {}).get("resources", {}).get("memory_threshold", 85)
        health_status["checks"]["memory"] = {
            "status": "healthy" if memory.percent < memory_threshold else "warning",
            "value": memory.percent,
            "threshold": memory_threshold
        }
        
        # Disk check
        disk = psutil.disk_usage('/')
        disk_threshold = self.config.get("monitoring", {}).get("resources", {}).get("disk_threshold", 90)
        health_status["checks"]["disk"] = {
            "status": "healthy" if disk.percent < disk_threshold else "warning",
            "value": disk.percent,
            "threshold": disk_threshold
        }
        
        # Update overall status
        if any(check["status"] == "warning" for check in health_status["checks"].values()):
            health_status["overall_status"] = "warning"
        
        return health_status
    
    def export_metrics(self, format_type: str = "json") -> str:
        """Export collected metrics in specified format."""
        summary = self.get_performance_summary()
        health = self.check_system_health()
        
        export_data = {
            "performance_summary": summary,
            "health_status": health,
            "export_timestamp": time.time(),
            "configuration": self.config.get("monitoring", {})
        }
        
        if format_type.lower() == "json":
            import json
            return json.dumps(export_data, indent=2)
        elif format_type.lower() == "yaml":
            return yaml.dump(export_data, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")


# Singleton instance for global access
_monitor_instance: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = PerformanceMonitor()
    return _monitor_instance


# Convenience decorators and functions
def monitor_performance(operation_name: str = "operation"):
    """Decorator for monitoring function performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            with monitor.monitor_execution(operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


if __name__ == "__main__":
    # Example usage and testing
    monitor = PerformanceMonitor()
    
    # Test monitoring
    with monitor.monitor_execution("test_operation"):
        time.sleep(0.1)  # Simulate work
    
    # Print results
    print("Performance Summary:")
    print(monitor.export_metrics("json"))
    
    print("\nHealth Check:")
    import json
    print(json.dumps(monitor.check_system_health(), indent=2))