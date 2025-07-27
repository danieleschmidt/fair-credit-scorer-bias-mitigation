"""Health check and metrics endpoints for monitoring."""

import json
import time
from typing import Dict, Any
import psutil
import os
from pathlib import Path

from src.fairness_metrics import compute_fairness_metrics
from src.data_loader_preprocessor import generate_data
import numpy as np


class HealthCheck:
    """Health check and monitoring utilities."""
    
    def __init__(self):
        self.start_time = time.time()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get basic health status of the application."""
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "uptime_seconds": time.time() - self.start_time,
            "version": "0.2.0",
            "environment": os.getenv("ENVIRONMENT", "development")
        }
    
    def get_readiness_status(self) -> Dict[str, Any]:
        """Check if the application is ready to serve requests."""
        try:
            # Test critical functionality
            self._test_data_generation()
            self._test_fairness_metrics()
            
            return {
                "status": "ready",
                "timestamp": time.time(),
                "checks": {
                    "data_generation": "pass",
                    "fairness_metrics": "pass"
                }
            }
        except Exception as e:
            return {
                "status": "not_ready",
                "timestamp": time.time(),
                "error": str(e),
                "checks": {
                    "data_generation": "fail",
                    "fairness_metrics": "fail"
                }
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get system and application metrics."""
        process = psutil.Process(os.getpid())
        
        return {
            "timestamp": time.time(),
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None
            },
            "process": {
                "pid": process.pid,
                "cpu_percent": process.cpu_percent(),
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "memory_percent": process.memory_percent(),
                "threads": process.num_threads(),
                "open_files": len(process.open_files()) if hasattr(process, 'open_files') else None
            },
            "application": {
                "uptime_seconds": time.time() - self.start_time,
                "version": "0.2.0",
                "environment": os.getenv("ENVIRONMENT", "development")
            }
        }
    
    def _test_data_generation(self):
        """Test data generation functionality."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=True) as tmp:
            data = generate_data(tmp.name, n_samples=100, random_state=42)
            if len(data) != 100:
                raise RuntimeError("Data generation test failed")
    
    def _test_fairness_metrics(self):
        """Test fairness metrics computation."""
        np.random.seed(42)
        n_samples = 100
        y_true = np.random.randint(0, 2, n_samples)
        y_pred = np.random.randint(0, 2, n_samples)
        y_prob = np.random.random(n_samples)
        protected_attr = np.random.randint(0, 2, n_samples)
        
        metrics = compute_fairness_metrics(y_true, y_pred, y_prob, protected_attr)
        if "demographic_parity_difference" not in metrics:
            raise RuntimeError("Fairness metrics test failed")


class MetricsCollector:
    """Collect and export application metrics."""
    
    def __init__(self):
        self.metrics_file = Path("metrics.json")
        self.counters = {}
        self.gauges = {}
        self.histograms = {}
    
    def increment_counter(self, name: str, value: int = 1, labels: Dict[str, str] = None):
        """Increment a counter metric."""
        key = f"{name}_{hash(str(labels))}" if labels else name
        self.counters[key] = self.counters.get(key, 0) + value
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric."""
        key = f"{name}_{hash(str(labels))}" if labels else name
        self.gauges[key] = value
    
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a histogram metric."""
        key = f"{name}_{hash(str(labels))}" if labels else name
        if key not in self.histograms:
            self.histograms[key] = []
        self.histograms[key].append(value)
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export all metrics."""
        return {
            "timestamp": time.time(),
            "counters": self.counters,
            "gauges": self.gauges,
            "histograms": {
                name: {
                    "count": len(values),
                    "sum": sum(values),
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0,
                    "mean": sum(values) / len(values) if values else 0
                }
                for name, values in self.histograms.items()
            }
        }
    
    def save_metrics(self):
        """Save metrics to file."""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.export_metrics(), f, indent=2)


# Global instances
health_checker = HealthCheck()
metrics_collector = MetricsCollector()


def get_prometheus_metrics() -> str:
    """Export metrics in Prometheus format."""
    metrics = metrics_collector.export_metrics()
    
    lines = []
    lines.append(f"# HELP fairness_eval_uptime_seconds Application uptime in seconds")
    lines.append(f"# TYPE fairness_eval_uptime_seconds gauge")
    lines.append(f"fairness_eval_uptime_seconds {time.time() - health_checker.start_time}")
    
    # Export counters
    for name, value in metrics["counters"].items():
        lines.append(f"# HELP {name} Counter metric")
        lines.append(f"# TYPE {name} counter")
        lines.append(f"{name} {value}")
    
    # Export gauges
    for name, value in metrics["gauges"].items():
        lines.append(f"# HELP {name} Gauge metric")
        lines.append(f"# TYPE {name} gauge")
        lines.append(f"{name} {value}")
    
    # Export histograms
    for name, hist_data in metrics["histograms"].items():
        lines.append(f"# HELP {name} Histogram metric")
        lines.append(f"# TYPE {name} histogram")
        lines.append(f"{name}_count {hist_data['count']}")
        lines.append(f"{name}_sum {hist_data['sum']}")
    
    return "\n".join(lines)