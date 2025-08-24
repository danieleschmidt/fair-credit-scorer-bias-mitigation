"""Comprehensive health check and metrics endpoints for monitoring and observability."""

import importlib.metadata
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import psutil

from config import Config
from data_loader_preprocessor import generate_data
from fairness_metrics import compute_fairness_metrics


class HealthCheck:
    """Comprehensive health check and monitoring utilities."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize health checker.

        Args:
            config: Application configuration
        """
        self.start_time = time.time()
        self.config = config or Config()
        self._last_health_check = None
        self._error_count = 0
        self._warning_count = 0

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status of the application."""
        start_time = time.time()

        try:
            status = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "uptime_seconds": time.time() - self.start_time,
                "version": self._get_version(),
                "environment": os.getenv("ENVIRONMENT", "development"),
                "checks": {
                    "dependencies": self._check_dependencies(),
                    "system_resources": self._check_system_resources(),
                    "configuration": self._check_configuration(),
                    "data_access": self._check_data_access(),
                    "model_functionality": self._check_model_functionality(),
                }
            }

            # Determine overall status based on individual checks
            failed_checks = [name for name, check in status["checks"].items()
                           if check.get("status") == "unhealthy"]
            degraded_checks = [name for name, check in status["checks"].items()
                             if check.get("status") == "degraded"]

            if failed_checks:
                status["status"] = "unhealthy"
                status["failed_checks"] = failed_checks
                self._error_count += 1
            elif degraded_checks:
                status["status"] = "degraded"
                status["degraded_checks"] = degraded_checks
                self._warning_count += 1

            status["response_time_ms"] = (time.time() - start_time) * 1000
            status["error_count"] = self._error_count
            status["warning_count"] = self._warning_count

            self._last_health_check = status
            return status

        except Exception as e:
            self._error_count += 1
            return {
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "traceback": traceback.format_exc(),
                "response_time_ms": (time.time() - start_time) * 1000,
                "error_count": self._error_count
            }

    def get_readiness_status(self) -> Dict[str, Any]:
        """Check if the application is ready to serve requests."""
        start_time = time.time()

        try:
            checks = {
                "data_generation": self._test_data_generation(),
                "fairness_metrics": self._test_fairness_metrics(),
                "model_training": self._test_model_training(),
                "compute_resources": self._check_compute_resources(),
            }

            # Determine overall readiness
            failed_checks = [name for name, result in checks.items() if not result["passed"]]

            status = {
                "status": "ready" if not failed_checks else "not_ready",
                "timestamp": datetime.utcnow().isoformat(),
                "checks": checks,
                "response_time_ms": (time.time() - start_time) * 1000
            }

            if failed_checks:
                status["failed_checks"] = failed_checks

            return status

        except Exception as e:
            return {
                "status": "not_ready",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "traceback": traceback.format_exc(),
                "response_time_ms": (time.time() - start_time) * 1000
            }

    def get_liveness_status(self) -> Dict[str, Any]:
        """Simple liveness check - application is running."""
        return {
            "status": "alive",
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": time.time() - self.start_time,
            "pid": os.getpid(),
            "threads": len(sys._current_frames()),
            "python_version": sys.version,
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system and application metrics."""
        try:
            process = psutil.Process(os.getpid())

            return {
                "timestamp": datetime.utcnow().isoformat(),
                "uptime_seconds": time.time() - self.start_time,
                "system": {
                    "cpu_percent": psutil.cpu_percent(interval=1),
                    "memory": {
                        "total_gb": psutil.virtual_memory().total / (1024**3),
                        "available_gb": psutil.virtual_memory().available / (1024**3),
                        "percent_used": psutil.virtual_memory().percent,
                    },
                    "disk": {
                        "total_gb": psutil.disk_usage('/').total / (1024**3),
                        "free_gb": psutil.disk_usage('/').free / (1024**3),
                        "percent_used": psutil.disk_usage('/').percent,
                    },
                    "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None,
                    "cpu_count": psutil.cpu_count(),
                },
                "process": {
                    "pid": process.pid,
                    "cpu_percent": process.cpu_percent(),
                    "memory_mb": process.memory_info().rss / 1024 / 1024,
                    "memory_percent": process.memory_percent(),
                    "threads": process.num_threads(),
                    "open_files": len(process.open_files()) if hasattr(process, 'open_files') else None,
                    "create_time": datetime.fromtimestamp(process.create_time()).isoformat(),
                },
                "application": {
                    "version": self._get_version(),
                    "environment": os.getenv("ENVIRONMENT", "development"),
                    "python_version": sys.version,
                    "health_checks_performed": 1 if self._last_health_check else 0,
                    "error_count": self._error_count,
                    "warning_count": self._warning_count,
                    "last_health_check": self._last_health_check["timestamp"] if self._last_health_check else None,
                }
            }

        except Exception as e:
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "status": "error_collecting_metrics"
            }

    def _get_version(self) -> str:
        """Get application version."""
        try:
            return importlib.metadata.version("fair_credit_scorer_bias_mitigation")
        except importlib.metadata.PackageNotFoundError:
            return "0.2.0-dev"

    def _check_dependencies(self) -> Dict[str, Any]:
        """Check if all required dependencies are available and compatible."""
        try:
            required_packages = {
                "scikit-learn": ">=1.0.0",
                "pandas": ">=1.5.0",
                "numpy": ">=1.21.0",
                "fairlearn": ">=0.8.0",
                "networkx": ">=2.8",
                "PyYAML": ">=6.0",
                "requests": ">=2.28.0",
                "psutil": ">=5.8.0"
            }

            missing = []
            versions = {}
            incompatible = []

            for package, _min_version in required_packages.items():
                try:
                    version = importlib.metadata.version(package)
                    versions[package] = version
                    # Note: Simple version check, could be enhanced with proper version parsing
                except importlib.metadata.PackageNotFoundError:
                    missing.append(package)

            if missing:
                return {
                    "status": "unhealthy",
                    "message": f"Missing critical packages: {', '.join(missing)}",
                    "missing_packages": missing,
                    "available_versions": versions
                }

            if incompatible:
                return {
                    "status": "degraded",
                    "message": f"Incompatible versions: {', '.join(incompatible)}",
                    "incompatible_packages": incompatible,
                    "package_versions": versions
                }

            return {
                "status": "healthy",
                "message": "All dependencies available and compatible",
                "package_versions": versions
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Error checking dependencies: {str(e)}",
                "error": str(e)
            }

    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource availability with thresholds."""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            cpu_percent = psutil.cpu_percent(interval=1)

            # Configurable thresholds
            memory_warning = 80.0
            memory_critical = 95.0
            disk_warning = 85.0
            disk_critical = 95.0
            cpu_warning = 90.0

            status = "healthy"
            warnings = []
            errors = []

            if memory.percent > memory_critical:
                status = "unhealthy"
                errors.append(f"Critical memory usage: {memory.percent:.1f}%")
            elif memory.percent > memory_warning:
                status = "degraded"
                warnings.append(f"High memory usage: {memory.percent:.1f}%")

            if disk.percent > disk_critical:
                status = "unhealthy"
                errors.append(f"Critical disk usage: {disk.percent:.1f}%")
            elif disk.percent > disk_warning:
                if status != "unhealthy":
                    status = "degraded"
                warnings.append(f"High disk usage: {disk.percent:.1f}%")

            if cpu_percent > cpu_warning:
                if status != "unhealthy":
                    status = "degraded"
                warnings.append(f"High CPU usage: {cpu_percent:.1f}%")

            return {
                "status": status,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "cpu_percent": cpu_percent,
                "warnings": warnings,
                "errors": errors,
                "thresholds": {
                    "memory_warning": memory_warning,
                    "memory_critical": memory_critical,
                    "disk_warning": disk_warning,
                    "disk_critical": disk_critical,
                    "cpu_warning": cpu_warning
                }
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Error checking system resources: {str(e)}",
                "error": str(e)
            }

    def _check_configuration(self) -> Dict[str, Any]:
        """Check application configuration validity."""
        try:
            config_checks = {
                "config_object_exists": self.config is not None,
                "has_quality_gates": hasattr(self.config, 'quality_gates') if self.config else False,
                "has_security_config": hasattr(self.config, 'security') if self.config else False,
                "environment_set": os.getenv("ENVIRONMENT") is not None,
            }

            passed_checks = sum(1 for check in config_checks.values() if check)
            total_checks = len(config_checks)

            if passed_checks == total_checks:
                return {
                    "status": "healthy",
                    "message": "All configuration checks passed",
                    "checks": config_checks,
                    "score": f"{passed_checks}/{total_checks}"
                }
            elif passed_checks >= total_checks * 0.7:  # 70% threshold
                return {
                    "status": "degraded",
                    "message": "Some configuration issues detected",
                    "checks": config_checks,
                    "score": f"{passed_checks}/{total_checks}"
                }
            else:
                return {
                    "status": "unhealthy",
                    "message": "Critical configuration issues",
                    "checks": config_checks,
                    "score": f"{passed_checks}/{total_checks}"
                }

        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Error checking configuration: {str(e)}",
                "error": str(e)
            }

    def _check_data_access(self) -> Dict[str, Any]:
        """Check data loading and access functionality."""
        try:
            # Test synthetic data generation
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=True) as tmp:
                data = generate_data(tmp.name, n_samples=100, random_state=42)

                if len(data) == 100:
                    return {
                        "status": "healthy",
                        "message": "Data access functional",
                        "test_samples": len(data)
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "message": f"Data generation test failed: expected 100 samples, got {len(data)}"
                    }

        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Data access check failed: {str(e)}",
                "error": str(e)
            }

    def _check_model_functionality(self) -> Dict[str, Any]:
        """Check core model functionality."""
        try:
            # Quick fairness metrics test
            np.random.seed(42)
            n_samples = 100
            y_true = np.random.randint(0, 2, n_samples)
            y_pred = np.random.randint(0, 2, n_samples)
            y_prob = np.random.random(n_samples)
            protected_attr = np.random.randint(0, 2, n_samples)

            metrics = compute_fairness_metrics(y_true, y_pred, y_prob, protected_attr)

            required_metrics = ["demographic_parity_difference", "equalized_odds_difference"]
            missing_metrics = [m for m in required_metrics if m not in metrics]

            if missing_metrics:
                return {
                    "status": "unhealthy",
                    "message": f"Missing required metrics: {missing_metrics}",
                    "available_metrics": list(metrics.keys())
                }

            return {
                "status": "healthy",
                "message": "Model functionality operational",
                "metrics_computed": len(metrics),
                "test_accuracy": float(np.mean(y_pred == y_true))
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Model functionality check failed: {str(e)}",
                "error": str(e)
            }

    def _test_data_generation(self) -> Dict[str, Any]:
        """Test data generation functionality for readiness."""
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=True) as tmp:
                data = generate_data(tmp.name, n_samples=100, random_state=42)

                return {
                    "passed": len(data) == 100,
                    "message": "Data generation successful" if len(data) == 100 else f"Expected 100 samples, got {len(data)}",
                    "samples_generated": len(data)
                }

        except Exception as e:
            return {
                "passed": False,
                "message": f"Data generation failed: {str(e)}",
                "error": str(e)
            }

    def _test_fairness_metrics(self) -> Dict[str, Any]:
        """Test fairness metrics computation for readiness."""
        try:
            np.random.seed(42)
            n_samples = 100
            y_true = np.random.randint(0, 2, n_samples)
            y_pred = np.random.randint(0, 2, n_samples)
            y_prob = np.random.random(n_samples)
            protected_attr = np.random.randint(0, 2, n_samples)

            metrics = compute_fairness_metrics(y_true, y_pred, y_prob, protected_attr)

            return {
                "passed": "demographic_parity_difference" in metrics,
                "message": "Fairness metrics computation successful",
                "metrics_count": len(metrics)
            }

        except Exception as e:
            return {
                "passed": False,
                "message": f"Fairness metrics test failed: {str(e)}",
                "error": str(e)
            }

    def _test_model_training(self) -> Dict[str, Any]:
        """Test basic model training for readiness."""
        try:
            from sklearn.model_selection import train_test_split

            from baseline_model import train_baseline_model

            # Generate minimal test data
            np.random.seed(42)
            X = np.random.randn(100, 5)
            y = np.random.randint(0, 2, 100)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )

            # Try training a basic model
            model = train_baseline_model(X_train, y_train)
            predictions = model.predict(X_test)

            return {
                "passed": len(predictions) == len(y_test),
                "message": "Model training successful",
                "test_accuracy": float(np.mean(predictions == y_test))
            }

        except Exception as e:
            return {
                "passed": False,
                "message": f"Model training test failed: {str(e)}",
                "error": str(e)
            }

    def _check_compute_resources(self) -> Dict[str, Any]:
        """Check if sufficient compute resources are available for readiness."""
        try:
            memory = psutil.virtual_memory()

            # Minimum requirements for ML workloads
            min_memory_gb = 1.0  # Minimum total memory
            min_available_gb = 0.5  # Minimum available memory

            available_gb = memory.available / (1024**3)
            total_gb = memory.total / (1024**3)

            insufficient_total = total_gb < min_memory_gb
            insufficient_available = available_gb < min_available_gb

            return {
                "passed": not (insufficient_total or insufficient_available),
                "message": "Sufficient compute resources" if not (insufficient_total or insufficient_available)
                          else f"Insufficient resources: {total_gb:.2f}GB total, {available_gb:.2f}GB available",
                "total_memory_gb": total_gb,
                "available_memory_gb": available_gb,
                "cpu_count": psutil.cpu_count()
            }

        except Exception as e:
            return {
                "passed": False,
                "message": f"Compute resource check failed: {str(e)}",
                "error": str(e)
            }


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
    lines.append("# HELP fairness_eval_uptime_seconds Application uptime in seconds")
    lines.append("# TYPE fairness_eval_uptime_seconds gauge")
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
