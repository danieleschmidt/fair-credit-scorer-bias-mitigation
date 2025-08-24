"""Application monitoring and instrumentation."""

import functools
import json
import logging
import time
from typing import Any, Callable, Dict

from health_check import metrics_collector


class ApplicationMonitor:
    """Monitor application performance and behavior."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def time_function(self, func_name: str = None):
        """Decorator to time function execution."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                name = func_name or f"{func.__module__}.{func.__name__}"
                start_time = time.time()

                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time

                    # Record metrics
                    metrics_collector.record_histogram(
                        "function_duration_seconds",
                        execution_time,
                        {"function": name}
                    )
                    metrics_collector.increment_counter(
                        "function_calls_total",
                        labels={"function": name, "status": "success"}
                    )

                    self.logger.debug(f"Function {name} executed in {execution_time:.4f}s")
                    return result

                except Exception as e:
                    execution_time = time.time() - start_time

                    # Record error metrics
                    metrics_collector.increment_counter(
                        "function_calls_total",
                        labels={"function": name, "status": "error"}
                    )
                    metrics_collector.record_histogram(
                        "function_duration_seconds",
                        execution_time,
                        {"function": name}
                    )

                    self.logger.error(f"Function {name} failed after {execution_time:.4f}s: {str(e)}")
                    raise

            return wrapper
        return decorator

    def track_model_performance(self, method: str, metrics: Dict[str, float]):
        """Track model performance metrics."""
        # Record accuracy
        if "accuracy" in metrics:
            metrics_collector.set_gauge(
                "model_accuracy",
                metrics["accuracy"],
                {"method": method}
            )

        # Record fairness metrics
        if "demographic_parity_difference" in metrics:
            metrics_collector.set_gauge(
                "model_demographic_parity_difference",
                metrics["demographic_parity_difference"],
                {"method": method}
            )

        if "equalized_odds_difference" in metrics:
            metrics_collector.set_gauge(
                "model_equalized_odds_difference",
                metrics["equalized_odds_difference"],
                {"method": method}
            )

        # Increment model evaluation counter
        metrics_collector.increment_counter(
            "model_evaluations_total",
            labels={"method": method}
        )

        self.logger.info(f"Model performance recorded for method: {method}")

    def log_structured(self, level: str, message: str, **kwargs):
        """Log structured data."""
        log_data = {
            "timestamp": time.time(),
            "level": level.upper(),
            "message": message,
            **kwargs
        }

        if level.lower() == "debug":
            self.logger.debug(json.dumps(log_data))
        elif level.lower() == "info":
            self.logger.info(json.dumps(log_data))
        elif level.lower() == "warning":
            self.logger.warning(json.dumps(log_data))
        elif level.lower() == "error":
            self.logger.error(json.dumps(log_data))
        elif level.lower() == "critical":
            self.logger.critical(json.dumps(log_data))


class AlertManager:
    """Manage application alerts and notifications."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.alert_thresholds = {
            "accuracy": 0.7,  # Alert if accuracy drops below 70%
            "demographic_parity_difference": 0.3,  # Alert if bias exceeds 30%
            "memory_usage_percent": 90,  # Alert if memory usage exceeds 90%
            "error_rate": 0.1  # Alert if error rate exceeds 10%
        }

    def check_model_performance_alerts(self, method: str, metrics: Dict[str, float]):
        """Check for model performance alerts."""
        alerts = []

        # Check accuracy threshold
        if "accuracy" in metrics and metrics["accuracy"] < self.alert_thresholds["accuracy"]:
            alerts.append({
                "type": "model_performance",
                "severity": "warning",
                "message": f"Model accuracy ({metrics['accuracy']:.3f}) below threshold ({self.alert_thresholds['accuracy']})",
                "method": method,
                "metric": "accuracy",
                "value": metrics["accuracy"],
                "threshold": self.alert_thresholds["accuracy"]
            })

        # Check bias threshold
        if "demographic_parity_difference" in metrics:
            bias = abs(metrics["demographic_parity_difference"])
            if bias > self.alert_thresholds["demographic_parity_difference"]:
                alerts.append({
                    "type": "model_bias",
                    "severity": "critical",
                    "message": f"Model bias ({bias:.3f}) exceeds threshold ({self.alert_thresholds['demographic_parity_difference']})",
                    "method": method,
                    "metric": "demographic_parity_difference",
                    "value": bias,
                    "threshold": self.alert_thresholds["demographic_parity_difference"]
                })

        # Log alerts
        for alert in alerts:
            self.logger.warning(f"ALERT: {alert['message']}", extra=alert)
            metrics_collector.increment_counter(
                "alerts_total",
                labels={"type": alert["type"], "severity": alert["severity"]}
            )

        return alerts

    def check_system_alerts(self, system_metrics: Dict[str, Any]):
        """Check for system-level alerts."""
        alerts = []

        # Check memory usage
        if "process" in system_metrics and "memory_percent" in system_metrics["process"]:
            memory_percent = system_metrics["process"]["memory_percent"]
            if memory_percent > self.alert_thresholds["memory_usage_percent"]:
                alerts.append({
                    "type": "system_resource",
                    "severity": "warning",
                    "message": f"High memory usage ({memory_percent:.1f}%)",
                    "metric": "memory_percent",
                    "value": memory_percent,
                    "threshold": self.alert_thresholds["memory_usage_percent"]
                })

        # Log alerts
        for alert in alerts:
            self.logger.warning(f"SYSTEM ALERT: {alert['message']}", extra=alert)
            metrics_collector.increment_counter(
                "system_alerts_total",
                labels={"type": alert["type"], "severity": alert["severity"]}
            )

        return alerts


# Global instances
monitor = ApplicationMonitor()
alert_manager = AlertManager()


def instrument_function(func_name: str = None):
    """Decorator to instrument functions with monitoring."""
    return monitor.time_function(func_name)


def track_performance(method: str, metrics: Dict[str, float]):
    """Track model performance and check for alerts."""
    monitor.track_model_performance(method, metrics)
    return alert_manager.check_model_performance_alerts(method, metrics)
