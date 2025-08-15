"""
Performance metrics collection and analysis system.

Comprehensive metrics collection for ML models, data pipelines,
and system performance with real-time monitoring and alerting.
"""

import asyncio
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Union

import psutil

from ..logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: Union[float, int]
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'tags': self.tags
        }


@dataclass
class MetricSummary:
    """Summary statistics for a metric."""
    name: str
    count: int
    min_value: float
    max_value: float
    mean: float
    median: float
    std_dev: float
    p95: float
    p99: float
    rate_per_second: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'count': self.count,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'mean': self.mean,
            'median': self.median,
            'std_dev': self.std_dev,
            'p95': self.p95,
            'p99': self.p99,
            'rate_per_second': self.rate_per_second
        }


@dataclass
class Alert:
    """Performance alert."""
    metric_name: str
    alert_type: str
    threshold: float
    current_value: float
    severity: str
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'metric_name': self.metric_name,
            'alert_type': self.alert_type,
            'threshold': self.threshold,
            'current_value': self.current_value,
            'severity': self.severity,
            'message': self.message,
            'timestamp': self.timestamp.isoformat()
        }


class MetricsCollector:
    """
    High-performance metrics collection system.
    
    Collects, aggregates, and analyzes performance metrics
    with configurable retention and alerting.
    """

    def __init__(
        self,
        retention_hours: int = 24,
        max_points_per_metric: int = 10000,
        aggregation_interval_seconds: int = 60
    ):
        """
        Initialize metrics collector.
        
        Args:
            retention_hours: How long to retain metric data
            max_points_per_metric: Maximum data points per metric
            aggregation_interval_seconds: Interval for aggregation
        """
        self.retention_hours = retention_hours
        self.max_points_per_metric = max_points_per_metric
        self.aggregation_interval_seconds = aggregation_interval_seconds

        # Metric storage
        self.metrics: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_points_per_metric)
        )

        # Alert configuration
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.active_alerts: List[Alert] = []

        # Aggregated metrics
        self.aggregated_metrics: Dict[str, List[MetricPoint]] = defaultdict(list)

        # Collection state
        self.is_collecting = False
        self.collection_start_time: Optional[datetime] = None

        logger.info("MetricsCollector initialized")

    def record_metric(
        self,
        name: str,
        value: Union[float, int],
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None
    ):
        """
        Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for categorization
            timestamp: Optional timestamp (defaults to now)
        """
        timestamp = timestamp or datetime.utcnow()
        tags = tags or {}

        metric_point = MetricPoint(
            name=name,
            value=value,
            timestamp=timestamp,
            tags=tags
        )

        self.metrics[name].append(metric_point)

        # Check for alerts
        self._check_alerts(name, value)

        # Cleanup old data
        self._cleanup_old_data(name)

    def record_timing(self, name: str, duration_ms: float, tags: Optional[Dict[str, str]] = None):
        """Record timing metric."""
        self.record_metric(f"{name}.duration_ms", duration_ms, tags)

    def record_counter(self, name: str, increment: int = 1, tags: Optional[Dict[str, str]] = None):
        """Record counter metric."""
        self.record_metric(f"{name}.count", increment, tags)

    def record_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record gauge metric."""
        self.record_metric(f"{name}.gauge", value, tags)

    def start_timer(self, name: str) -> Callable:
        """
        Start a timer and return a function to stop it.
        
        Args:
            name: Timer name
            
        Returns:
            Function to call to record the timing
        """
        start_time = time.perf_counter()

        def stop_timer(tags: Optional[Dict[str, str]] = None):
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.record_timing(name, duration_ms, tags)
            return duration_ms

        return stop_timer

    def get_metric_summary(self, name: str, hours: int = 1) -> Optional[MetricSummary]:
        """
        Get summary statistics for a metric.
        
        Args:
            name: Metric name
            hours: Hours of data to analyze
            
        Returns:
            Metric summary or None if no data
        """
        if name not in self.metrics:
            return None

        # Filter data by time window
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_points = [
            point for point in self.metrics[name]
            if point.timestamp >= cutoff_time
        ]

        if not recent_points:
            return None

        values = [point.value for point in recent_points]

        # Calculate statistics
        count = len(values)
        min_value = min(values)
        max_value = max(values)
        mean = statistics.mean(values)
        median = statistics.median(values)
        std_dev = statistics.stdev(values) if count > 1 else 0

        # Calculate percentiles
        sorted_values = sorted(values)
        p95_idx = int(count * 0.95)
        p99_idx = int(count * 0.99)
        p95 = sorted_values[min(p95_idx, count - 1)]
        p99 = sorted_values[min(p99_idx, count - 1)]

        # Calculate rate
        time_span_hours = hours
        rate_per_second = count / (time_span_hours * 3600) if time_span_hours > 0 else 0

        return MetricSummary(
            name=name,
            count=count,
            min_value=min_value,
            max_value=max_value,
            mean=mean,
            median=median,
            std_dev=std_dev,
            p95=p95,
            p99=p99,
            rate_per_second=rate_per_second
        )

    def get_all_metrics_summary(self, hours: int = 1) -> Dict[str, MetricSummary]:
        """Get summary for all metrics."""
        summaries = {}

        for metric_name in self.metrics.keys():
            summary = self.get_metric_summary(metric_name, hours)
            if summary:
                summaries[metric_name] = summary

        return summaries

    def set_alert_rule(
        self,
        metric_name: str,
        alert_type: str,
        threshold: float,
        severity: str = "warning",
        window_minutes: int = 5
    ):
        """
        Set an alert rule for a metric.
        
        Args:
            metric_name: Name of metric to monitor
            alert_type: Type of alert (threshold, rate, etc.)
            threshold: Alert threshold value
            severity: Alert severity level
            window_minutes: Time window for evaluation
        """
        self.alert_rules[metric_name] = {
            'alert_type': alert_type,
            'threshold': threshold,
            'severity': severity,
            'window_minutes': window_minutes
        }

        logger.info(f"Alert rule set for {metric_name}: {alert_type} > {threshold}")

    def _check_alerts(self, metric_name: str, value: float):
        """Check if metric value triggers any alerts."""
        if metric_name not in self.alert_rules:
            return

        rule = self.alert_rules[metric_name]
        alert_type = rule['alert_type']
        threshold = rule['threshold']
        severity = rule['severity']

        triggered = False
        message = ""

        if alert_type == "threshold_high" and value > threshold:
            triggered = True
            message = f"{metric_name} exceeded threshold: {value} > {threshold}"
        elif alert_type == "threshold_low" and value < threshold:
            triggered = True
            message = f"{metric_name} below threshold: {value} < {threshold}"
        elif alert_type == "rate_high":
            # Check rate over window
            summary = self.get_metric_summary(metric_name, hours=rule['window_minutes']/60)
            if summary and summary.rate_per_second > threshold:
                triggered = True
                message = f"{metric_name} rate exceeded: {summary.rate_per_second:.2f}/s > {threshold}/s"

        if triggered:
            alert = Alert(
                metric_name=metric_name,
                alert_type=alert_type,
                threshold=threshold,
                current_value=value,
                severity=severity,
                message=message
            )

            self.active_alerts.append(alert)
            logger.warning(f"Alert triggered: {message}")

    def _cleanup_old_data(self, metric_name: str):
        """Remove old metric data beyond retention period."""
        if metric_name not in self.metrics:
            return

        cutoff_time = datetime.utcnow() - timedelta(hours=self.retention_hours)

        # Filter out old points
        points = self.metrics[metric_name]
        while points and points[0].timestamp < cutoff_time:
            points.popleft()

    def get_recent_alerts(self, hours: int = 1) -> List[Alert]:
        """Get recent alerts."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [
            alert for alert in self.active_alerts
            if alert.timestamp >= cutoff_time
        ]

    def export_metrics(self, format: str = "json") -> str:
        """Export metrics data."""
        if format == "json":
            import json

            export_data = {
                'metrics': {},
                'summaries': {},
                'alerts': [alert.to_dict() for alert in self.active_alerts[-100:]],  # Last 100 alerts
                'export_timestamp': datetime.utcnow().isoformat()
            }

            # Export recent data for each metric
            for metric_name in self.metrics.keys():
                recent_points = list(self.metrics[metric_name])[-1000:]  # Last 1000 points
                export_data['metrics'][metric_name] = [point.to_dict() for point in recent_points]

                summary = self.get_metric_summary(metric_name)
                if summary:
                    export_data['summaries'][metric_name] = summary.to_dict()

            return json.dumps(export_data, indent=2)

        else:
            raise ValueError(f"Unsupported export format: {format}")


class SystemMetricsCollector:
    """
    System-level metrics collector for infrastructure monitoring.
    
    Automatically collects CPU, memory, disk, and network metrics.
    """

    def __init__(self, metrics_collector: MetricsCollector):
        """
        Initialize system metrics collector.
        
        Args:
            metrics_collector: Main metrics collector instance
        """
        self.metrics_collector = metrics_collector
        self.is_collecting = False
        self.collection_interval = 5  # seconds

        logger.info("SystemMetricsCollector initialized")

    async def start_collection(self):
        """Start automatic system metrics collection."""
        self.is_collecting = True
        logger.info("Starting system metrics collection")

        while self.is_collecting:
            try:
                self._collect_system_metrics()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"System metrics collection error: {e}")
                await asyncio.sleep(self.collection_interval)

    def stop_collection(self):
        """Stop system metrics collection."""
        self.is_collecting = False
        logger.info("System metrics collection stopped")

    def _collect_system_metrics(self):
        """Collect current system metrics."""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        self.metrics_collector.record_gauge("system.cpu.percent", cpu_percent)

        # Memory metrics
        memory = psutil.virtual_memory()
        self.metrics_collector.record_gauge("system.memory.percent", memory.percent)
        self.metrics_collector.record_gauge("system.memory.available_mb", memory.available / 1024 / 1024)
        self.metrics_collector.record_gauge("system.memory.used_mb", memory.used / 1024 / 1024)

        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        self.metrics_collector.record_gauge("system.disk.percent", disk_percent)
        self.metrics_collector.record_gauge("system.disk.free_gb", disk.free / 1024 / 1024 / 1024)

        # Network metrics (if available)
        try:
            network = psutil.net_io_counters()
            self.metrics_collector.record_counter("system.network.bytes_sent", network.bytes_sent)
            self.metrics_collector.record_counter("system.network.bytes_recv", network.bytes_recv)
        except Exception as e:
            logger.debug(f"Network metrics not available: {e}")

        # Process-specific metrics
        try:
            process = psutil.Process()
            process_memory = process.memory_info()
            self.metrics_collector.record_gauge("process.memory.rss_mb", process_memory.rss / 1024 / 1024)
            self.metrics_collector.record_gauge("process.cpu.percent", process.cpu_percent())
        except Exception as e:
            logger.debug(f"Process metrics not available: {e}")


class MLMetricsCollector:
    """
    ML-specific metrics collector for model performance monitoring.
    
    Collects model prediction metrics, accuracy, latency, and bias indicators.
    """

    def __init__(self, metrics_collector: MetricsCollector):
        """
        Initialize ML metrics collector.
        
        Args:
            metrics_collector: Main metrics collector instance
        """
        self.metrics_collector = metrics_collector

        # ML-specific metric tracking
        self.prediction_history = deque(maxlen=10000)
        self.accuracy_window = deque(maxlen=1000)

        logger.info("MLMetricsCollector initialized")

    def record_prediction(
        self,
        model_name: str,
        prediction_time_ms: float,
        confidence: Optional[float] = None,
        features: Optional[Dict[str, Any]] = None
    ):
        """
        Record a model prediction.
        
        Args:
            model_name: Name of the model
            prediction_time_ms: Prediction latency in milliseconds
            confidence: Prediction confidence score
            features: Input features (for drift detection)
        """
        tags = {"model": model_name}

        # Record timing
        self.metrics_collector.record_timing("ml.prediction.latency", prediction_time_ms, tags)

        # Record confidence if available
        if confidence is not None:
            self.metrics_collector.record_gauge("ml.prediction.confidence", confidence, tags)

        # Count predictions
        self.metrics_collector.record_counter("ml.predictions.total", 1, tags)

        # Store for drift analysis
        self.prediction_history.append({
            'model': model_name,
            'timestamp': datetime.utcnow(),
            'latency_ms': prediction_time_ms,
            'confidence': confidence,
            'features': features
        })

    def record_model_accuracy(
        self,
        model_name: str,
        accuracy: float,
        metric_type: str = "accuracy"
    ):
        """
        Record model accuracy metrics.
        
        Args:
            model_name: Name of the model
            accuracy: Accuracy value (0-1)
            metric_type: Type of accuracy metric
        """
        tags = {"model": model_name, "metric_type": metric_type}

        self.metrics_collector.record_gauge(f"ml.model.{metric_type}", accuracy, tags)

        # Track accuracy trend
        self.accuracy_window.append({
            'model': model_name,
            'accuracy': accuracy,
            'timestamp': datetime.utcnow()
        })

    def record_bias_metric(
        self,
        model_name: str,
        bias_type: str,
        bias_value: float,
        protected_attribute: str
    ):
        """
        Record bias detection metrics.
        
        Args:
            model_name: Name of the model
            bias_type: Type of bias (demographic_parity, equalized_odds, etc.)
            bias_value: Bias metric value
            protected_attribute: Protected attribute name
        """
        tags = {
            "model": model_name,
            "bias_type": bias_type,
            "protected_attribute": protected_attribute
        }

        self.metrics_collector.record_gauge("ml.bias.metric", bias_value, tags)

        # Alert on high bias
        if bias_value > 0.1:  # 10% bias threshold
            logger.warning(f"High bias detected: {bias_type}={bias_value:.3f} for {protected_attribute}")

    def record_data_drift(
        self,
        feature_name: str,
        drift_score: float,
        drift_type: str = "statistical"
    ):
        """
        Record data drift metrics.
        
        Args:
            feature_name: Name of the feature
            drift_score: Drift score (0-1, higher means more drift)
            drift_type: Type of drift detection used
        """
        tags = {"feature": feature_name, "drift_type": drift_type}

        self.metrics_collector.record_gauge("ml.drift.score", drift_score, tags)

        # Alert on significant drift
        if drift_score > 0.3:  # 30% drift threshold
            logger.warning(f"Data drift detected: {feature_name} drift_score={drift_score:.3f}")

    def get_model_performance_summary(self, model_name: str, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for a specific model."""
        # Get prediction metrics
        latency_summary = self.metrics_collector.get_metric_summary(
            "ml.prediction.latency.duration_ms", hours
        )

        confidence_summary = self.metrics_collector.get_metric_summary(
            "ml.prediction.confidence.gauge", hours
        )

        prediction_count = self.metrics_collector.get_metric_summary(
            "ml.predictions.total.count", hours
        )

        # Calculate recent accuracy
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_accuracy = [
            record['accuracy'] for record in self.accuracy_window
            if record['model'] == model_name and record['timestamp'] >= cutoff_time
        ]

        summary = {
            'model_name': model_name,
            'time_window_hours': hours,
            'prediction_count': prediction_count.count if prediction_count else 0,
            'average_latency_ms': latency_summary.mean if latency_summary else 0,
            'p95_latency_ms': latency_summary.p95 if latency_summary else 0,
            'average_confidence': confidence_summary.mean if confidence_summary else 0,
            'recent_accuracy': statistics.mean(recent_accuracy) if recent_accuracy else None,
            'accuracy_trend': self._calculate_accuracy_trend(model_name, hours)
        }

        return summary

    def _calculate_accuracy_trend(self, model_name: str, hours: int) -> str:
        """Calculate accuracy trend (improving, stable, declining)."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_accuracy = [
            record for record in self.accuracy_window
            if record['model'] == model_name and record['timestamp'] >= cutoff_time
        ]

        if len(recent_accuracy) < 2:
            return "insufficient_data"

        # Simple trend calculation
        first_half = recent_accuracy[:len(recent_accuracy)//2]
        second_half = recent_accuracy[len(recent_accuracy)//2:]

        first_avg = statistics.mean([r['accuracy'] for r in first_half])
        second_avg = statistics.mean([r['accuracy'] for r in second_half])

        if second_avg > first_avg + 0.02:
            return "improving"
        elif second_avg < first_avg - 0.02:
            return "declining"
        else:
            return "stable"


# CLI interface
def main():
    """CLI interface for metrics collection."""
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Metrics Collection CLI")
    parser.add_argument("command", choices=["collect", "export", "demo"])
    parser.add_argument("--duration", type=int, default=60, help="Collection duration (seconds)")
    parser.add_argument("--output", help="Output file path")

    args = parser.parse_args()

    # Initialize metrics collector
    collector = MetricsCollector()
    system_collector = SystemMetricsCollector(collector)
    ml_collector = MLMetricsCollector(collector)

    if args.command == "collect":
        # Collect system metrics
        async def collect_metrics():
            await system_collector.start_collection()

        try:
            asyncio.run(asyncio.wait_for(collect_metrics(), timeout=args.duration))
        except asyncio.TimeoutError:
            system_collector.stop_collection()

        # Show summary
        summaries = collector.get_all_metrics_summary()
        print("Metrics Summary:")
        for name, summary in summaries.items():
            print(f"  {name}: mean={summary.mean:.2f}, p95={summary.p95:.2f}")

    elif args.command == "export":
        # Export metrics data
        if not args.output:
            args.output = "metrics_export.json"

        exported_data = collector.export_metrics()

        with open(args.output, 'w') as f:
            f.write(exported_data)

        print(f"Metrics exported to {args.output}")

    elif args.command == "demo":
        # Demo ML metrics
        import random

        print("Recording demo ML metrics...")

        # Simulate predictions
        for i in range(100):
            ml_collector.record_prediction(
                model_name="demo_model",
                prediction_time_ms=random.uniform(10, 100),
                confidence=random.uniform(0.5, 0.95)
            )

            if i % 20 == 0:
                ml_collector.record_model_accuracy(
                    model_name="demo_model",
                    accuracy=random.uniform(0.8, 0.95)
                )

        # Show ML summary
        summary = ml_collector.get_model_performance_summary("demo_model")
        print("ML Performance Summary:")
        print(f"  Predictions: {summary['prediction_count']}")
        print(f"  Average latency: {summary['average_latency_ms']:.1f}ms")
        print(f"  Average confidence: {summary['average_confidence']:.3f}")


if __name__ == "__main__":
    main()
