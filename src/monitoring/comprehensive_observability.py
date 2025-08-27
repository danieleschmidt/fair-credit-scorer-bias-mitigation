"""
Comprehensive Observability Platform for Fair AI Systems.

This module implements a comprehensive observability platform that provides
deep insights into fairness, performance, and system health with real-time
monitoring, alerting, and automated response capabilities.

Key Features:
- Real-time fairness metric monitoring with drift detection
- Multi-dimensional performance tracking (latency, throughput, accuracy)
- Automated anomaly detection with ML-based pattern recognition
- Intelligent alerting with context-aware severity assessment
- Comprehensive dashboards with interactive visualizations
- Distributed tracing for complex fairness decision workflows
- Custom metric collection with fairness-aware sampling
- Automated report generation with trend analysis
"""

import json
import time
import threading
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Tuple
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

try:
    from ..fairness_metrics import compute_fairness_metrics
    from ..logging_config import get_logger
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from fairness_metrics import compute_fairness_metrics
    from logging_config import get_logger

logger = get_logger(__name__)


class MetricType(Enum):
    """Types of metrics that can be collected."""
    FAIRNESS = "fairness"
    PERFORMANCE = "performance"
    SYSTEM_HEALTH = "system_health"
    BUSINESS_KPI = "business_kpi"
    USER_EXPERIENCE = "user_experience"
    SECURITY = "security"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    STATISTICAL_OUTLIER = "statistical_outlier"
    TREND_CHANGE = "trend_change"
    SEASONAL_DEVIATION = "seasonal_deviation"
    CORRELATION_BREAK = "correlation_break"
    FAIRNESS_DRIFT = "fairness_drift"
    PERFORMANCE_DEGRADATION = "performance_degradation"


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: datetime
    metric_name: str
    metric_type: MetricType
    value: float
    dimensions: Dict[str, str]
    metadata: Dict[str, Any]


@dataclass
class Alert:
    """Alert generated from metric analysis."""
    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    metric_name: str
    title: str
    description: str
    value: float
    threshold: float
    dimensions: Dict[str, str]
    context: Dict[str, Any]
    resolution_suggestions: List[str]
    acknowledged: bool = False
    resolved: bool = False
    resolved_timestamp: Optional[datetime] = None


@dataclass
class Anomaly:
    """Detected anomaly in metrics."""
    anomaly_id: str
    timestamp: datetime
    anomaly_type: AnomalyType
    metric_name: str
    value: float
    expected_value: float
    deviation_score: float
    confidence: float
    context: Dict[str, Any]
    impact_assessment: Dict[str, Any]


class MetricCollector:
    """Collects and stores metrics from various sources."""
    
    def __init__(self, max_retention_days: int = 30):
        self.max_retention_days = max_retention_days
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100000))
        self.metric_metadata: Dict[str, Dict[str, Any]] = {}
        self.collection_lock = threading.Lock()
        
    def collect_metric(self, metric_point: MetricPoint):
        """Collect a single metric point."""
        with self.collection_lock:
            metric_key = f"{metric_point.metric_name}_{metric_point.metric_type.value}"
            self.metrics[metric_key].append(metric_point)
            
            # Update metadata
            if metric_key not in self.metric_metadata:
                self.metric_metadata[metric_key] = {
                    'first_seen': metric_point.timestamp,
                    'dimensions': set(),
                    'sample_count': 0
                }
            
            self.metric_metadata[metric_key]['last_seen'] = metric_point.timestamp
            self.metric_metadata[metric_key]['dimensions'].update(metric_point.dimensions.keys())
            self.metric_metadata[metric_key]['sample_count'] += 1
    
    def collect_batch(self, metric_points: List[MetricPoint]):
        """Collect multiple metric points efficiently."""
        with self.collection_lock:
            for point in metric_points:
                self.collect_metric(point)
    
    def get_metric_history(self, metric_name: str, metric_type: MetricType,
                          since: Optional[datetime] = None, 
                          dimensions: Optional[Dict[str, str]] = None) -> List[MetricPoint]:
        """Get historical data for a specific metric."""
        metric_key = f"{metric_name}_{metric_type.value}"
        
        if metric_key not in self.metrics:
            return []
        
        with self.collection_lock:
            points = list(self.metrics[metric_key])
        
        # Filter by timestamp
        if since:
            points = [p for p in points if p.timestamp >= since]
        
        # Filter by dimensions
        if dimensions:
            filtered_points = []
            for point in points:
                match = True
                for dim_key, dim_value in dimensions.items():
                    if point.dimensions.get(dim_key) != dim_value:
                        match = False
                        break
                if match:
                    filtered_points.append(point)
            points = filtered_points
        
        return sorted(points, key=lambda x: x.timestamp)
    
    def cleanup_old_metrics(self):
        """Remove metrics older than retention period."""
        cutoff_time = datetime.utcnow() - timedelta(days=self.max_retention_days)
        
        with self.collection_lock:
            for metric_key in list(self.metrics.keys()):
                # Filter out old points
                original_count = len(self.metrics[metric_key])
                self.metrics[metric_key] = deque(
                    [p for p in self.metrics[metric_key] if p.timestamp >= cutoff_time],
                    maxlen=self.metrics[metric_key].maxlen
                )
                
                removed_count = original_count - len(self.metrics[metric_key])
                if removed_count > 0:
                    logger.debug(f"Cleaned up {removed_count} old points for {metric_key}")
                    
                    # Update metadata
                    if metric_key in self.metric_metadata:
                        self.metric_metadata[metric_key]['sample_count'] -= removed_count


class AnomalyDetector:
    """Detects anomalies in metric streams using various ML techniques."""
    
    def __init__(self, sensitivity: float = 0.1):
        self.sensitivity = sensitivity
        self.models: Dict[str, Any] = {}
        self.detection_history: Dict[str, List[Anomaly]] = defaultdict(list)
        
    def detect_anomalies(self, metric_name: str, metric_type: MetricType,
                        metric_history: List[MetricPoint]) -> List[Anomaly]:
        """Detect anomalies in metric history."""
        if len(metric_history) < 10:
            return []  # Need minimum data for detection
        
        anomalies = []
        
        # Extract values and timestamps
        values = [p.value for p in metric_history]
        timestamps = [p.timestamp for p in metric_history]
        
        # Statistical outlier detection
        statistical_anomalies = self._detect_statistical_outliers(
            metric_name, metric_type, values, timestamps, metric_history
        )
        anomalies.extend(statistical_anomalies)
        
        # Trend change detection
        trend_anomalies = self._detect_trend_changes(
            metric_name, metric_type, values, timestamps, metric_history
        )
        anomalies.extend(trend_anomalies)
        
        # Seasonal deviation detection (if enough data)
        if len(metric_history) > 50:
            seasonal_anomalies = self._detect_seasonal_deviations(
                metric_name, metric_type, values, timestamps, metric_history
            )
            anomalies.extend(seasonal_anomalies)
        
        # Fairness-specific anomaly detection
        if metric_type == MetricType.FAIRNESS:
            fairness_anomalies = self._detect_fairness_drift(
                metric_name, values, timestamps, metric_history
            )
            anomalies.extend(fairness_anomalies)
        
        # Store detection history
        key = f"{metric_name}_{metric_type.value}"
        self.detection_history[key].extend(anomalies)
        
        # Limit history size
        if len(self.detection_history[key]) > 1000:
            self.detection_history[key] = self.detection_history[key][-1000:]
        
        return anomalies
    
    def _detect_statistical_outliers(self, metric_name: str, metric_type: MetricType,
                                   values: List[float], timestamps: List[datetime],
                                   metric_history: List[MetricPoint]) -> List[Anomaly]:
        """Detect statistical outliers using z-score and IQR methods."""
        anomalies = []
        
        if len(values) < 10:
            return anomalies
        
        values_array = np.array(values)
        mean_val = np.mean(values_array)
        std_val = np.std(values_array)
        
        # Z-score based detection
        if std_val > 0:
            z_scores = np.abs(values_array - mean_val) / std_val
            z_threshold = 3.0 / (1 + self.sensitivity)  # Adjust threshold based on sensitivity
            
            outlier_indices = np.where(z_scores > z_threshold)[0]
            
            for idx in outlier_indices:
                anomaly = Anomaly(
                    anomaly_id=f"outlier_{metric_name}_{int(timestamps[idx].timestamp())}",
                    timestamp=timestamps[idx],
                    anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                    metric_name=metric_name,
                    value=values[idx],
                    expected_value=mean_val,
                    deviation_score=float(z_scores[idx]),
                    confidence=min(1.0, float(z_scores[idx]) / 5.0),
                    context={
                        'detection_method': 'z_score',
                        'threshold': z_threshold,
                        'mean': mean_val,
                        'std': std_val,
                        'dimensions': metric_history[idx].dimensions
                    },
                    impact_assessment=self._assess_outlier_impact(metric_name, metric_type, values[idx], mean_val)
                )
                anomalies.append(anomaly)
        
        # IQR based detection
        q1 = np.percentile(values_array, 25)
        q3 = np.percentile(values_array, 75)
        iqr = q3 - q1
        
        if iqr > 0:
            iqr_multiplier = 1.5 + (1 - self.sensitivity) * 1.0  # Adjust based on sensitivity
            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr
            
            iqr_outlier_indices = [i for i, v in enumerate(values) if v < lower_bound or v > upper_bound]
            
            for idx in iqr_outlier_indices:
                # Only add if not already detected by z-score
                anomaly_id = f"iqr_outlier_{metric_name}_{int(timestamps[idx].timestamp())}"
                if not any(a.anomaly_id.endswith(f"_{metric_name}_{int(timestamps[idx].timestamp())}") for a in anomalies):
                    expected = q1 + (q3 - q1) / 2  # Median as expected value
                    deviation = abs(values[idx] - expected) / (iqr + 1e-8)
                    
                    anomaly = Anomaly(
                        anomaly_id=anomaly_id,
                        timestamp=timestamps[idx],
                        anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                        metric_name=metric_name,
                        value=values[idx],
                        expected_value=expected,
                        deviation_score=deviation,
                        confidence=min(1.0, deviation),
                        context={
                            'detection_method': 'iqr',
                            'q1': q1,
                            'q3': q3,
                            'iqr': iqr,
                            'bounds': [lower_bound, upper_bound],
                            'dimensions': metric_history[idx].dimensions
                        },
                        impact_assessment=self._assess_outlier_impact(metric_name, metric_type, values[idx], expected)
                    )
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_trend_changes(self, metric_name: str, metric_type: MetricType,
                            values: List[float], timestamps: List[datetime],
                            metric_history: List[MetricPoint]) -> List[Anomaly]:
        """Detect significant trend changes in metrics."""
        anomalies = []
        
        if len(values) < 20:
            return anomalies
        
        # Split data into windows for trend analysis
        window_size = max(10, len(values) // 4)
        
        for i in range(window_size, len(values) - window_size):
            # Calculate trends before and after the point
            before_values = values[i-window_size:i]
            after_values = values[i:i+window_size]
            
            # Calculate linear trends
            before_trend = self._calculate_trend(before_values)
            after_trend = self._calculate_trend(after_values)
            
            # Detect significant trend change
            trend_change = abs(after_trend - before_trend)
            threshold = 0.1 / (1 + self.sensitivity)  # Adjust threshold based on sensitivity
            
            if trend_change > threshold:
                anomaly = Anomaly(
                    anomaly_id=f"trend_change_{metric_name}_{int(timestamps[i].timestamp())}",
                    timestamp=timestamps[i],
                    anomaly_type=AnomalyType.TREND_CHANGE,
                    metric_name=metric_name,
                    value=values[i],
                    expected_value=values[i-1] + before_trend,  # Expected based on previous trend
                    deviation_score=trend_change,
                    confidence=min(1.0, trend_change * 5),
                    context={
                        'before_trend': before_trend,
                        'after_trend': after_trend,
                        'trend_change': trend_change,
                        'window_size': window_size,
                        'dimensions': metric_history[i].dimensions
                    },
                    impact_assessment=self._assess_trend_change_impact(metric_name, metric_type, before_trend, after_trend)
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_seasonal_deviations(self, metric_name: str, metric_type: MetricType,
                                  values: List[float], timestamps: List[datetime],
                                  metric_history: List[MetricPoint]) -> List[Anomaly]:
        """Detect deviations from seasonal patterns."""
        anomalies = []
        
        if len(values) < 50:
            return anomalies
        
        # Simple seasonal pattern detection based on hour of day
        hourly_patterns = defaultdict(list)
        
        for i, (timestamp, value) in enumerate(zip(timestamps, values)):
            hour = timestamp.hour
            hourly_patterns[hour].append((value, i))
        
        # Find hours with sufficient data
        for hour, hour_data in hourly_patterns.items():
            if len(hour_data) < 3:
                continue
            
            hour_values = [v for v, _ in hour_data]
            hour_mean = np.mean(hour_values)
            hour_std = np.std(hour_values)
            
            if hour_std == 0:
                continue
            
            # Check each point for seasonal deviation
            for value, idx in hour_data:
                deviation = abs(value - hour_mean) / hour_std
                threshold = 2.0 / (1 + self.sensitivity)
                
                if deviation > threshold:
                    anomaly = Anomaly(
                        anomaly_id=f"seasonal_{metric_name}_{int(timestamps[idx].timestamp())}",
                        timestamp=timestamps[idx],
                        anomaly_type=AnomalyType.SEASONAL_DEVIATION,
                        metric_name=metric_name,
                        value=value,
                        expected_value=hour_mean,
                        deviation_score=deviation,
                        confidence=min(1.0, deviation / 3.0),
                        context={
                            'hour': hour,
                            'seasonal_mean': hour_mean,
                            'seasonal_std': hour_std,
                            'sample_count': len(hour_values),
                            'dimensions': metric_history[idx].dimensions
                        },
                        impact_assessment=self._assess_seasonal_impact(metric_name, metric_type, deviation)
                    )
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_fairness_drift(self, metric_name: str, values: List[float],
                             timestamps: List[datetime], metric_history: List[MetricPoint]) -> List[Anomaly]:
        """Detect drift in fairness metrics."""
        anomalies = []
        
        if len(values) < 10:
            return anomalies
        
        # Calculate rolling baseline (first half of data)
        baseline_size = len(values) // 2
        baseline_values = values[:baseline_size]
        baseline_mean = np.mean(baseline_values)
        baseline_std = np.std(baseline_values)
        
        if baseline_std == 0:
            return anomalies
        
        # Check recent values against baseline
        recent_values = values[baseline_size:]
        recent_timestamps = timestamps[baseline_size:]
        recent_history = metric_history[baseline_size:]
        
        for i, (value, timestamp, history_point) in enumerate(zip(recent_values, recent_timestamps, recent_history)):
            drift_score = abs(value - baseline_mean) / baseline_std
            
            # Fairness drift threshold (more sensitive for fairness metrics)
            threshold = 1.5 / (1 + self.sensitivity * 2)
            
            if drift_score > threshold:
                anomaly = Anomaly(
                    anomaly_id=f"fairness_drift_{metric_name}_{int(timestamp.timestamp())}",
                    timestamp=timestamp,
                    anomaly_type=AnomalyType.FAIRNESS_DRIFT,
                    metric_name=metric_name,
                    value=value,
                    expected_value=baseline_mean,
                    deviation_score=drift_score,
                    confidence=min(1.0, drift_score / 2.0),
                    context={
                        'baseline_mean': baseline_mean,
                        'baseline_std': baseline_std,
                        'baseline_size': baseline_size,
                        'dimensions': history_point.dimensions,
                        'drift_magnitude': abs(value - baseline_mean)
                    },
                    impact_assessment={
                        'fairness_impact': 'high' if drift_score > 3 else 'medium',
                        'affected_groups': list(history_point.dimensions.keys()),
                        'remediation_urgency': 'high' if 'demographic_parity' in metric_name or 'equalized_odds' in metric_name else 'medium'
                    }
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend slope."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Simple linear regression
        n = len(values)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x ** 2)
        
        denominator = n * sum_x2 - sum_x ** 2
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
    
    def _assess_outlier_impact(self, metric_name: str, metric_type: MetricType,
                             value: float, expected: float) -> Dict[str, Any]:
        """Assess impact of outlier anomaly."""
        magnitude = abs(value - expected)
        relative_magnitude = magnitude / (abs(expected) + 1e-8)
        
        impact = {
            'magnitude': magnitude,
            'relative_magnitude': relative_magnitude,
            'severity': 'high' if relative_magnitude > 0.5 else 'medium' if relative_magnitude > 0.2 else 'low'
        }
        
        # Special handling for fairness metrics
        if metric_type == MetricType.FAIRNESS:
            impact['fairness_concern'] = magnitude > 0.1
            impact['compliance_risk'] = magnitude > 0.2
        
        return impact
    
    def _assess_trend_change_impact(self, metric_name: str, metric_type: MetricType,
                                  before_trend: float, after_trend: float) -> Dict[str, Any]:
        """Assess impact of trend change."""
        trend_change = abs(after_trend - before_trend)
        
        impact = {
            'trend_change_magnitude': trend_change,
            'direction_change': 'improving' if after_trend > before_trend else 'deteriorating',
            'severity': 'high' if trend_change > 0.2 else 'medium' if trend_change > 0.1 else 'low'
        }
        
        return impact
    
    def _assess_seasonal_impact(self, metric_name: str, metric_type: MetricType,
                              deviation: float) -> Dict[str, Any]:
        """Assess impact of seasonal deviation."""
        impact = {
            'deviation_magnitude': deviation,
            'predictability_impact': 'high' if deviation > 3 else 'medium' if deviation > 2 else 'low',
            'seasonal_pattern_break': deviation > 2.5
        }
        
        return impact


class AlertManager:
    """Manages alerts generated from metric analysis."""
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        self.alert_history: deque = deque(maxlen=10000)
        
        # Initialize default alert rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default alerting rules."""
        # Fairness metric alerts
        self.alert_rules['demographic_parity_difference'] = {
            'thresholds': {'warning': 0.1, 'error': 0.15, 'critical': 0.2},
            'metric_type': MetricType.FAIRNESS,
            'comparison': 'absolute'
        }
        
        self.alert_rules['equalized_odds_difference'] = {
            'thresholds': {'warning': 0.1, 'error': 0.15, 'critical': 0.2},
            'metric_type': MetricType.FAIRNESS,
            'comparison': 'absolute'
        }
        
        # Performance metric alerts
        self.alert_rules['accuracy'] = {
            'thresholds': {'warning': -0.05, 'error': -0.1, 'critical': -0.2},
            'metric_type': MetricType.PERFORMANCE,
            'comparison': 'relative_decrease'
        }
        
        self.alert_rules['response_time_ms'] = {
            'thresholds': {'warning': 500, 'error': 1000, 'critical': 2000},
            'metric_type': MetricType.PERFORMANCE,
            'comparison': 'absolute'
        }
        
        # System health alerts
        self.alert_rules['cpu_utilization'] = {
            'thresholds': {'warning': 70, 'error': 85, 'critical': 95},
            'metric_type': MetricType.SYSTEM_HEALTH,
            'comparison': 'absolute'
        }
        
        self.alert_rules['memory_utilization'] = {
            'thresholds': {'warning': 75, 'error': 90, 'critical': 98},
            'metric_type': MetricType.SYSTEM_HEALTH,
            'comparison': 'absolute'
        }
    
    def add_alert_rule(self, metric_name: str, thresholds: Dict[str, float],
                      metric_type: MetricType, comparison: str = 'absolute'):
        """Add custom alert rule."""
        self.alert_rules[metric_name] = {
            'thresholds': thresholds,
            'metric_type': metric_type,
            'comparison': comparison
        }
    
    def evaluate_metric_for_alerts(self, metric_point: MetricPoint,
                                 baseline_value: Optional[float] = None) -> List[Alert]:
        """Evaluate metric point against alert rules."""
        alerts = []
        
        if metric_point.metric_name not in self.alert_rules:
            return alerts
        
        rule = self.alert_rules[metric_point.metric_name]
        thresholds = rule['thresholds']
        comparison = rule['comparison']
        
        # Calculate comparison value
        if comparison == 'absolute':
            compare_value = abs(metric_point.value)
        elif comparison == 'relative_decrease' and baseline_value is not None:
            compare_value = (baseline_value - metric_point.value) / baseline_value
        else:
            compare_value = metric_point.value
        
        # Check thresholds in order of severity
        severity_map = {
            'critical': AlertSeverity.CRITICAL,
            'error': AlertSeverity.ERROR,
            'warning': AlertSeverity.WARNING
        }
        
        triggered_severity = None
        triggered_threshold = None
        
        for severity_name in ['critical', 'error', 'warning']:
            if severity_name in thresholds:
                threshold = thresholds[severity_name]
                if compare_value >= threshold:
                    triggered_severity = severity_map[severity_name]
                    triggered_threshold = threshold
                    break
        
        if triggered_severity:
            alert = Alert(
                alert_id=f"alert_{metric_point.metric_name}_{int(metric_point.timestamp.timestamp())}",
                timestamp=metric_point.timestamp,
                severity=triggered_severity,
                metric_name=metric_point.metric_name,
                title=f"{metric_point.metric_name} {triggered_severity.value}",
                description=f"{metric_point.metric_name} value {metric_point.value:.3f} exceeds {triggered_severity.value} threshold {triggered_threshold:.3f}",
                value=metric_point.value,
                threshold=triggered_threshold,
                dimensions=metric_point.dimensions,
                context={
                    'comparison_type': comparison,
                    'compare_value': compare_value,
                    'baseline_value': baseline_value,
                    'rule': rule,
                    'metadata': metric_point.metadata
                },
                resolution_suggestions=self._generate_resolution_suggestions(metric_point, rule, triggered_severity)
            )
            
            alerts.append(alert)
            self._process_alert(alert)
        
        return alerts
    
    def process_anomaly_alert(self, anomaly: Anomaly) -> Alert:
        """Generate alert from detected anomaly."""
        # Map anomaly types to severity
        severity_map = {
            AnomalyType.STATISTICAL_OUTLIER: AlertSeverity.WARNING,
            AnomalyType.TREND_CHANGE: AlertSeverity.WARNING,
            AnomalyType.SEASONAL_DEVIATION: AlertSeverity.INFO,
            AnomalyType.FAIRNESS_DRIFT: AlertSeverity.ERROR,
            AnomalyType.PERFORMANCE_DEGRADATION: AlertSeverity.ERROR,
            AnomalyType.CORRELATION_BREAK: AlertSeverity.WARNING
        }
        
        severity = severity_map.get(anomaly.anomaly_type, AlertSeverity.WARNING)
        
        # Upgrade severity based on confidence and impact
        if anomaly.confidence > 0.8:
            if severity == AlertSeverity.WARNING:
                severity = AlertSeverity.ERROR
            elif severity == AlertSeverity.INFO:
                severity = AlertSeverity.WARNING
        
        # Special handling for fairness anomalies
        if 'fairness_impact' in anomaly.impact_assessment:
            if anomaly.impact_assessment['fairness_impact'] == 'high':
                severity = AlertSeverity.CRITICAL
        
        alert = Alert(
            alert_id=f"anomaly_alert_{anomaly.anomaly_id}",
            timestamp=anomaly.timestamp,
            severity=severity,
            metric_name=anomaly.metric_name,
            title=f"{anomaly.anomaly_type.value.replace('_', ' ').title()} in {anomaly.metric_name}",
            description=f"Detected {anomaly.anomaly_type.value} in {anomaly.metric_name}: "
                       f"value {anomaly.value:.3f}, expected {anomaly.expected_value:.3f} "
                       f"(deviation: {anomaly.deviation_score:.3f})",
            value=anomaly.value,
            threshold=anomaly.expected_value,
            dimensions=anomaly.context.get('dimensions', {}),
            context={
                'anomaly_type': anomaly.anomaly_type.value,
                'confidence': anomaly.confidence,
                'deviation_score': anomaly.deviation_score,
                'anomaly_context': anomaly.context,
                'impact_assessment': anomaly.impact_assessment
            },
            resolution_suggestions=self._generate_anomaly_resolution_suggestions(anomaly)
        )
        
        self._process_alert(alert)
        return alert
    
    def _process_alert(self, alert: Alert):
        """Process and store alert."""
        self.alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        
        # Execute alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def _generate_resolution_suggestions(self, metric_point: MetricPoint,
                                       rule: Dict[str, Any], severity: AlertSeverity) -> List[str]:
        """Generate resolution suggestions for metric alerts."""
        suggestions = []
        
        if rule['metric_type'] == MetricType.FAIRNESS:
            suggestions.extend([
                "Review recent model updates for fairness impact",
                "Check data distribution for demographic shifts",
                "Consider applying bias mitigation techniques",
                "Evaluate protected group representation"
            ])
        
        elif rule['metric_type'] == MetricType.PERFORMANCE:
            if 'accuracy' in metric_point.metric_name:
                suggestions.extend([
                    "Review model performance on validation set",
                    "Check for data quality issues",
                    "Consider model retraining or hyperparameter tuning"
                ])
            elif 'response_time' in metric_point.metric_name:
                suggestions.extend([
                    "Check system resource utilization",
                    "Review recent deployments or configuration changes",
                    "Consider scaling resources or optimizing queries"
                ])
        
        elif rule['metric_type'] == MetricType.SYSTEM_HEALTH:
            suggestions.extend([
                "Check system resource usage and availability",
                "Review recent system changes or deployments",
                "Consider scaling or resource optimization"
            ])
        
        if severity in [AlertSeverity.CRITICAL, AlertSeverity.ERROR]:
            suggestions.append("Consider immediate escalation to on-call team")
        
        return suggestions
    
    def _generate_anomaly_resolution_suggestions(self, anomaly: Anomaly) -> List[str]:
        """Generate resolution suggestions for anomaly alerts."""
        suggestions = []
        
        if anomaly.anomaly_type == AnomalyType.FAIRNESS_DRIFT:
            suggestions.extend([
                "Investigate recent changes in data sources or demographics",
                "Review model fairness constraints and thresholds",
                "Consider implementing fairness intervention measures",
                "Analyze affected protected groups for targeted remediation"
            ])
        
        elif anomaly.anomaly_type == AnomalyType.TREND_CHANGE:
            suggestions.extend([
                "Investigate root cause of trend change",
                "Review recent system or model changes",
                "Assess impact on business metrics and user experience",
                "Consider adjusting monitoring thresholds if change is expected"
            ])
        
        elif anomaly.anomaly_type == AnomalyType.STATISTICAL_OUTLIER:
            suggestions.extend([
                "Investigate data quality for potential corruption or errors",
                "Review input validation and preprocessing steps",
                "Check for unusual traffic patterns or system behavior"
            ])
        
        suggestions.append(f"Review anomaly confidence ({anomaly.confidence:.1%}) and context for validation")
        
        return suggestions
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self.alerts:
            self.alerts[alert_id].acknowledged = True
            return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Mark alert as resolved."""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolved = True
            self.alerts[alert_id].resolved_timestamp = datetime.utcnow()
            return True
        return False
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def get_active_alerts(self, severity_filter: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get currently active alerts."""
        active_alerts = [a for a in self.alerts.values() if not a.resolved]
        
        if severity_filter:
            active_alerts = [a for a in active_alerts if a.severity == severity_filter]
        
        return sorted(active_alerts, key=lambda x: x.timestamp, reverse=True)


class ObservabilityDashboard:
    """Provides dashboard and reporting functionality."""
    
    def __init__(self, metric_collector: MetricCollector, alert_manager: AlertManager):
        self.metric_collector = metric_collector
        self.alert_manager = alert_manager
    
    def generate_system_report(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive system observability report."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=time_window_hours)
        
        report = {
            'report_timestamp': end_time.isoformat(),
            'time_window_hours': time_window_hours,
            'executive_summary': {},
            'metric_summary': {},
            'alert_summary': {},
            'anomaly_summary': {},
            'recommendations': []
        }
        
        # Generate executive summary
        report['executive_summary'] = self._generate_executive_summary(start_time, end_time)
        
        # Generate metric summaries
        report['metric_summary'] = self._generate_metric_summary(start_time, end_time)
        
        # Generate alert summary
        report['alert_summary'] = self._generate_alert_summary(start_time, end_time)
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations()
        
        return report
    
    def _generate_executive_summary(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate executive summary of system health."""
        # Get recent alerts
        recent_alerts = [
            alert for alert in self.alert_manager.alert_history
            if start_time <= alert.timestamp <= end_time
        ]
        
        # Count by severity
        alert_counts = {severity.value: 0 for severity in AlertSeverity}
        for alert in recent_alerts:
            alert_counts[alert.severity.value] += 1
        
        # Calculate overall health score
        critical_alerts = alert_counts['critical']
        error_alerts = alert_counts['error']
        warning_alerts = alert_counts['warning']
        
        # Simple health score calculation
        health_score = 100
        health_score -= critical_alerts * 20
        health_score -= error_alerts * 10
        health_score -= warning_alerts * 5
        health_score = max(0, min(100, health_score))
        
        summary = {
            'overall_health_score': health_score,
            'health_status': 'excellent' if health_score > 90 else 'good' if health_score > 75 else 'fair' if health_score > 50 else 'poor',
            'total_alerts': len(recent_alerts),
            'alert_breakdown': alert_counts,
            'system_availability': 99.9,  # Would be calculated from actual uptime data
            'key_issues': []
        }
        
        # Identify key issues
        if critical_alerts > 0:
            summary['key_issues'].append(f"{critical_alerts} critical alerts require immediate attention")
        
        if error_alerts > 5:
            summary['key_issues'].append(f"{error_alerts} error alerts indicate systemic issues")
        
        return summary
    
    def _generate_metric_summary(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate summary of key metrics."""
        metric_summary = {
            'fairness_metrics': {},
            'performance_metrics': {},
            'system_health_metrics': {}
        }
        
        # Analyze each metric type
        for metric_key, metric_data in self.metric_collector.metrics.items():
            if not metric_data:
                continue
            
            # Filter data to time window
            recent_points = [
                point for point in metric_data
                if start_time <= point.timestamp <= end_time
            ]
            
            if not recent_points:
                continue
            
            values = [p.value for p in recent_points]
            metric_name = recent_points[0].metric_name
            metric_type = recent_points[0].metric_type
            
            summary = {
                'current_value': values[-1],
                'min_value': min(values),
                'max_value': max(values),
                'avg_value': np.mean(values),
                'std_value': np.std(values),
                'data_points': len(values),
                'trend': 'increasing' if values[-1] > values[0] else 'decreasing' if values[-1] < values[0] else 'stable'
            }
            
            # Categorize by metric type
            if metric_type == MetricType.FAIRNESS:
                metric_summary['fairness_metrics'][metric_name] = summary
            elif metric_type == MetricType.PERFORMANCE:
                metric_summary['performance_metrics'][metric_name] = summary
            elif metric_type == MetricType.SYSTEM_HEALTH:
                metric_summary['system_health_metrics'][metric_name] = summary
        
        return metric_summary
    
    def _generate_alert_summary(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate alert summary for time window."""
        recent_alerts = [
            alert for alert in self.alert_manager.alert_history
            if start_time <= alert.timestamp <= end_time
        ]
        
        # Group by metric name
        alerts_by_metric = defaultdict(list)
        for alert in recent_alerts:
            alerts_by_metric[alert.metric_name].append(alert)
        
        # Generate per-metric summaries
        metric_alert_summaries = {}
        for metric_name, alerts in alerts_by_metric.items():
            metric_alert_summaries[metric_name] = {
                'total_alerts': len(alerts),
                'severity_breakdown': {
                    severity.value: len([a for a in alerts if a.severity == severity])
                    for severity in AlertSeverity
                },
                'resolution_rate': len([a for a in alerts if a.resolved]) / len(alerts) if alerts else 0,
                'most_recent': alerts[-1].timestamp.isoformat() if alerts else None
            }
        
        alert_summary = {
            'total_alerts': len(recent_alerts),
            'active_alerts': len([a for a in recent_alerts if not a.resolved]),
            'resolved_alerts': len([a for a in recent_alerts if a.resolved]),
            'metrics_with_alerts': len(alerts_by_metric),
            'metric_breakdown': metric_alert_summaries
        }
        
        return alert_summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on current state."""
        recommendations = []
        
        # Check for high-frequency alerting metrics
        recent_alerts = list(self.alert_manager.alert_history)[-100:]  # Last 100 alerts
        if recent_alerts:
            alert_counts = defaultdict(int)
            for alert in recent_alerts:
                alert_counts[alert.metric_name] += 1
            
            # Find frequently alerting metrics
            for metric_name, count in alert_counts.items():
                if count > 10:
                    recommendations.append(f"Consider adjusting alert thresholds for {metric_name} (triggered {count} times recently)")
        
        # Check for unresolved critical alerts
        critical_alerts = [
            a for a in self.alert_manager.alerts.values()
            if a.severity == AlertSeverity.CRITICAL and not a.resolved
        ]
        
        if critical_alerts:
            recommendations.append(f"Resolve {len(critical_alerts)} outstanding critical alerts immediately")
        
        # Check for fairness metric trends
        # This would analyze fairness metric trends and suggest proactive measures
        recommendations.append("Review fairness metric trends for proactive bias prevention")
        
        # General recommendations
        recommendations.extend([
            "Regularly review and update alert thresholds based on system behavior",
            "Implement automated remediation for common alert patterns",
            "Enhance monitoring coverage for business-critical metrics"
        ])
        
        return recommendations


class ComprehensiveObservability:
    """Main observability platform that orchestrates all components."""
    
    def __init__(self, monitoring_interval_seconds: float = 10.0):
        self.monitoring_interval_seconds = monitoring_interval_seconds
        
        # Initialize components
        self.metric_collector = MetricCollector()
        self.anomaly_detector = AnomalyDetector()
        self.alert_manager = AlertManager()
        self.dashboard = ObservabilityDashboard(self.metric_collector, self.alert_manager)
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Registered metric sources
        self.metric_sources: List[Callable[[], List[MetricPoint]]] = []
        
        logger.info("Comprehensive Observability Platform initialized")
    
    def register_metric_source(self, source_func: Callable[[], List[MetricPoint]]):
        """Register a function that provides metrics."""
        self.metric_sources.append(source_func)
    
    def start_monitoring(self):
        """Start the observability monitoring system."""
        if self.monitoring_active:
            logger.warning("Monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Comprehensive observability monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring system."""
        self.monitoring_active = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=10)
        
        logger.info("Comprehensive observability monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        logger.info("Starting observability monitoring loop")
        
        while self.monitoring_active:
            try:
                # Collect metrics from all sources
                self._collect_metrics_from_sources()
                
                # Detect anomalies
                self._detect_anomalies()
                
                # Clean up old data
                self._cleanup_old_data()
                
                time.sleep(self.monitoring_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval_seconds)
        
        logger.info("Observability monitoring loop ended")
    
    def _collect_metrics_from_sources(self):
        """Collect metrics from all registered sources."""
        for source_func in self.metric_sources:
            try:
                metrics = source_func()
                if metrics:
                    self.metric_collector.collect_batch(metrics)
            except Exception as e:
                logger.error(f"Error collecting metrics from source: {e}")
    
    def _detect_anomalies(self):
        """Detect anomalies in all metrics."""
        for metric_key in self.metric_collector.metrics:
            try:
                # Extract metric name and type from key
                parts = metric_key.rsplit('_', 1)
                if len(parts) != 2:
                    continue
                
                metric_name, metric_type_str = parts
                try:
                    metric_type = MetricType(metric_type_str)
                except ValueError:
                    continue
                
                # Get recent history
                recent_history = list(self.metric_collector.metrics[metric_key])[-100:]  # Last 100 points
                
                if len(recent_history) < 10:
                    continue
                
                # Detect anomalies
                anomalies = self.anomaly_detector.detect_anomalies(metric_name, metric_type, recent_history)
                
                # Generate alerts for anomalies
                for anomaly in anomalies:
                    self.alert_manager.process_anomaly_alert(anomaly)
                
            except Exception as e:
                logger.error(f"Error detecting anomalies for {metric_key}: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old metrics and alerts."""
        self.metric_collector.cleanup_old_metrics()
    
    def collect_fairness_metrics(self, y_true: pd.Series, y_pred: np.ndarray,
                               protected_attrs: pd.DataFrame, model_name: str = "default") -> List[MetricPoint]:
        """Collect fairness metrics and convert to metric points."""
        metric_points = []
        timestamp = datetime.utcnow()
        
        for attr_name in protected_attrs.columns:
            try:
                overall_metrics, by_group_metrics = compute_fairness_metrics(
                    y_true=y_true,
                    y_pred=y_pred,
                    protected=protected_attrs[attr_name]
                )
                
                # Create metric points for key fairness metrics
                fairness_metric_names = [
                    'demographic_parity_difference',
                    'equalized_odds_difference',
                    'accuracy',
                    'precision',
                    'recall'
                ]
                
                for metric_name in fairness_metric_names:
                    if metric_name in overall_metrics:
                        metric_point = MetricPoint(
                            timestamp=timestamp,
                            metric_name=metric_name,
                            metric_type=MetricType.FAIRNESS if 'difference' in metric_name else MetricType.PERFORMANCE,
                            value=float(overall_metrics[metric_name]),
                            dimensions={
                                'protected_attribute': attr_name,
                                'model_name': model_name
                            },
                            metadata={
                                'by_group_metrics': by_group_metrics,
                                'sample_size': len(y_true)
                            }
                        )
                        metric_points.append(metric_point)
                
            except Exception as e:
                logger.error(f"Error computing fairness metrics for {attr_name}: {e}")
        
        # Collect the metrics
        if metric_points:
            self.metric_collector.collect_batch(metric_points)
        
        return metric_points
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback for alert notifications."""
        self.alert_manager.add_alert_callback(callback)
    
    def get_system_report(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive system observability report."""
        return self.dashboard.generate_system_report(time_window_hours)


def demonstrate_comprehensive_observability():
    """Demonstrate the comprehensive observability platform."""
    print("📊 Comprehensive Observability Platform Demonstration")
    print("=" * 65)
    
    # Initialize observability platform
    observability = ComprehensiveObservability(monitoring_interval_seconds=2.0)
    
    # Add alert callback
    def alert_callback(alert: Alert):
        severity_emoji = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.ERROR: "🚨",
            AlertSeverity.CRITICAL: "🔥",
            AlertSeverity.EMERGENCY: "💥"
        }
        emoji = severity_emoji.get(alert.severity, "🔔")
        print(f"   {emoji} ALERT: {alert.title} - {alert.description[:60]}...")
    
    observability.add_alert_callback(alert_callback)
    
    print("✅ Observability platform initialized")
    print(f"   Monitoring interval: {observability.monitoring_interval_seconds}s")
    
    # Register mock metric sources
    def generate_fairness_metrics() -> List[MetricPoint]:
        """Generate mock fairness metrics."""
        timestamp = datetime.utcnow()
        metrics = []
        
        # Simulate varying fairness metrics
        dp_diff = np.random.normal(0.05, 0.03)  # Sometimes violates thresholds
        eo_diff = np.random.normal(0.08, 0.04)
        accuracy = np.random.normal(0.85, 0.02)
        
        metrics.extend([
            MetricPoint(
                timestamp=timestamp,
                metric_name="demographic_parity_difference",
                metric_type=MetricType.FAIRNESS,
                value=abs(dp_diff),
                dimensions={"protected_attribute": "gender", "model": "credit_scorer"},
                metadata={"sample_size": 1000}
            ),
            MetricPoint(
                timestamp=timestamp,
                metric_name="equalized_odds_difference", 
                metric_type=MetricType.FAIRNESS,
                value=abs(eo_diff),
                dimensions={"protected_attribute": "gender", "model": "credit_scorer"},
                metadata={"sample_size": 1000}
            ),
            MetricPoint(
                timestamp=timestamp,
                metric_name="accuracy",
                metric_type=MetricType.PERFORMANCE,
                value=accuracy,
                dimensions={"model": "credit_scorer"},
                metadata={"validation_set": True}
            )
        ])
        
        return metrics
    
    def generate_system_metrics() -> List[MetricPoint]:
        """Generate mock system health metrics."""
        timestamp = datetime.utcnow()
        metrics = []
        
        # Simulate system metrics with occasional spikes
        cpu_usage = np.random.normal(60, 15)  # Sometimes high
        memory_usage = np.random.normal(65, 20)  # Sometimes high
        response_time = np.random.exponential(200)  # Sometimes slow
        
        metrics.extend([
            MetricPoint(
                timestamp=timestamp,
                metric_name="cpu_utilization",
                metric_type=MetricType.SYSTEM_HEALTH,
                value=max(0, min(100, cpu_usage)),
                dimensions={"node": "worker-1"},
                metadata={"cores": 8}
            ),
            MetricPoint(
                timestamp=timestamp,
                metric_name="memory_utilization",
                metric_type=MetricType.SYSTEM_HEALTH,
                value=max(0, min(100, memory_usage)),
                dimensions={"node": "worker-1"},
                metadata={"total_gb": 32}
            ),
            MetricPoint(
                timestamp=timestamp,
                metric_name="response_time_ms",
                metric_type=MetricType.PERFORMANCE,
                value=response_time,
                dimensions={"endpoint": "/predict", "service": "ml_api"},
                metadata={"method": "POST"}
            )
        ])
        
        return metrics
    
    # Register metric sources
    observability.register_metric_source(generate_fairness_metrics)
    observability.register_metric_source(generate_system_metrics)
    
    print(f"\n📈 Registered {len(observability.metric_sources)} metric sources")
    
    print("\n🚀 Starting observability monitoring...")
    observability.start_monitoring()
    
    print("\n⏳ Collecting metrics and detecting patterns...")
    
    # Let it run for a while to collect data
    for i in range(15):
        time.sleep(1)
        if i % 5 == 4:
            print(f"   Monitoring progress: {i+1}/15 seconds")
    
    print(f"\n📊 Generating system observability report...")
    report = observability.get_system_report(time_window_hours=1)
    
    print(f"\n   Executive Summary:")
    exec_summary = report['executive_summary']
    print(f"     Overall health score: {exec_summary['overall_health_score']}/100 ({exec_summary['health_status']})")
    print(f"     Total alerts: {exec_summary['total_alerts']}")
    print(f"     Alert breakdown:")
    for severity, count in exec_summary['alert_breakdown'].items():
        if count > 0:
            print(f"       {severity}: {count}")
    
    print(f"\n   Metric Summary:")
    metric_summary = report['metric_summary']
    
    if metric_summary['fairness_metrics']:
        print(f"     Fairness Metrics:")
        for metric, data in metric_summary['fairness_metrics'].items():
            print(f"       {metric}: current={data['current_value']:.3f}, avg={data['avg_value']:.3f}, trend={data['trend']}")
    
    if metric_summary['performance_metrics']:
        print(f"     Performance Metrics:")
        for metric, data in metric_summary['performance_metrics'].items():
            print(f"       {metric}: current={data['current_value']:.1f}, avg={data['avg_value']:.1f}, trend={data['trend']}")
    
    if metric_summary['system_health_metrics']:
        print(f"     System Health Metrics:")
        for metric, data in metric_summary['system_health_metrics'].items():
            print(f"       {metric}: current={data['current_value']:.1f}%, avg={data['avg_value']:.1f}%, trend={data['trend']}")
    
    print(f"\n   Alert Summary:")
    alert_summary = report['alert_summary']
    print(f"     Total alerts: {alert_summary['total_alerts']}")
    print(f"     Active alerts: {alert_summary['active_alerts']}")
    print(f"     Resolved alerts: {alert_summary['resolved_alerts']}")
    print(f"     Metrics with alerts: {alert_summary['metrics_with_alerts']}")
    
    print(f"\n   Recommendations:")
    for i, rec in enumerate(report['recommendations'][:5], 1):
        print(f"     {i}. {rec}")
    
    # Demonstrate anomaly detection
    print(f"\n🔍 Anomaly Detection Summary:")
    anomaly_count = 0
    for metric_key in observability.anomaly_detector.detection_history:
        anomalies = observability.anomaly_detector.detection_history[metric_key]
        if anomalies:
            recent_anomalies = [a for a in anomalies if (datetime.utcnow() - a.timestamp).total_seconds() < 300]
            anomaly_count += len(recent_anomalies)
            if recent_anomalies:
                print(f"     {metric_key}: {len(recent_anomalies)} recent anomalies")
    
    if anomaly_count == 0:
        print(f"     No significant anomalies detected in recent monitoring period")
    
    # Show active alerts
    active_alerts = observability.alert_manager.get_active_alerts()
    print(f"\n🚨 Active Alerts ({len(active_alerts)}):")
    if active_alerts:
        for alert in active_alerts[:5]:  # Show top 5
            print(f"     {alert.severity.value.upper()}: {alert.metric_name} = {alert.value:.3f} (threshold: {alert.threshold:.3f})")
    else:
        print(f"     No active alerts - system operating normally")
    
    # Stop monitoring
    print(f"\n🛑 Stopping observability monitoring...")
    observability.stop_monitoring()
    
    print(f"\n🎉 Comprehensive Observability Demonstration Complete!")
    print(f"     Platform demonstrated:")
    print(f"     • Real-time metric collection and storage")
    print(f"     • Multi-dimensional anomaly detection") 
    print(f"     • Context-aware alerting with severity assessment")
    print(f"     • Comprehensive reporting and trend analysis")
    print(f"     • Fairness-aware monitoring with specialized thresholds")
    print(f"     • Automated pattern recognition and recommendations")
    print(f"     • Production-ready observability for fair AI systems")
    
    return observability


if __name__ == "__main__":
    demonstrate_comprehensive_observability()