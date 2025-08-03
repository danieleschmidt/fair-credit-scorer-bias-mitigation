"""
Real-time bias monitoring and alerting system.

This module provides comprehensive bias monitoring capabilities including
drift detection, alerting, and automated remediation suggestions.
"""

import asyncio
import json
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass, asdict
from enum import Enum

import pandas as pd
import numpy as np
from scipy import stats

from ..fairness_metrics import compute_fairness_metrics
from ..logging_config import get_logger

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class BiasAlert:
    """Bias monitoring alert."""
    id: str
    timestamp: datetime
    severity: AlertSeverity
    metric_name: str
    current_value: float
    threshold: float
    model_name: str
    message: str
    suggested_actions: List[str]
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['severity'] = self.severity.value
        return data


@dataclass
class MonitoringWindow:
    """Time window for monitoring metrics."""
    start_time: datetime
    end_time: datetime
    sample_count: int
    metrics: Dict[str, float]
    drift_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert window to dictionary."""
        return {
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'sample_count': self.sample_count,
            'metrics': self.metrics,
            'drift_score': self.drift_score
        }


class DriftDetector:
    """Statistical drift detection algorithms."""
    
    def __init__(self, window_size: int = 1000, alpha: float = 0.05):
        """
        Initialize drift detector.
        
        Args:
            window_size: Size of the reference window
            alpha: Significance level for statistical tests
        """
        self.window_size = window_size
        self.alpha = alpha
        self.reference_data = {}
        
    def update_reference(self, metric_name: str, values: List[float]):
        """Update reference distribution for a metric."""
        self.reference_data[metric_name] = values[-self.window_size:]
        logger.debug(f"Updated reference for {metric_name} with {len(values)} samples")
    
    def detect_drift(self, metric_name: str, current_values: List[float]) -> Tuple[bool, float]:
        """
        Detect drift using Kolmogorov-Smirnov test.
        
        Args:
            metric_name: Name of the metric
            current_values: Current metric values
            
        Returns:
            Tuple of (drift_detected, p_value)
        """
        if metric_name not in self.reference_data:
            logger.warning(f"No reference data for {metric_name}")
            return False, 1.0
        
        if len(current_values) < 10:
            logger.warning(f"Insufficient current data for {metric_name}")
            return False, 1.0
        
        try:
            # Kolmogorov-Smirnov test
            statistic, p_value = stats.ks_2samp(
                self.reference_data[metric_name],
                current_values
            )
            
            drift_detected = p_value < self.alpha
            
            logger.debug(f"Drift test for {metric_name}: p={p_value:.4f}, drift={drift_detected}")
            return drift_detected, p_value
            
        except Exception as e:
            logger.error(f"Drift detection failed for {metric_name}: {e}")
            return False, 1.0
    
    def compute_drift_magnitude(self, metric_name: str, current_values: List[float]) -> float:
        """
        Compute magnitude of drift using effect size.
        
        Args:
            metric_name: Name of the metric
            current_values: Current metric values
            
        Returns:
            Effect size (Cohen's d)
        """
        if metric_name not in self.reference_data or len(current_values) < 2:
            return 0.0
        
        try:
            ref_values = self.reference_data[metric_name]
            
            # Cohen's d effect size
            pooled_std = np.sqrt(
                ((len(ref_values) - 1) * np.var(ref_values, ddof=1) +
                 (len(current_values) - 1) * np.var(current_values, ddof=1)) /
                (len(ref_values) + len(current_values) - 2)
            )
            
            if pooled_std == 0:
                return 0.0
            
            effect_size = abs(np.mean(current_values) - np.mean(ref_values)) / pooled_std
            return effect_size
            
        except Exception as e:
            logger.error(f"Effect size calculation failed for {metric_name}: {e}")
            return 0.0


class BiasMonitor:
    """
    Production bias monitoring system.
    
    Features:
    - Real-time bias drift detection
    - Configurable alerting thresholds
    - Historical trend analysis
    - Automated remediation suggestions
    - Performance impact tracking
    """
    
    def __init__(
        self,
        window_duration: timedelta = timedelta(hours=1),
        max_windows: int = 168,  # 1 week of hourly windows
        drift_detector_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize bias monitor.
        
        Args:
            window_duration: Duration of each monitoring window
            max_windows: Maximum number of windows to retain
            drift_detector_config: Configuration for drift detector
        """
        self.window_duration = window_duration
        self.max_windows = max_windows
        
        # Data storage
        self.monitoring_windows = defaultdict(deque)  # model_name -> deque of windows
        self.recent_predictions = defaultdict(deque)  # model_name -> deque of predictions
        self.alerts = deque(maxlen=1000)  # Recent alerts
        
        # Drift detection
        drift_config = drift_detector_config or {}
        self.drift_detector = DriftDetector(**drift_config)
        
        # Thresholds for alerting
        self.thresholds = {
            "demographic_parity_difference": {"high": 0.1, "critical": 0.2},
            "equalized_odds_difference": {"high": 0.1, "critical": 0.2},
            "accuracy_difference": {"high": 0.05, "critical": 0.1},
            "accuracy": {"low": 0.7, "critical": 0.6}  # Performance degradation
        }
        
        logger.info("BiasMonitor initialized")
    
    def add_prediction(
        self,
        model_name: str,
        prediction: int,
        true_label: Optional[int],
        protected_attribute: Any,
        features: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ):
        """
        Add a new prediction for monitoring.
        
        Args:
            model_name: Name of the model
            prediction: Model prediction
            true_label: True label (if available)
            protected_attribute: Protected attribute value
            features: Input features (optional)
            timestamp: Prediction timestamp (current time if None)
        """
        timestamp = timestamp or datetime.utcnow()
        
        prediction_data = {
            'timestamp': timestamp,
            'prediction': prediction,
            'true_label': true_label,
            'protected_attribute': protected_attribute,
            'features': features or {}
        }
        
        self.recent_predictions[model_name].append(prediction_data)
        
        # Limit memory usage
        if len(self.recent_predictions[model_name]) > 10000:
            self.recent_predictions[model_name].popleft()
        
        logger.debug(f"Added prediction for {model_name}")
    
    def process_monitoring_window(self, model_name: str) -> Optional[MonitoringWindow]:
        """
        Process current monitoring window and detect bias drift.
        
        Args:
            model_name: Name of the model to monitor
            
        Returns:
            Monitoring window with results (if sufficient data)
        """
        try:
            current_time = datetime.utcnow()
            window_start = current_time - self.window_duration
            
            # Get predictions in current window
            recent_preds = list(self.recent_predictions[model_name])
            window_preds = [
                p for p in recent_preds
                if p['timestamp'] >= window_start and p['true_label'] is not None
            ]
            
            if len(window_preds) < 10:
                logger.debug(f"Insufficient data for {model_name} monitoring window")
                return None
            
            # Convert to DataFrame for metrics computation
            df = pd.DataFrame(window_preds)
            
            # Compute fairness metrics
            overall, by_group = compute_fairness_metrics(
                df['true_label'],
                df['prediction'],
                df['protected_attribute']
            )
            
            # Create monitoring window
            window = MonitoringWindow(
                start_time=window_start,
                end_time=current_time,
                sample_count=len(window_preds),
                metrics=overall.to_dict()
            )
            
            # Detect drift
            drift_scores = []
            for metric_name in ["demographic_parity_difference", "equalized_odds_difference", "accuracy"]:
                if metric_name in overall:
                    # Use historical windows for drift detection
                    historical_values = self._get_historical_values(model_name, metric_name)
                    current_value = overall[metric_name]
                    
                    if len(historical_values) > 0:
                        drift_detected, p_value = self.drift_detector.detect_drift(
                            f"{model_name}_{metric_name}",
                            [current_value]
                        )
                        
                        magnitude = self.drift_detector.compute_drift_magnitude(
                            f"{model_name}_{metric_name}",
                            [current_value]
                        )
                        
                        drift_scores.append(magnitude)
                        
                        # Generate alerts if needed
                        if drift_detected:
                            self._generate_drift_alert(model_name, metric_name, current_value, magnitude)
            
            window.drift_score = np.mean(drift_scores) if drift_scores else 0.0
            
            # Store window
            self.monitoring_windows[model_name].append(window)
            if len(self.monitoring_windows[model_name]) > self.max_windows:
                self.monitoring_windows[model_name].popleft()
            
            # Update drift detector reference
            for metric_name, value in window.metrics.items():
                historical_values = self._get_historical_values(model_name, metric_name)
                historical_values.append(value)
                self.drift_detector.update_reference(f"{model_name}_{metric_name}", historical_values)
            
            # Check thresholds and generate alerts
            self._check_thresholds(model_name, window.metrics)
            
            logger.info(f"Processed monitoring window for {model_name}: {len(window_preds)} samples")
            return window
            
        except Exception as e:
            logger.error(f"Failed to process monitoring window for {model_name}: {e}")
            return None
    
    def get_monitoring_status(self, model_name: str) -> Dict[str, Any]:
        """
        Get current monitoring status for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Monitoring status dictionary
        """
        try:
            windows = list(self.monitoring_windows[model_name])
            recent_alerts = [
                alert for alert in self.alerts
                if alert.model_name == model_name and
                alert.timestamp >= datetime.utcnow() - timedelta(hours=24)
            ]
            
            # Compute trends
            trends = self._compute_trends(model_name)
            
            status = {
                "model_name": model_name,
                "last_updated": datetime.utcnow().isoformat(),
                "monitoring_windows": len(windows),
                "recent_alerts": len(recent_alerts),
                "alert_summary": self._summarize_alerts(recent_alerts),
                "current_metrics": windows[-1].metrics if windows else {},
                "trends": trends,
                "drift_score": windows[-1].drift_score if windows else 0.0,
                "recommendations": self._generate_recommendations(model_name, recent_alerts)
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get monitoring status for {model_name}: {e}")
            return {"error": str(e)}
    
    def get_historical_data(
        self,
        model_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get historical monitoring data for a model.
        
        Args:
            model_name: Name of the model
            start_time: Start of time range (24 hours ago if None)
            end_time: End of time range (now if None)
            
        Returns:
            List of monitoring windows
        """
        end_time = end_time or datetime.utcnow()
        start_time = start_time or (end_time - timedelta(hours=24))
        
        windows = list(self.monitoring_windows[model_name])
        filtered_windows = [
            window for window in windows
            if start_time <= window.end_time <= end_time
        ]
        
        return [window.to_dict() for window in filtered_windows]
    
    def _get_historical_values(self, model_name: str, metric_name: str) -> List[float]:
        """Get historical values for a metric."""
        windows = list(self.monitoring_windows[model_name])
        values = []
        
        for window in windows:
            if metric_name in window.metrics:
                values.append(window.metrics[metric_name])
        
        return values
    
    def _check_thresholds(self, model_name: str, metrics: Dict[str, float]):
        """Check metrics against thresholds and generate alerts."""
        for metric_name, value in metrics.items():
            if metric_name not in self.thresholds:
                continue
            
            thresholds = self.thresholds[metric_name]
            
            # Determine severity
            severity = None
            threshold_violated = None
            
            if "critical" in thresholds:
                if (metric_name == "accuracy" and value < thresholds["critical"]) or \
                   (metric_name != "accuracy" and abs(value) > thresholds["critical"]):
                    severity = AlertSeverity.CRITICAL
                    threshold_violated = thresholds["critical"]
            
            if severity is None and "high" in thresholds:
                if (metric_name == "accuracy" and value < thresholds["high"]) or \
                   (metric_name != "accuracy" and abs(value) > thresholds["high"]):
                    severity = AlertSeverity.HIGH
                    threshold_violated = thresholds["high"]
            
            # Generate alert if threshold violated
            if severity is not None:
                self._generate_threshold_alert(model_name, metric_name, value, threshold_violated, severity)
    
    def _generate_threshold_alert(
        self,
        model_name: str,
        metric_name: str,
        value: float,
        threshold: float,
        severity: AlertSeverity
    ):
        """Generate alert for threshold violation."""
        alert_id = f"{model_name}_{metric_name}_{int(time.time())}"
        
        message = f"{metric_name} threshold violation: {value:.4f} (threshold: {threshold:.4f})"
        
        # Generate suggested actions
        suggestions = self._get_remediation_suggestions(metric_name, value)
        
        alert = BiasAlert(
            id=alert_id,
            timestamp=datetime.utcnow(),
            severity=severity,
            metric_name=metric_name,
            current_value=value,
            threshold=threshold,
            model_name=model_name,
            message=message,
            suggested_actions=suggestions
        )
        
        self.alerts.append(alert)
        logger.warning(f"Generated {severity.value} alert for {model_name}: {message}")
    
    def _generate_drift_alert(
        self,
        model_name: str,
        metric_name: str,
        value: float,
        magnitude: float
    ):
        """Generate alert for detected drift."""
        alert_id = f"{model_name}_{metric_name}_drift_{int(time.time())}"
        
        # Determine severity based on magnitude
        if magnitude > 1.0:
            severity = AlertSeverity.HIGH
        elif magnitude > 0.5:
            severity = AlertSeverity.MEDIUM
        else:
            severity = AlertSeverity.LOW
        
        message = f"Bias drift detected in {metric_name}: effect size {magnitude:.3f}"
        suggestions = [
            "Review recent data for distribution changes",
            "Check for data quality issues",
            "Consider retraining the model",
            "Investigate environmental changes"
        ]
        
        alert = BiasAlert(
            id=alert_id,
            timestamp=datetime.utcnow(),
            severity=severity,
            metric_name=metric_name,
            current_value=value,
            threshold=0.0,  # No specific threshold for drift
            model_name=model_name,
            message=message,
            suggested_actions=suggestions,
            metadata={"drift_magnitude": magnitude}
        )
        
        self.alerts.append(alert)
        logger.warning(f"Generated drift alert for {model_name}: {message}")
    
    def _get_remediation_suggestions(self, metric_name: str, value: float) -> List[str]:
        """Get remediation suggestions for metric violations."""
        suggestions = []
        
        if "demographic_parity" in metric_name:
            suggestions.extend([
                "Apply demographic parity constraint during training",
                "Use preprocessing techniques like reweighting",
                "Implement fairness-aware post-processing",
                "Review feature selection for proxy variables"
            ])
        
        elif "equalized_odds" in metric_name:
            suggestions.extend([
                "Apply equalized odds constraint during training",
                "Use calibration techniques across groups",
                "Implement threshold optimization per group",
                "Review model complexity and regularization"
            ])
        
        elif "accuracy" in metric_name:
            suggestions.extend([
                "Retrain model with more recent data",
                "Review feature engineering pipeline",
                "Check for data drift or quality issues",
                "Consider ensemble methods or model updates"
            ])
        
        suggestions.append("Monitor model performance more frequently")
        suggestions.append("Consider A/B testing with alternative models")
        
        return suggestions
    
    def _compute_trends(self, model_name: str) -> Dict[str, str]:
        """Compute trends for key metrics."""
        windows = list(self.monitoring_windows[model_name])
        if len(windows) < 2:
            return {}
        
        trends = {}
        key_metrics = ["demographic_parity_difference", "equalized_odds_difference", "accuracy"]
        
        for metric_name in key_metrics:
            values = [w.metrics.get(metric_name, 0) for w in windows[-10:]]  # Last 10 windows
            if len(values) >= 2:
                # Simple trend detection
                if values[-1] > values[0]:
                    trends[metric_name] = "increasing"
                elif values[-1] < values[0]:
                    trends[metric_name] = "decreasing"
                else:
                    trends[metric_name] = "stable"
        
        return trends
    
    def _summarize_alerts(self, alerts: List[BiasAlert]) -> Dict[str, int]:
        """Summarize alerts by severity."""
        summary = defaultdict(int)
        for alert in alerts:
            summary[alert.severity.value] += 1
        return dict(summary)
    
    def _generate_recommendations(self, model_name: str, recent_alerts: List[BiasAlert]) -> List[str]:
        """Generate high-level recommendations based on alert patterns."""
        recommendations = []
        
        if len(recent_alerts) == 0:
            recommendations.append("Model performance is stable - continue monitoring")
            return recommendations
        
        # Analyze alert patterns
        critical_alerts = [a for a in recent_alerts if a.severity == AlertSeverity.CRITICAL]
        high_alerts = [a for a in recent_alerts if a.severity == AlertSeverity.HIGH]
        
        if len(critical_alerts) > 0:
            recommendations.append("URGENT: Critical bias violations detected - immediate action required")
            recommendations.append("Consider temporarily disabling the model until issues are resolved")
        
        elif len(high_alerts) > 2:
            recommendations.append("Multiple high-severity alerts - schedule model review and retraining")
            recommendations.append("Implement enhanced monitoring with shorter time windows")
        
        # Metric-specific recommendations
        fairness_alerts = [a for a in recent_alerts if "parity" in a.metric_name or "odds" in a.metric_name]
        if len(fairness_alerts) > 0:
            recommendations.append("Fairness violations detected - review bias mitigation strategies")
        
        performance_alerts = [a for a in recent_alerts if "accuracy" in a.metric_name]
        if len(performance_alerts) > 0:
            recommendations.append("Performance degradation detected - check for data drift")
        
        return recommendations


# CLI interface
def main():
    """CLI interface for bias monitoring operations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Bias Monitor CLI")
    parser.add_argument("command", choices=["status", "history", "simulate"])
    parser.add_argument("--model-name", required=True, help="Model name")
    parser.add_argument("--hours", type=int, default=24, help="Hours of history to show")
    
    args = parser.parse_args()
    
    monitor = BiasMonitor()
    
    if args.command == "status":
        status = monitor.get_monitoring_status(args.model_name)
        print("Monitoring Status:")
        print(json.dumps(status, indent=2, default=str))
    
    elif args.command == "history":
        start_time = datetime.utcnow() - timedelta(hours=args.hours)
        history = monitor.get_historical_data(args.model_name, start_time)
        print(f"Historical Data ({args.hours} hours):")
        print(json.dumps(history, indent=2, default=str))
    
    elif args.command == "simulate":
        # Simulate some predictions for testing
        print(f"Simulating predictions for {args.model_name}...")
        
        for i in range(100):
            monitor.add_prediction(
                model_name=args.model_name,
                prediction=np.random.randint(0, 2),
                true_label=np.random.randint(0, 2),
                protected_attribute=np.random.choice(['A', 'B']),
                features={"feature_1": np.random.normal()}
            )
        
        # Process monitoring window
        window = monitor.process_monitoring_window(args.model_name)
        if window:
            print("Monitoring window processed:")
            print(json.dumps(window.to_dict(), indent=2, default=str))
        
        # Show status
        status = monitor.get_monitoring_status(args.model_name)
        print("Current status:")
        print(json.dumps(status, indent=2, default=str))


if __name__ == "__main__":
    main()