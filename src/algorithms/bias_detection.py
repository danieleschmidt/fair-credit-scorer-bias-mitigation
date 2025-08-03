"""
Real-time bias detection algorithms.

Advanced algorithms for detecting bias drift, distribution shifts,
and fairness violations in production machine learning systems.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from collections import deque
from datetime import datetime, timedelta
import warnings

from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope

from ..logging_config import get_logger
from ..fairness_metrics import compute_fairness_metrics

logger = get_logger(__name__)


class BiasType(Enum):
    """Types of bias that can be detected."""
    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUALIZED_ODDS = "equalized_odds"
    CALIBRATION = "calibration"
    REPRESENTATION = "representation"


class DriftType(Enum):
    """Types of data drift."""
    FEATURE_DRIFT = "feature_drift"
    LABEL_DRIFT = "label_drift"
    PREDICTION_DRIFT = "prediction_drift"
    CONCEPT_DRIFT = "concept_drift"


@dataclass
class BiasAlert:
    """Bias detection alert."""
    timestamp: datetime
    bias_type: BiasType
    severity: str
    metric_value: float
    threshold: float
    affected_groups: List[str]
    description: str
    recommendations: List[str]
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'bias_type': self.bias_type.value,
            'severity': self.severity,
            'metric_value': self.metric_value,
            'threshold': self.threshold,
            'affected_groups': self.affected_groups,
            'description': self.description,
            'recommendations': self.recommendations,
            'confidence': self.confidence
        }


@dataclass
class DriftAlert:
    """Data drift detection alert."""
    timestamp: datetime
    drift_type: DriftType
    severity: str
    drift_score: float
    threshold: float
    affected_features: List[str]
    statistical_test: str
    p_value: float
    description: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'drift_type': self.drift_type.value,
            'severity': self.severity,
            'drift_score': self.drift_score,
            'threshold': self.threshold,
            'affected_features': self.affected_features,
            'statistical_test': self.statistical_test,
            'p_value': self.p_value,
            'description': self.description
        }


class RealTimeBiasDetector:
    """
    Real-time bias detection system for production ML models.
    
    Monitors predictions and detects various types of bias in real-time
    using sliding windows and statistical tests.
    """
    
    def __init__(
        self,
        window_size: int = 1000,
        alert_thresholds: Optional[Dict[str, float]] = None,
        protected_attributes: Optional[List[str]] = None
    ):
        """
        Initialize bias detector.
        
        Args:
            window_size: Size of sliding window for analysis
            alert_thresholds: Thresholds for different bias metrics
            protected_attributes: List of protected attribute names
        """
        self.window_size = window_size
        self.protected_attributes = protected_attributes or []
        
        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'demographic_parity_difference': 0.1,
            'equalized_odds_difference': 0.1,
            'calibration_error': 0.05,
            'representation_imbalance': 0.3
        }
        
        # Sliding windows for different data types
        self.prediction_window = deque(maxlen=window_size)
        self.true_label_window = deque(maxlen=window_size)
        self.protected_window = deque(maxlen=window_size)
        self.feature_window = deque(maxlen=window_size)
        self.timestamp_window = deque(maxlen=window_size)
        
        # Alert history
        self.alerts: List[BiasAlert] = []
        
        # Baseline statistics (computed from reference data)
        self.baseline_stats: Optional[Dict[str, Any]] = None
        
        logger.info("RealTimeBiasDetector initialized")
    
    def add_prediction(
        self,
        prediction: float,
        true_label: Optional[float],
        protected_attributes: Dict[str, Any],
        features: Optional[Dict[str, float]] = None,
        timestamp: Optional[datetime] = None
    ):
        """
        Add a new prediction to the monitoring system.
        
        Args:
            prediction: Model prediction (probability or class)
            true_label: True label (if available)
            protected_attributes: Protected attribute values
            features: Feature values
            timestamp: Prediction timestamp
        """
        timestamp = timestamp or datetime.utcnow()
        
        # Add to sliding windows
        self.prediction_window.append(prediction)
        self.true_label_window.append(true_label)
        self.protected_window.append(protected_attributes)
        self.feature_window.append(features or {})
        self.timestamp_window.append(timestamp)
        
        # Trigger bias detection if window is full
        if len(self.prediction_window) >= self.window_size:
            self._detect_bias()
    
    def set_baseline(self, reference_data: pd.DataFrame, target_column: str):
        """
        Set baseline statistics from reference data.
        
        Args:
            reference_data: Reference dataset
            target_column: Target column name
        """
        logger.info("Computing baseline statistics")
        
        self.baseline_stats = {}
        
        # Compute overall statistics
        self.baseline_stats['overall'] = {
            'target_rate': reference_data[target_column].mean(),
            'feature_means': reference_data.select_dtypes(include=[np.number]).mean().to_dict(),
            'feature_stds': reference_data.select_dtypes(include=[np.number]).std().to_dict()
        }
        
        # Compute group-specific statistics
        for attr in self.protected_attributes:
            if attr in reference_data.columns:
                group_stats = {}
                for group_value in reference_data[attr].unique():
                    group_data = reference_data[reference_data[attr] == group_value]
                    group_stats[group_value] = {
                        'size': len(group_data),
                        'target_rate': group_data[target_column].mean(),
                        'proportion': len(group_data) / len(reference_data)
                    }
                
                self.baseline_stats[attr] = group_stats
        
        logger.info("Baseline statistics computed")
    
    def _detect_bias(self):
        """Detect bias in current window."""
        try:
            # Convert windows to pandas for analysis
            current_data = self._windows_to_dataframe()
            
            if current_data.empty or current_data['true_label'].isnull().all():
                return  # Need true labels for bias detection
            
            # Clean data
            current_data = current_data.dropna(subset=['true_label'])
            
            if len(current_data) < 10:
                return  # Need minimum sample size
            
            # Detect different types of bias
            self._detect_demographic_parity_bias(current_data)
            self._detect_equalized_odds_bias(current_data)
            self._detect_calibration_bias(current_data)
            self._detect_representation_bias(current_data)
            
        except Exception as e:
            logger.error(f"Bias detection failed: {e}")
    
    def _detect_demographic_parity_bias(self, data: pd.DataFrame):
        """Detect demographic parity violations."""
        for attr in self.protected_attributes:
            if attr not in data.columns:
                continue
            
            try:
                # Compute demographic parity difference
                overall, by_group = compute_fairness_metrics(
                    data['true_label'],
                    data['prediction'] > 0.5,  # Binary predictions
                    data[attr]
                )
                
                dp_diff = abs(overall['demographic_parity_difference'])
                threshold = self.alert_thresholds['demographic_parity_difference']
                
                if dp_diff > threshold:
                    # Identify most affected groups
                    selection_rates = by_group['selection_rate']
                    max_rate = selection_rates.max()
                    min_rate = selection_rates.min()
                    
                    affected_groups = [
                        str(group) for group, rate in selection_rates.items()
                        if rate == min_rate or rate == max_rate
                    ]
                    
                    alert = BiasAlert(
                        timestamp=datetime.utcnow(),
                        bias_type=BiasType.DEMOGRAPHIC_PARITY,
                        severity=self._calculate_severity(dp_diff, threshold),
                        metric_value=dp_diff,
                        threshold=threshold,
                        affected_groups=affected_groups,
                        description=f"Demographic parity violation for {attr}: {dp_diff:.3f}",
                        recommendations=self._get_demographic_parity_recommendations(),
                        confidence=self._calculate_confidence(len(data))
                    )
                    
                    self.alerts.append(alert)
                    logger.warning(f"Demographic parity bias detected: {alert.description}")
                    
            except Exception as e:
                logger.error(f"Demographic parity detection failed for {attr}: {e}")
    
    def _detect_equalized_odds_bias(self, data: pd.DataFrame):
        """Detect equalized odds violations."""
        for attr in self.protected_attributes:
            if attr not in data.columns:
                continue
            
            try:
                overall, by_group = compute_fairness_metrics(
                    data['true_label'],
                    data['prediction'] > 0.5,
                    data[attr]
                )
                
                eo_diff = abs(overall['equalized_odds_difference'])
                threshold = self.alert_thresholds['equalized_odds_difference']
                
                if eo_diff > threshold:
                    # Analyze TPR and FPR differences
                    tpr_by_group = by_group['true_positive_rate']
                    fpr_by_group = by_group['false_positive_rate']
                    
                    alert = BiasAlert(
                        timestamp=datetime.utcnow(),
                        bias_type=BiasType.EQUALIZED_ODDS,
                        severity=self._calculate_severity(eo_diff, threshold),
                        metric_value=eo_diff,
                        threshold=threshold,
                        affected_groups=list(tpr_by_group.index.astype(str)),
                        description=f"Equalized odds violation for {attr}: {eo_diff:.3f}",
                        recommendations=self._get_equalized_odds_recommendations(),
                        confidence=self._calculate_confidence(len(data))
                    )
                    
                    self.alerts.append(alert)
                    logger.warning(f"Equalized odds bias detected: {alert.description}")
                    
            except Exception as e:
                logger.error(f"Equalized odds detection failed for {attr}: {e}")
    
    def _detect_calibration_bias(self, data: pd.DataFrame):
        """Detect calibration bias across groups."""
        for attr in self.protected_attributes:
            if attr not in data.columns:
                continue
            
            try:
                # Compute calibration error for each group
                calibration_errors = {}
                
                for group_value in data[attr].unique():
                    group_data = data[data[attr] == group_value]
                    
                    if len(group_data) < 10:
                        continue
                    
                    # Bin predictions and compute calibration error
                    bins = np.linspace(0, 1, 11)
                    bin_indices = np.digitize(group_data['prediction'], bins) - 1
                    
                    calibration_error = 0
                    for i in range(len(bins) - 1):
                        bin_mask = bin_indices == i
                        if bin_mask.sum() > 0:
                            bin_predictions = group_data['prediction'][bin_mask]
                            bin_true_labels = group_data['true_label'][bin_mask]
                            
                            avg_prediction = bin_predictions.mean()
                            true_fraction = bin_true_labels.mean()
                            
                            calibration_error += abs(avg_prediction - true_fraction) * bin_mask.sum()
                    
                    calibration_errors[group_value] = calibration_error / len(group_data)
                
                # Check for significant differences in calibration
                if len(calibration_errors) >= 2:
                    max_error = max(calibration_errors.values())
                    threshold = self.alert_thresholds['calibration_error']
                    
                    if max_error > threshold:
                        alert = BiasAlert(
                            timestamp=datetime.utcnow(),
                            bias_type=BiasType.CALIBRATION,
                            severity=self._calculate_severity(max_error, threshold),
                            metric_value=max_error,
                            threshold=threshold,
                            affected_groups=list(calibration_errors.keys()),
                            description=f"Calibration bias detected for {attr}: {max_error:.3f}",
                            recommendations=self._get_calibration_recommendations(),
                            confidence=self._calculate_confidence(len(data))
                        )
                        
                        self.alerts.append(alert)
                        logger.warning(f"Calibration bias detected: {alert.description}")
                        
            except Exception as e:
                logger.error(f"Calibration detection failed for {attr}: {e}")
    
    def _detect_representation_bias(self, data: pd.DataFrame):
        """Detect representation imbalances."""
        for attr in self.protected_attributes:
            if attr not in data.columns:
                continue
            
            try:
                # Compute group proportions
                group_counts = data[attr].value_counts()
                group_proportions = group_counts / len(data)
                
                # Compare with baseline if available
                if self.baseline_stats and attr in self.baseline_stats:
                    baseline_proportions = {
                        k: v['proportion'] 
                        for k, v in self.baseline_stats[attr].items()
                    }
                    
                    # Calculate representation drift
                    max_drift = 0
                    for group_value in group_proportions.index:
                        if group_value in baseline_proportions:
                            current_prop = group_proportions[group_value]
                            baseline_prop = baseline_proportions[group_value]
                            drift = abs(current_prop - baseline_prop)
                            max_drift = max(max_drift, drift)
                    
                    threshold = self.alert_thresholds['representation_imbalance']
                    
                    if max_drift > threshold:
                        alert = BiasAlert(
                            timestamp=datetime.utcnow(),
                            bias_type=BiasType.REPRESENTATION,
                            severity=self._calculate_severity(max_drift, threshold),
                            metric_value=max_drift,
                            threshold=threshold,
                            affected_groups=list(group_proportions.index.astype(str)),
                            description=f"Representation bias for {attr}: {max_drift:.3f}",
                            recommendations=self._get_representation_recommendations(),
                            confidence=self._calculate_confidence(len(data))
                        )
                        
                        self.alerts.append(alert)
                        logger.warning(f"Representation bias detected: {alert.description}")
                        
            except Exception as e:
                logger.error(f"Representation detection failed for {attr}: {e}")
    
    def _windows_to_dataframe(self) -> pd.DataFrame:
        """Convert sliding windows to DataFrame for analysis."""
        data = []
        
        for i in range(len(self.prediction_window)):
            row = {
                'prediction': self.prediction_window[i],
                'true_label': self.true_label_window[i],
                'timestamp': self.timestamp_window[i]
            }
            
            # Add protected attributes
            if i < len(self.protected_window):
                protected = self.protected_window[i]
                if protected:
                    row.update(protected)
            
            # Add features
            if i < len(self.feature_window):
                features = self.feature_window[i]
                if features:
                    row.update(features)
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _calculate_severity(self, value: float, threshold: float) -> str:
        """Calculate alert severity based on threshold exceedance."""
        ratio = value / threshold
        
        if ratio >= 3.0:
            return "critical"
        elif ratio >= 2.0:
            return "high"
        elif ratio >= 1.5:
            return "medium"
        else:
            return "low"
    
    def _calculate_confidence(self, sample_size: int) -> float:
        """Calculate confidence based on sample size."""
        # Simple confidence calculation based on sample size
        if sample_size >= 1000:
            return 0.95
        elif sample_size >= 500:
            return 0.90
        elif sample_size >= 100:
            return 0.80
        elif sample_size >= 50:
            return 0.70
        else:
            return 0.60
    
    def _get_demographic_parity_recommendations(self) -> List[str]:
        """Get recommendations for demographic parity violations."""
        return [
            "Apply demographic parity constraints during model training",
            "Use preprocessing techniques like reweighting or resampling",
            "Implement post-processing threshold optimization",
            "Review feature selection for potential proxy variables",
            "Consider fairness-aware ensemble methods"
        ]
    
    def _get_equalized_odds_recommendations(self) -> List[str]:
        """Get recommendations for equalized odds violations."""
        return [
            "Apply equalized odds constraints during training",
            "Use fairness-aware loss functions",
            "Implement group-specific threshold optimization",
            "Consider adversarial debiasing techniques",
            "Review model complexity and regularization"
        ]
    
    def _get_calibration_recommendations(self) -> List[str]:
        """Get recommendations for calibration bias."""
        return [
            "Apply group-specific calibration techniques",
            "Use isotonic regression for probability calibration",
            "Implement Platt scaling per group",
            "Review prediction confidence across groups",
            "Consider temperature scaling adjustments"
        ]
    
    def _get_representation_recommendations(self) -> List[str]:
        """Get recommendations for representation bias."""
        return [
            "Review data collection and sampling procedures",
            "Implement stratified sampling strategies",
            "Monitor data sources for representation drift",
            "Consider synthetic data generation for underrepresented groups",
            "Adjust model training to account for group imbalances"
        ]
    
    def get_recent_alerts(self, hours: int = 24) -> List[BiasAlert]:
        """Get alerts from the last N hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [alert for alert in self.alerts if alert.timestamp >= cutoff_time]
    
    def get_bias_summary(self) -> Dict[str, Any]:
        """Get summary of bias detection results."""
        recent_alerts = self.get_recent_alerts()
        
        summary = {
            'total_alerts': len(recent_alerts),
            'alerts_by_type': {},
            'alerts_by_severity': {},
            'most_affected_attributes': {},
            'window_size': self.window_size,
            'current_window_length': len(self.prediction_window)
        }
        
        # Count by type
        for bias_type in BiasType:
            count = len([a for a in recent_alerts if a.bias_type == bias_type])
            summary['alerts_by_type'][bias_type.value] = count
        
        # Count by severity
        for severity in ['low', 'medium', 'high', 'critical']:
            count = len([a for a in recent_alerts if a.severity == severity])
            summary['alerts_by_severity'][severity] = count
        
        # Most affected attributes
        attr_counts = {}
        for alert in recent_alerts:
            for group in alert.affected_groups:
                attr_counts[group] = attr_counts.get(group, 0) + 1
        
        summary['most_affected_attributes'] = dict(
            sorted(attr_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        )
        
        return summary


class DriftDetectionAlgorithm:
    """
    Statistical drift detection for features and predictions.
    
    Implements multiple statistical tests for detecting different
    types of data drift in machine learning systems.
    """
    
    def __init__(
        self,
        reference_window_size: int = 1000,
        detection_window_size: int = 100,
        alpha: float = 0.05
    ):
        """
        Initialize drift detection algorithm.
        
        Args:
            reference_window_size: Size of reference window
            detection_window_size: Size of detection window
            alpha: Significance level for statistical tests
        """
        self.reference_window_size = reference_window_size
        self.detection_window_size = detection_window_size
        self.alpha = alpha
        
        # Reference data storage
        self.reference_data: Optional[pd.DataFrame] = None
        self.reference_labels: Optional[pd.Series] = None
        
        # Detection windows
        self.current_data = deque(maxlen=detection_window_size)
        self.current_labels = deque(maxlen=detection_window_size)
        
        # Alert history
        self.drift_alerts: List[DriftAlert] = []
        
        logger.info("DriftDetectionAlgorithm initialized")
    
    def set_reference_data(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Set reference data for drift detection."""
        # Sample if data is too large
        if len(X) > self.reference_window_size:
            sample_indices = np.random.choice(
                len(X), self.reference_window_size, replace=False
            )
            self.reference_data = X.iloc[sample_indices].copy()
            if y is not None:
                self.reference_labels = y.iloc[sample_indices].copy()
        else:
            self.reference_data = X.copy()
            self.reference_labels = y.copy() if y is not None else None
        
        logger.info(f"Reference data set: {len(self.reference_data)} samples")
    
    def add_sample(self, features: Dict[str, float], label: Optional[float] = None):
        """Add new sample for drift detection."""
        self.current_data.append(features)
        self.current_labels.append(label)
        
        # Trigger drift detection if window is full
        if len(self.current_data) >= self.detection_window_size:
            self._detect_drift()
    
    def _detect_drift(self):
        """Detect various types of drift."""
        if self.reference_data is None:
            logger.warning("No reference data set for drift detection")
            return
        
        try:
            # Convert current window to DataFrame
            current_df = pd.DataFrame(list(self.current_data))
            current_labels_series = pd.Series(list(self.current_labels))
            
            # Feature drift detection
            self._detect_feature_drift(current_df)
            
            # Label drift detection
            if self.reference_labels is not None and not current_labels_series.isnull().all():
                self._detect_label_drift(current_labels_series.dropna())
            
        except Exception as e:
            logger.error(f"Drift detection failed: {e}")
    
    def _detect_feature_drift(self, current_data: pd.DataFrame):
        """Detect drift in individual features."""
        for column in self.reference_data.columns:
            if column not in current_data.columns:
                continue
            
            try:
                ref_values = self.reference_data[column].dropna()
                current_values = current_data[column].dropna()
                
                if len(ref_values) == 0 or len(current_values) == 0:
                    continue
                
                # Determine appropriate test based on data type
                if pd.api.types.is_numeric_dtype(ref_values):
                    drift_detected, test_name, p_value, drift_score = self._ks_test(
                        ref_values, current_values
                    )
                else:
                    drift_detected, test_name, p_value, drift_score = self._chi_square_test(
                        ref_values, current_values
                    )
                
                if drift_detected:
                    alert = DriftAlert(
                        timestamp=datetime.utcnow(),
                        drift_type=DriftType.FEATURE_DRIFT,
                        severity=self._calculate_drift_severity(drift_score),
                        drift_score=drift_score,
                        threshold=self.alpha,
                        affected_features=[column],
                        statistical_test=test_name,
                        p_value=p_value,
                        description=f"Feature drift detected in {column} (p={p_value:.4f})"
                    )
                    
                    self.drift_alerts.append(alert)
                    logger.warning(f"Feature drift detected: {alert.description}")
                    
            except Exception as e:
                logger.error(f"Feature drift detection failed for {column}: {e}")
    
    def _detect_label_drift(self, current_labels: pd.Series):
        """Detect drift in label distribution."""
        try:
            ref_labels = self.reference_labels.dropna()
            
            if len(ref_labels) == 0 or len(current_labels) == 0:
                return
            
            # Use chi-square test for label distribution
            drift_detected, test_name, p_value, drift_score = self._chi_square_test(
                ref_labels, current_labels
            )
            
            if drift_detected:
                alert = DriftAlert(
                    timestamp=datetime.utcnow(),
                    drift_type=DriftType.LABEL_DRIFT,
                    severity=self._calculate_drift_severity(drift_score),
                    drift_score=drift_score,
                    threshold=self.alpha,
                    affected_features=["target"],
                    statistical_test=test_name,
                    p_value=p_value,
                    description=f"Label drift detected (p={p_value:.4f})"
                )
                
                self.drift_alerts.append(alert)
                logger.warning(f"Label drift detected: {alert.description}")
                
        except Exception as e:
            logger.error(f"Label drift detection failed: {e}")
    
    def _ks_test(self, ref_data: pd.Series, current_data: pd.Series) -> Tuple[bool, str, float, float]:
        """Perform Kolmogorov-Smirnov test for numerical data."""
        try:
            statistic, p_value = stats.ks_2samp(ref_data, current_data)
            drift_detected = p_value < self.alpha
            
            return drift_detected, "Kolmogorov-Smirnov", p_value, statistic
            
        except Exception as e:
            logger.error(f"KS test failed: {e}")
            return False, "Kolmogorov-Smirnov", 1.0, 0.0
    
    def _chi_square_test(self, ref_data: pd.Series, current_data: pd.Series) -> Tuple[bool, str, float, float]:
        """Perform chi-square test for categorical data."""
        try:
            # Get value counts
            ref_counts = ref_data.value_counts()
            current_counts = current_data.value_counts()
            
            # Align indices
            all_values = set(ref_counts.index) | set(current_counts.index)
            ref_aligned = ref_counts.reindex(all_values, fill_value=0)
            current_aligned = current_counts.reindex(all_values, fill_value=0)
            
            # Perform chi-square test
            statistic, p_value = stats.chisquare(
                current_aligned.values,
                ref_aligned.values
            )
            
            drift_detected = p_value < self.alpha
            
            return drift_detected, "Chi-square", p_value, statistic
            
        except Exception as e:
            logger.error(f"Chi-square test failed: {e}")
            return False, "Chi-square", 1.0, 0.0
    
    def _calculate_drift_severity(self, drift_score: float) -> str:
        """Calculate drift severity based on test statistic."""
        # Normalize drift score (this is a simplified approach)
        if drift_score > 0.5:
            return "critical"
        elif drift_score > 0.3:
            return "high"
        elif drift_score > 0.1:
            return "medium"
        else:
            return "low"
    
    def get_recent_drift_alerts(self, hours: int = 24) -> List[DriftAlert]:
        """Get drift alerts from the last N hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [alert for alert in self.drift_alerts if alert.timestamp >= cutoff_time]
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """Get summary of drift detection results."""
        recent_alerts = self.get_recent_drift_alerts()
        
        summary = {
            'total_alerts': len(recent_alerts),
            'alerts_by_type': {},
            'alerts_by_severity': {},
            'most_affected_features': {},
            'detection_window_size': self.detection_window_size,
            'current_window_length': len(self.current_data)
        }
        
        # Count by type
        for drift_type in DriftType:
            count = len([a for a in recent_alerts if a.drift_type == drift_type])
            summary['alerts_by_type'][drift_type.value] = count
        
        # Count by severity
        for severity in ['low', 'medium', 'high', 'critical']:
            count = len([a for a in recent_alerts if a.severity == severity])
            summary['alerts_by_severity'][severity] = count
        
        # Most affected features
        feature_counts = {}
        for alert in recent_alerts:
            for feature in alert.affected_features:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        summary['most_affected_features'] = dict(
            sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        )
        
        return summary


# CLI interface
def main():
    """CLI interface for bias detection testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Bias Detection CLI")
    parser.add_argument("command", choices=["simulate", "analyze"])
    parser.add_argument("--data", help="Data file path")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to simulate")
    parser.add_argument("--bias-level", type=float, default=0.1, help="Bias level for simulation")
    
    args = parser.parse_args()
    
    if args.command == "simulate":
        # Create bias detector
        detector = RealTimeBiasDetector(window_size=100)
        
        print(f"Simulating {args.samples} predictions with bias level {args.bias_level}")
        
        # Simulate biased predictions
        for i in range(args.samples):
            # Simulate protected attribute
            protected_attr = np.random.choice(['A', 'B'])
            
            # Introduce bias
            if protected_attr == 'A':
                prediction = np.random.beta(2, 3)  # Lower predictions for group A
            else:
                prediction = np.random.beta(3, 2)  # Higher predictions for group B
            
            true_label = np.random.binomial(1, prediction)
            
            detector.add_prediction(
                prediction=prediction,
                true_label=true_label,
                protected_attributes={'group': protected_attr}
            )
        
        # Show results
        summary = detector.get_bias_summary()
        print("Bias Detection Summary:")
        print(f"  Total alerts: {summary['total_alerts']}")
        print(f"  Alerts by type: {summary['alerts_by_type']}")
        print(f"  Alerts by severity: {summary['alerts_by_severity']}")
        
        recent_alerts = detector.get_recent_alerts()
        if recent_alerts:
            print("\nRecent Alerts:")
            for alert in recent_alerts[-3:]:  # Show last 3 alerts
                print(f"  - {alert.bias_type.value}: {alert.description}")
    
    elif args.command == "analyze":
        if not args.data:
            print("Error: --data required for analyze command")
            return
        
        # Load and analyze real data
        data = pd.read_csv(args.data)
        print(f"Analyzing data with {len(data)} samples")
        
        # This would implement real data analysis
        print("Real data analysis not implemented in demo")


if __name__ == "__main__":
    main()