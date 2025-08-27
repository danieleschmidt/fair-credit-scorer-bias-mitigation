#!/usr/bin/env python3
"""
Comprehensive test suite for robust systems components.
Validates autonomous error recovery and comprehensive observability.
"""

import pytest
import numpy as np
import pandas as pd
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import tempfile
import json

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from robust_systems.autonomous_error_recovery import (
    ErrorPatternDetector,
    RecoveryManager,
    AutonomousErrorRecovery,
    demonstrate_autonomous_error_recovery
)
from monitoring.comprehensive_observability import (
    MetricCollector,
    AnomalyDetector,
    AlertManager,
    ObservabilityDashboard,
    demonstrate_comprehensive_observability
)


class TestErrorPatternDetector:
    """Test the ErrorPatternDetector component."""
    
    def test_detector_initialization(self):
        """Test detector initializes with correct parameters."""
        detector = ErrorPatternDetector(
            pattern_memory_size=100,
            similarity_threshold=0.8
        )
        assert detector.pattern_memory_size == 100
        assert detector.similarity_threshold == 0.8
        assert len(detector.error_patterns) == 0
    
    def test_error_recording(self):
        """Test recording error patterns."""
        detector = ErrorPatternDetector()
        
        error_info = {
            'error_type': 'ValueError',
            'error_message': 'Invalid input shape',
            'context': {'function': 'predict', 'input_shape': (10, 5)},
            'timestamp': datetime.now(),
            'fairness_impact': 0.3
        }
        
        detector.record_error_pattern(error_info)
        assert len(detector.error_patterns) == 1
        assert detector.error_patterns[0]['error_type'] == 'ValueError'
    
    def test_pattern_prediction(self):
        """Test predicting error patterns."""
        detector = ErrorPatternDetector()
        
        # Record several similar errors
        for i in range(3):
            error_info = {
                'error_type': 'ValueError',
                'error_message': f'Invalid input shape {i}',
                'context': {'function': 'predict'},
                'timestamp': datetime.now(),
                'fairness_impact': 0.2 + i*0.1
            }
            detector.record_error_pattern(error_info)
        
        # Test prediction
        current_context = {'function': 'predict', 'model': 'classifier'}
        prediction = detector.predict_error_likelihood(current_context)
        
        assert 'error_probability' in prediction
        assert 'predicted_patterns' in prediction
        assert 0 <= prediction['error_probability'] <= 1
    
    def test_pattern_similarity(self):
        """Test error pattern similarity calculation."""
        detector = ErrorPatternDetector()
        
        pattern1 = {
            'error_type': 'ValueError',
            'context': {'function': 'predict', 'model': 'svm'}
        }
        pattern2 = {
            'error_type': 'ValueError',
            'context': {'function': 'predict', 'model': 'rf'}
        }
        pattern3 = {
            'error_type': 'TypeError',
            'context': {'function': 'train', 'model': 'svm'}
        }
        
        # Similar patterns should have high similarity
        sim1 = detector._calculate_pattern_similarity(pattern1, pattern2)
        assert sim1 > 0.5
        
        # Different patterns should have lower similarity
        sim2 = detector._calculate_pattern_similarity(pattern1, pattern3)
        assert sim2 < sim1


class TestRecoveryManager:
    """Test the RecoveryManager component."""
    
    def test_manager_initialization(self):
        """Test recovery manager initializes correctly."""
        manager = RecoveryManager()
        assert hasattr(manager, 'recovery_strategies')
        assert hasattr(manager, 'fairness_preservation_rules')
        assert len(manager.recovery_strategies) > 0
    
    @patch('robust_systems.autonomous_error_recovery.RecoveryManager._rollback_to_safe_state')
    def test_error_recovery_execution(self, mock_rollback):
        """Test executing error recovery."""
        manager = RecoveryManager()
        mock_rollback.return_value = {'status': 'success', 'action': 'rollback'}
        
        error_info = {
            'error_type': 'ValueError',
            'severity': 'high',
            'fairness_impact': 0.7,
            'context': {'function': 'predict'}
        }
        
        recovery_result = manager.execute_recovery(error_info)
        
        assert 'recovery_actions' in recovery_result
        assert 'fairness_preserved' in recovery_result
        assert isinstance(recovery_result['recovery_actions'], list)
    
    def test_fairness_preservation_during_recovery(self):
        """Test fairness is preserved during recovery actions."""
        manager = RecoveryManager()
        
        error_with_high_fairness_impact = {
            'error_type': 'BiasError',
            'fairness_impact': 0.9,
            'context': {'protected_groups': ['gender', 'race']}
        }
        
        recovery_plan = manager._plan_recovery_strategy(error_with_high_fairness_impact)
        
        # Should include fairness preservation actions
        assert any('fairness' in action.lower() for action in recovery_plan)
        assert len(recovery_plan) > 0
    
    def test_recovery_strategy_selection(self):
        """Test appropriate recovery strategy selection."""
        manager = RecoveryManager()
        
        # Test different error severities
        low_severity_error = {'severity': 'low', 'fairness_impact': 0.1}
        high_severity_error = {'severity': 'high', 'fairness_impact': 0.8}
        
        low_strategy = manager._select_recovery_strategy(low_severity_error)
        high_strategy = manager._select_recovery_strategy(high_severity_error)
        
        # High severity should trigger more comprehensive recovery
        assert high_strategy != low_strategy
        assert len(high_strategy) >= len(low_strategy)


class TestAutonomousErrorRecovery:
    """Test the AutonomousErrorRecovery orchestrator."""
    
    def test_recovery_system_initialization(self):
        """Test the full recovery system initializes."""
        recovery_system = AutonomousErrorRecovery()
        assert hasattr(recovery_system, 'error_detector')
        assert hasattr(recovery_system, 'recovery_manager')
        assert hasattr(recovery_system, 'monitoring_enabled')
    
    @pytest.mark.asyncio
    async def test_continuous_monitoring(self):
        """Test continuous error monitoring."""
        recovery_system = AutonomousErrorRecovery()
        
        # Mock the monitoring loop to run briefly
        with patch.object(recovery_system, '_monitoring_loop') as mock_loop:
            mock_loop.return_value = None
            
            # Start monitoring
            recovery_system.start_autonomous_monitoring()
            assert recovery_system.monitoring_enabled == True
            
            # Stop monitoring
            recovery_system.stop_autonomous_monitoring()
            assert recovery_system.monitoring_enabled == False
    
    def test_error_handling_integration(self):
        """Test integrated error handling."""
        recovery_system = AutonomousErrorRecovery()
        
        # Simulate an error
        error_context = {
            'error_type': 'ModelDegradationError',
            'error_message': 'Model performance below threshold',
            'fairness_metrics': {'demographic_parity': 0.6},
            'timestamp': datetime.now()
        }
        
        result = recovery_system.handle_error(error_context)
        
        assert 'error_handled' in result
        assert 'recovery_actions_taken' in result
        assert 'fairness_impact_mitigated' in result
    
    def test_demonstrate_recovery_system(self):
        """Test the demonstration function."""
        result = demonstrate_autonomous_error_recovery()
        
        assert result is not None
        assert 'error_recovery_system' in result
        assert 'recovery_demonstrations' in result


class TestMetricCollector:
    """Test the MetricCollector component."""
    
    def test_collector_initialization(self):
        """Test metric collector initializes correctly."""
        collector = MetricCollector()
        assert hasattr(collector, 'metrics_buffer')
        assert hasattr(collector, 'fairness_trackers')
        assert isinstance(collector.metrics_buffer, dict)
    
    def test_fairness_metrics_collection(self):
        """Test collecting fairness metrics."""
        collector = MetricCollector()
        
        # Sample predictions and protected attributes
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        protected_attr = np.array([0, 0, 1, 1, 0, 1])
        
        metrics = collector.collect_fairness_metrics(y_true, y_pred, protected_attr)
        
        assert 'demographic_parity' in metrics
        assert 'equalized_odds' in metrics
        assert 'statistical_parity' in metrics
        assert all(isinstance(v, (int, float)) for v in metrics.values())
    
    def test_performance_metrics_collection(self):
        """Test collecting performance metrics."""
        collector = MetricCollector()
        
        # Mock system metrics
        with patch('psutil.cpu_percent', return_value=45.0), \
             patch('psutil.virtual_memory') as mock_memory:
            
            mock_memory.return_value.percent = 60.0
            
            metrics = collector.collect_system_metrics()
            
            assert 'cpu_usage' in metrics
            assert 'memory_usage' in metrics
            assert metrics['cpu_usage'] == 45.0
            assert metrics['memory_usage'] == 60.0
    
    def test_metrics_aggregation(self):
        """Test metrics aggregation over time windows."""
        collector = MetricCollector()
        
        # Add sample metrics over time
        for i in range(10):
            metrics = {
                'accuracy': 0.8 + i*0.01,
                'fairness_score': 0.9 - i*0.005,
                'timestamp': datetime.now() - timedelta(minutes=i)
            }
            collector.store_metrics(metrics)
        
        # Test aggregation
        aggregated = collector.aggregate_metrics(window_minutes=5)
        
        assert 'accuracy_mean' in aggregated
        assert 'fairness_score_mean' in aggregated
        assert 'metrics_count' in aggregated


class TestAnomalyDetector:
    """Test the AnomalyDetector component."""
    
    def test_detector_initialization(self):
        """Test anomaly detector initializes."""
        detector = AnomalyDetector()
        assert hasattr(detector, 'baseline_metrics')
        assert hasattr(detector, 'anomaly_thresholds')
        assert hasattr(detector, 'detection_models')
    
    def test_baseline_establishment(self):
        """Test establishing baseline metrics."""
        detector = AnomalyDetector()
        
        # Generate sample baseline data
        baseline_data = []
        for i in range(100):
            baseline_data.append({
                'accuracy': 0.85 + np.random.normal(0, 0.02),
                'fairness_score': 0.90 + np.random.normal(0, 0.01),
                'response_time': 50 + np.random.normal(0, 5)
            })
        
        detector.establish_baseline(baseline_data)
        
        assert 'accuracy' in detector.baseline_metrics
        assert 'fairness_score' in detector.baseline_metrics
        assert 'response_time' in detector.baseline_metrics
    
    def test_anomaly_detection(self):
        """Test detecting anomalies in metrics."""
        detector = AnomalyDetector()
        
        # Establish baseline first
        baseline_data = [
            {'accuracy': 0.85, 'fairness_score': 0.90},
            {'accuracy': 0.86, 'fairness_score': 0.89},
            {'accuracy': 0.84, 'fairness_score': 0.91}
        ]
        detector.establish_baseline(baseline_data)
        
        # Test normal metrics (should not be anomalous)
        normal_metrics = {'accuracy': 0.85, 'fairness_score': 0.90}
        normal_result = detector.detect_anomalies(normal_metrics)
        assert not normal_result['anomaly_detected']
        
        # Test anomalous metrics
        anomalous_metrics = {'accuracy': 0.60, 'fairness_score': 0.70}
        anomaly_result = detector.detect_anomalies(anomalous_metrics)
        assert anomaly_result['anomaly_detected']
        assert 'anomalous_metrics' in anomaly_result
    
    def test_fairness_anomaly_detection(self):
        """Test specific fairness anomaly detection."""
        detector = AnomalyDetector()
        
        # Baseline with good fairness
        baseline = [{'demographic_parity': 0.95, 'equalized_odds': 0.93}] * 10
        detector.establish_baseline(baseline)
        
        # Test fairness degradation
        degraded_fairness = {'demographic_parity': 0.70, 'equalized_odds': 0.65}
        result = detector.detect_fairness_anomalies(degraded_fairness)
        
        assert result['fairness_anomaly_detected']
        assert 'affected_metrics' in result


class TestAlertManager:
    """Test the AlertManager component."""
    
    def test_manager_initialization(self):
        """Test alert manager initializes."""
        alert_manager = AlertManager()
        assert hasattr(alert_manager, 'alert_channels')
        assert hasattr(alert_manager, 'alert_history')
        assert hasattr(alert_manager, 'severity_thresholds')
    
    def test_alert_generation(self):
        """Test generating alerts."""
        alert_manager = AlertManager()
        
        anomaly_info = {
            'anomaly_type': 'fairness_degradation',
            'severity': 'high',
            'affected_metrics': ['demographic_parity'],
            'timestamp': datetime.now(),
            'details': {'old_value': 0.9, 'new_value': 0.6}
        }
        
        alert = alert_manager.generate_alert(anomaly_info)
        
        assert 'alert_id' in alert
        assert 'severity' in alert
        assert 'message' in alert
        assert 'timestamp' in alert
    
    def test_alert_routing(self):
        """Test alert routing based on severity."""
        alert_manager = AlertManager()
        
        # High severity alert
        high_severity_alert = {
            'severity': 'critical',
            'message': 'Critical fairness violation detected'
        }
        
        # Low severity alert
        low_severity_alert = {
            'severity': 'info',
            'message': 'Minor performance fluctuation'
        }
        
        high_channels = alert_manager._determine_alert_channels(high_severity_alert)
        low_channels = alert_manager._determine_alert_channels(low_severity_alert)
        
        # Critical alerts should go to more channels
        assert len(high_channels) >= len(low_channels)
    
    @patch('robust_systems.autonomous_error_recovery.AlertManager._send_email_alert')
    def test_alert_delivery(self, mock_send_email):
        """Test alert delivery mechanism."""
        alert_manager = AlertManager()
        mock_send_email.return_value = {'status': 'sent'}
        
        alert = {
            'alert_id': 'test_123',
            'severity': 'high',
            'message': 'Test alert',
            'channels': ['email', 'slack']
        }
        
        delivery_result = alert_manager.send_alert(alert)
        
        assert 'delivery_results' in delivery_result
        assert len(delivery_result['delivery_results']) > 0


class TestObservabilityDashboard:
    """Test the ObservabilityDashboard component."""
    
    def test_dashboard_initialization(self):
        """Test dashboard initializes with required components."""
        dashboard = ObservabilityDashboard()
        assert hasattr(dashboard, 'metric_collector')
        assert hasattr(dashboard, 'anomaly_detector')
        assert hasattr(dashboard, 'alert_manager')
    
    def test_real_time_monitoring(self):
        """Test real-time monitoring capabilities."""
        dashboard = ObservabilityDashboard()
        
        # Mock current system state
        with patch.object(dashboard.metric_collector, 'collect_all_metrics') as mock_collect:
            mock_collect.return_value = {
                'accuracy': 0.85,
                'fairness_score': 0.90,
                'cpu_usage': 45.0,
                'memory_usage': 60.0
            }
            
            status = dashboard.get_system_status()
            
            assert 'current_metrics' in status
            assert 'system_health' in status
            assert 'timestamp' in status
    
    def test_historical_reporting(self):
        """Test historical data reporting."""
        dashboard = ObservabilityDashboard()
        
        # Mock historical data
        with patch.object(dashboard.metric_collector, 'get_historical_metrics') as mock_history:
            mock_history.return_value = [
                {'timestamp': '2024-01-01', 'accuracy': 0.85},
                {'timestamp': '2024-01-02', 'accuracy': 0.86}
            ]
            
            report = dashboard.generate_performance_report(
                start_time=datetime.now() - timedelta(days=7),
                end_time=datetime.now()
            )
            
            assert 'metrics_summary' in report
            assert 'trends' in report
            assert 'anomalies' in report
    
    def test_demonstrate_observability(self):
        """Test the demonstration function."""
        result = demonstrate_comprehensive_observability()
        
        assert result is not None
        assert 'observability_dashboard' in result
        assert 'monitoring_results' in result


class TestIntegrationScenarios:
    """Test integration between robust system components."""
    
    def test_error_recovery_with_observability(self):
        """Test integration between error recovery and observability."""
        # Initialize systems
        recovery_system = AutonomousErrorRecovery()
        dashboard = ObservabilityDashboard()
        
        # Simulate error detection through observability
        anomaly_metrics = {
            'accuracy': 0.60,  # Below normal
            'fairness_score': 0.70,  # Degraded fairness
            'timestamp': datetime.now()
        }
        
        # Detect anomaly
        anomaly_result = dashboard.anomaly_detector.detect_anomalies(anomaly_metrics)
        
        if anomaly_result.get('anomaly_detected'):
            # Convert to error context for recovery
            error_context = {
                'error_type': 'PerformanceDegradation',
                'error_message': 'Anomalous metrics detected',
                'metrics': anomaly_metrics,
                'timestamp': datetime.now()
            }
            
            # Execute recovery
            recovery_result = recovery_system.handle_error(error_context)
            
            assert recovery_result['error_handled']
            assert 'recovery_actions_taken' in recovery_result
    
    def test_end_to_end_robust_pipeline(self):
        """Test complete robust systems pipeline."""
        # Run demonstrations
        recovery_result = demonstrate_autonomous_error_recovery()
        observability_result = demonstrate_comprehensive_observability()
        
        # Validate results
        assert recovery_result is not None
        assert observability_result is not None
        
        # Test system integration
        recovery_system = recovery_result['error_recovery_system']
        dashboard = observability_result['observability_dashboard']
        
        assert hasattr(recovery_system, 'error_detector')
        assert hasattr(dashboard, 'metric_collector')


if __name__ == "__main__":
    # Run comprehensive tests
    pytest.main([__file__, "-v", "--tb=short"])