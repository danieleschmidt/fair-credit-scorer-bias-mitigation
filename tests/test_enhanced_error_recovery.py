"""
Comprehensive tests for enhanced error recovery framework.
"""

import asyncio
import json
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.enhanced_error_recovery import (
    CircuitBreaker,
    ErrorCategory,
    ErrorClassifier,
    ErrorEvent,
    ErrorRecoveryManager,
    ErrorSeverity,
    FallbackModel,
    HealthMonitor,
    RecoveryStrategy,
    RetryConfig
)


class TestErrorClassifier:
    """Test error classification functionality."""
    
    def test_classify_common_errors(self):
        """Test classification of common error types."""
        classifier = ErrorClassifier()
        
        # Test ValueError classification
        error = ValueError("Invalid data shape")
        category, severity = classifier.classify_error(error)
        assert category == ErrorCategory.DATA_ERROR
        assert severity == ErrorSeverity.MEDIUM
        
        # Test KeyError classification
        error = KeyError("Column 'feature1' not found")
        category, severity = classifier.classify_error(error)
        assert category == ErrorCategory.DATA_ERROR
        assert severity == ErrorSeverity.HIGH
        
        # Test FileNotFoundError classification
        error = FileNotFoundError("Model file not found")
        category, severity = classifier.classify_error(error)
        assert category == ErrorCategory.SYSTEM_ERROR
        assert severity == ErrorSeverity.HIGH
    
    def test_classify_with_context(self):
        """Test error classification with context."""
        classifier = ErrorClassifier()
        
        # Test with fairness context
        error = RuntimeError("Computation failed")
        context = {'operation': 'fairness_evaluation', 'model': 'test'}
        category, severity = classifier.classify_error(error, context)
        assert category == ErrorCategory.FAIRNESS_ERROR
        assert severity == ErrorSeverity.HIGH
        
        # Test with validation context
        error = RuntimeError("Validation failed")
        context = {'validation_step': 'schema_check'}
        category, severity = classifier.classify_error(error, context)
        assert category == ErrorCategory.VALIDATION_ERROR
        assert severity == ErrorSeverity.MEDIUM
    
    def test_classify_unknown_error(self):
        """Test classification of unknown error types."""
        classifier = ErrorClassifier()
        
        # Custom exception not in rules
        class CustomError(Exception):
            pass
        
        error = CustomError("Unknown error")
        category, severity = classifier.classify_error(error)
        assert category == ErrorCategory.UNKNOWN
        assert severity == ErrorSeverity.MEDIUM


class TestRetryConfig:
    """Test retry configuration functionality."""
    
    def test_exponential_backoff_delay(self):
        """Test exponential backoff delay calculation."""
        config = RetryConfig(
            max_attempts=5,
            base_delay=1.0,
            max_delay=30.0,
            exponential_base=2.0,
            jitter=False
        )
        
        # Test delay progression
        assert config.get_delay(0) == 1.0
        assert config.get_delay(1) == 2.0
        assert config.get_delay(2) == 4.0
        assert config.get_delay(3) == 8.0
        assert config.get_delay(4) == 16.0
    
    def test_max_delay_limit(self):
        """Test maximum delay limit."""
        config = RetryConfig(
            base_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0,
            jitter=False
        )
        
        # Should cap at max_delay
        assert config.get_delay(10) == 10.0
    
    def test_jitter_effect(self):
        """Test jitter adds randomness."""
        config = RetryConfig(base_delay=1.0, jitter=True)
        
        delays = [config.get_delay(0) for _ in range(10)]
        
        # With jitter, delays should vary
        assert len(set(delays)) > 1  # Should have some variation
        
        # All delays should be positive
        assert all(d >= 0 for d in delays)


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state."""
        cb = CircuitBreaker("test", failure_threshold=3)
        
        @cb
        def successful_function():
            return "success"
        
        # Should work normally when closed
        result = successful_function()
        assert result == "success"
        assert cb.state.state == "closed"
        assert cb.state.failure_count == 0
    
    def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after threshold failures."""
        cb = CircuitBreaker("test", failure_threshold=3)
        
        @cb
        def failing_function():
            raise ValueError("Test failure")
        
        # Should fail threshold times, then open
        for i in range(3):
            with pytest.raises(ValueError):
                failing_function()
        
        # Circuit should now be open
        assert cb.state.state == "open"
        assert cb.state.failure_count == 3
        
        # Next call should raise circuit breaker exception
        with pytest.raises(Exception, match="CircuitBreaker.*is open"):
            failing_function()
    
    def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker half-open state and recovery."""
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0.1)
        
        call_count = 0
        
        @cb
        def sometimes_failing_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ValueError("Initial failures")
            return "success"
        
        # Fail twice to open circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                sometimes_failing_function()
        
        assert cb.state.state == "open"
        
        # Wait for recovery timeout
        time.sleep(0.15)
        
        # Next call should succeed and close circuit
        result = sometimes_failing_function()
        assert result == "success"
        assert cb.state.state == "closed"
        assert cb.state.failure_count == 0


class TestFallbackModel:
    """Test fallback model functionality."""
    
    def test_fallback_model_classification(self):
        """Test fallback model for classification tasks."""
        model = FallbackModel("test")
        
        # Create classification data
        X = pd.DataFrame({'feature1': [1, 2, 3, 4, 5]})
        y = pd.Series([0, 1, 0, 1, 1])  # More 1s than 0s
        
        model.fit(X, y)
        
        assert model.fitted
        assert model.fallback_prediction == 1  # Majority class
        
        # Test predictions
        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert all(pred == 1 for pred in predictions)
    
    def test_fallback_model_regression(self):
        """Test fallback model for regression tasks."""
        model = FallbackModel("test")
        
        # Create regression data
        X = pd.DataFrame({'feature1': [1, 2, 3, 4, 5]})
        y = pd.Series([10.5, 20.3, 15.7, 25.1, 18.9])
        
        model.fit(X, y)
        
        assert model.fitted
        assert abs(model.fallback_prediction - y.mean()) < 0.01
        
        # Test predictions
        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert all(abs(pred - y.mean()) < 0.01 for pred in predictions)


class TestHealthMonitor:
    """Test health monitoring functionality."""
    
    def test_health_monitor_success_tracking(self):
        """Test health monitor success tracking."""
        monitor = HealthMonitor()
        
        # Record some successes
        for i in range(10):
            monitor.record_success(response_time=0.1 + i * 0.01)
        
        status = monitor.get_health_status()
        
        assert status['health_score'] == 1.0  # No errors
        assert status['error_rate'] == 0.0
        assert status['total_operations'] == 10
        assert status['status'] == 'healthy'
        assert abs(status['average_response_time'] - 0.145) < 0.01
    
    def test_health_monitor_error_tracking(self):
        """Test health monitor error tracking."""
        monitor = HealthMonitor()
        
        # Record mixed successes and errors
        for _ in range(7):
            monitor.record_success(response_time=0.1)
        
        for _ in range(3):
            monitor.record_error()
        
        status = monitor.get_health_status()
        
        assert status['health_score'] == 0.7  # 70% success rate
        assert status['error_rate'] == 0.3
        assert status['total_operations'] == 10
        assert status['status'] == 'degraded'  # Between 0.5 and 0.8
    
    def test_health_monitor_window_trimming(self):
        """Test health monitor window trimming."""
        monitor = HealthMonitor(window_size=5)
        
        # Record more responses than window size
        for i in range(10):
            monitor.record_success(response_time=i)
        
        # Should only keep last 5 response times
        assert len(monitor.metrics['response_times']) == 5
        assert monitor.metrics['response_times'] == [5, 6, 7, 8, 9]


class TestErrorRecoveryManager:
    """Test error recovery manager functionality."""
    
    def test_retry_decorator_success_after_failures(self):
        """Test retry decorator with eventual success."""
        manager = ErrorRecoveryManager()
        
        call_count = 0
        
        @manager.with_retry(RetryConfig(max_attempts=3, base_delay=0.01))
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"Failure {call_count}")
            return "success"
        
        result = flaky_function()
        assert result == "success"
        assert call_count == 3
        
        # Check error events were logged
        error_events = [e for e in manager.error_events if e.error_type == "ValueError"]
        assert len(error_events) == 2  # Two failures before success
    
    def test_retry_decorator_all_attempts_fail(self):
        """Test retry decorator when all attempts fail."""
        manager = ErrorRecoveryManager()
        
        @manager.with_retry(RetryConfig(max_attempts=2, base_delay=0.01))
        def always_failing_function():
            raise RuntimeError("Always fails")
        
        with pytest.raises(RuntimeError, match="Always fails"):
            always_failing_function()
        
        # Check error events were logged
        error_events = [e for e in manager.error_events if e.error_type == "RuntimeError"]
        assert len(error_events) == 2  # One for each attempt
    
    def test_fallback_decorator_with_fallback_value(self):
        """Test fallback decorator with fallback value."""
        manager = ErrorRecoveryManager()
        
        @manager.with_fallback(fallback_value="fallback_result")
        def failing_function():
            raise ValueError("Function failed")
        
        result = failing_function()
        assert result == "fallback_result"
        
        # Check error was logged
        error_events = [e for e in manager.error_events if e.error_type == "ValueError"]
        assert len(error_events) == 1
        assert error_events[0].context.get('fallback_used') is True
    
    def test_fallback_decorator_with_model(self):
        """Test fallback decorator with fallback model."""
        manager = ErrorRecoveryManager()
        
        # Create and register fallback model
        fallback_model = FallbackModel("test_fallback")
        X_train = pd.DataFrame({'feature1': [1, 2, 3, 4, 5]})
        y_train = pd.Series([0, 1, 0, 1, 1])
        fallback_model.fit(X_train, y_train)
        manager.register_fallback_model("test_fallback", fallback_model)
        
        @manager.with_fallback(fallback_model_name="test_fallback")
        def failing_prediction_function(X):
            raise ValueError("Prediction failed")
        
        X_test = pd.DataFrame({'feature1': [1, 2, 3]})
        result = failing_prediction_function(X_test)
        
        assert len(result) == len(X_test)
        assert all(pred == 1 for pred in result)  # Fallback model prediction
    
    def test_resilient_wrapper_integration(self):
        """Test comprehensive resilient wrapper."""
        manager = ErrorRecoveryManager()
        
        # Register fallback model
        fallback_model = FallbackModel("test_fallback")
        X_train = pd.DataFrame({'feature1': [1, 2, 3]})
        y_train = pd.Series([1, 0, 1])
        fallback_model.fit(X_train, y_train)
        manager.register_fallback_model("test_fallback", fallback_model)
        
        call_count = 0
        
        @manager.create_resilient_wrapper(
            name="test_function",
            retry_config=RetryConfig(max_attempts=2, base_delay=0.01),
            enable_circuit_breaker=True,
            enable_fallback=True,
            fallback_model_name="test_fallback"
        )
        def flaky_prediction_function(X):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"Failure {call_count}")
            return np.array([0, 1, 0])
        
        X_test = pd.DataFrame({'feature1': [1, 2, 3]})
        
        # First call should fail and use fallback
        result = flaky_prediction_function(X_test)
        assert len(result) == len(X_test)
        
        # Reset call count and try again - should succeed on retry
        call_count = 0
        result = flaky_prediction_function(X_test)
        assert len(result) == len(X_test)
    
    def test_error_statistics_generation(self):
        """Test error statistics generation."""
        manager = ErrorRecoveryManager()
        
        # Manually add some error events
        now = datetime.now()
        
        for i in range(5):
            event = ErrorEvent(
                timestamp=now - timedelta(hours=i),
                error_type="ValueError",
                error_message=f"Error {i}",
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.DATA_ERROR,
                context={}
            )
            manager.error_events.append(event)
        
        for i in range(3):
            event = ErrorEvent(
                timestamp=now - timedelta(hours=i),
                error_type="KeyError",
                error_message=f"Key error {i}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.DATA_ERROR,
                context={}
            )
            manager.error_events.append(event)
        
        stats = manager.get_error_statistics(hours=24)
        
        assert stats['total_errors'] == 8
        assert stats['categories']['data_error'] == 8
        assert stats['severities']['medium'] == 5
        assert stats['severities']['high'] == 3
        assert ('ValueError', 5) in stats['top_errors']
        assert ('KeyError', 3) in stats['top_errors']
    
    def test_error_log_persistence(self):
        """Test error log persistence to disk."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Create manager with custom log path
            manager = ErrorRecoveryManager(error_log_path=temp_path)
            
            # Generate some errors
            @manager.with_retry(RetryConfig(max_attempts=1, base_delay=0.01))
            def failing_function():
                raise ValueError("Test error")
            
            try:
                failing_function()
            except:
                pass
            
            # Force save
            manager._save_error_log()
            
            # Check file exists and contains data
            assert Path(temp_path).exists()
            
            with open(temp_path, 'r') as f:
                log_data = json.load(f)
            
            assert len(log_data) > 0
            assert log_data[0]['error_type'] == 'ValueError'
            assert log_data[0]['error_message'] == 'Test error'
            
        finally:
            # Clean up
            if Path(temp_path).exists():
                Path(temp_path).unlink()
    
    def test_resilience_report_generation(self):
        """Test resilience report generation."""
        manager = ErrorRecoveryManager()
        
        # Add some test data
        manager.health_monitor.record_success(0.1)
        manager.health_monitor.record_success(0.2)
        manager.health_monitor.record_error()
        
        # Register some components
        manager.register_circuit_breaker("test_service")
        
        fallback_model = FallbackModel("test_model")
        manager.register_fallback_model("test_model", fallback_model)
        
        # Generate report
        report = manager.generate_resilience_report()
        
        assert "System Resilience Report" in report
        assert "Health Status" in report
        assert "Error Statistics" in report
        assert "Circuit Breakers" in report
        assert "Fallback Models" in report
        assert "test_service: closed" in report
        assert "test_model: not fitted" in report


@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration tests with realistic scenarios."""
    
    def test_ml_pipeline_resilience(self):
        """Test resilience in ML pipeline scenario."""
        manager = ErrorRecoveryManager()
        
        # Create fallback model
        fallback_model = FallbackModel("pipeline_fallback")
        X_train = pd.DataFrame({'feature1': range(100), 'feature2': range(100, 200)})
        y_train = pd.Series(np.random.binomial(1, 0.6, 100))
        fallback_model.fit(X_train, y_train)
        manager.register_fallback_model("pipeline_fallback", fallback_model)
        
        # Simulate unreliable ML prediction service
        prediction_count = 0
        
        @manager.create_resilient_wrapper(
            name="ml_prediction_service",
            retry_config=RetryConfig(max_attempts=3, base_delay=0.01),
            enable_circuit_breaker=True,
            enable_fallback=True,
            fallback_model_name="pipeline_fallback"
        )
        def predict_with_failures(X):
            nonlocal prediction_count
            prediction_count += 1
            
            # Fail 30% of the time
            if np.random.random() < 0.3:
                raise RuntimeError("Service temporarily unavailable")
            
            return np.random.binomial(1, 0.7, len(X))
        
        # Make multiple predictions
        results = []
        for _ in range(20):
            X_test = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
            try:
                result = predict_with_failures(X_test)
                results.append(result)
            except Exception as e:
                # Should not happen with resilient wrapper
                pytest.fail(f"Unexpected exception: {e}")
        
        # Should have results for all attempts
        assert len(results) == 20
        
        # Check some errors were handled
        stats = manager.get_error_statistics(hours=1)
        # May have errors depending on random failures
        
        # Generate resilience report
        report = manager.generate_resilience_report()
        assert "ml_prediction_service" in report
    
    def test_database_connection_resilience(self):
        """Test resilience with database connection scenario."""
        manager = ErrorRecoveryManager()
        
        # Register circuit breaker for database
        db_circuit_breaker = manager.register_circuit_breaker(
            "database", failure_threshold=3, recovery_timeout=0.1
        )
        
        connection_attempts = 0
        
        @db_circuit_breaker
        @manager.with_retry(RetryConfig(max_attempts=3, base_delay=0.01))
        def connect_to_database():
            nonlocal connection_attempts
            connection_attempts += 1
            
            # Simulate database being down for first few attempts
            if connection_attempts <= 3:
                raise ConnectionError("Database unreachable")
            
            return "Connected successfully"
        
        # Should fail initially and open circuit breaker
        with pytest.raises(ConnectionError):
            connect_to_database()
        
        # Circuit should be open now
        assert db_circuit_breaker.state.state == "open"
        
        # Wait for recovery timeout
        time.sleep(0.15)
        
        # Should succeed after recovery
        result = connect_to_database()
        assert result == "Connected successfully"
        assert db_circuit_breaker.state.state == "closed"
    
    def test_api_rate_limiting_resilience(self):
        """Test resilience with API rate limiting scenario."""
        manager = ErrorRecoveryManager()
        
        api_calls = 0
        
        @manager.with_retry(
            RetryConfig(max_attempts=5, base_delay=0.01, exponential_base=1.5),
            retry_on=(ConnectionError, TimeoutError)
        )
        def call_rate_limited_api():
            nonlocal api_calls
            api_calls += 1
            
            # Simulate rate limiting for first few calls
            if api_calls <= 3:
                raise ConnectionError("Rate limit exceeded")
            
            return {"status": "success", "data": "api_response"}
        
        result = call_rate_limited_api()
        assert result["status"] == "success"
        assert api_calls == 4  # 3 failures + 1 success
        
        # Check retry attempts were logged
        stats = manager.get_error_statistics(hours=1)
        assert stats['total_errors'] >= 3


if __name__ == "__main__":
    pytest.main([__file__])