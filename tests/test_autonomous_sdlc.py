"""
Test suite for autonomous SDLC implementation.
Tests all generations and quality gates.
"""

import unittest
import asyncio
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autonomous_sdlc_executor import (
    AutonomousSDLCExecutor,
    SDLCConfiguration,
    ProjectType,
    GenerationPhase,
    QualityGate
)
from usage_metrics_tracker import (
    UsageMetricsTracker,
    MetricType,
    ExportFormat,
    MetricEntry
)
from self_improving_system import (
    SelfImprovingSystem,
    AdaptiveCache,
    CircuitBreaker,
    AutoScaler
)
from robust_error_handling import (
    RobustErrorHandler,
    ErrorSeverity,
    ErrorCategory,
    RobustValidator
)
from comprehensive_logging import (
    ComprehensiveLogger,
    LogLevel,
    LogCategory
)
from scalable_performance_engine import (
    ScalablePerformanceEngine,
    IntelligentCache,
    ResourcePool,
    ResourceType
)

class TestAutonomousSDLC(unittest.TestCase):
    """Test autonomous SDLC execution."""
    
    def setUp(self):
        self.config = SDLCConfiguration(
            project_type=ProjectType.ML_RESEARCH,
            research_mode=True
        )
        self.executor = AutonomousSDLCExecutor(self.config)
    
    def test_sdlc_configuration(self):
        """Test SDLC configuration initialization."""
        self.assertEqual(self.config.project_type, ProjectType.ML_RESEARCH)
        self.assertTrue(self.config.research_mode)
        self.assertEqual(len(self.config.generations), 3)
        self.assertGreater(len(self.config.quality_gates), 0)
    
    def test_generation_phases(self):
        """Test generation phase enumeration."""
        phases = list(GenerationPhase)
        self.assertEqual(len(phases), 3)
        self.assertIn(GenerationPhase.MAKE_IT_WORK, phases)
        self.assertIn(GenerationPhase.MAKE_IT_ROBUST, phases)
        self.assertIn(GenerationPhase.MAKE_IT_SCALE, phases)
    
    @patch('subprocess.run')
    def test_quality_gates(self, mock_run):
        """Test quality gate execution."""
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        
        gate = QualityGate("test_gate", "echo 'test'")
        result = self.executor._run_single_quality_gate(gate)
        
        self.assertTrue(result["passed"])
        self.assertEqual(result["returncode"], 0)

class TestUsageMetricsTracker(unittest.TestCase):
    """Test usage metrics tracking system."""
    
    def setUp(self):
        self.tracker = UsageMetricsTracker(storage_path=":memory:")
    
    def test_metric_tracking(self):
        """Test basic metric tracking."""
        metric_id = self.tracker.track_metric(
            "test_metric",
            100.0,
            MetricType.PERFORMANCE
        )
        
        self.assertIsInstance(metric_id, str)
        self.assertEqual(self.tracker.performance_stats["total_metrics"], 1)
    
    def test_prediction_tracking(self):
        """Test prediction tracking with fairness monitoring."""
        metric_id = self.tracker.track_prediction(
            model_name="test_model",
            prediction=0.75,
            protected_attributes={"group": "A"},
            user_id="user123"
        )
        
        self.assertIsInstance(metric_id, str)
    
    def test_fairness_metric_tracking(self):
        """Test fairness metric tracking with bias detection."""
        # Test normal case
        self.tracker.track_fairness_metric(
            "demographic_parity",
            0.05,
            "group_a",
            threshold=0.1
        )
        
        # Test bias alert case
        self.tracker.track_fairness_metric(
            "demographic_parity",
            0.15,
            "group_b", 
            threshold=0.1
        )
        
        alerts = self.tracker.get_bias_alerts()
        self.assertGreater(len(alerts), 0)
    
    def test_export_functionality(self):
        """Test metrics export in different formats."""
        # Add some test data
        for i in range(10):
            self.tracker.track_metric(f"metric_{i}", i * 10.0)
        
        # Test JSON export
        output_path = self.tracker.export_metrics(
            ExportFormat.JSON,
            "/tmp/test_metrics.json"
        )
        
        self.assertTrue(output_path.exists())
        
        # Clean up
        output_path.unlink()
    
    def test_aggregated_metrics(self):
        """Test aggregated metrics calculation."""
        # Add test data
        for i in range(5):
            self.tracker.track_prediction("model1", 0.8)
            self.tracker.track_fairness_metric("fairness", 0.1, "group1")
        
        aggregated = self.tracker.get_aggregated_metrics()
        
        self.assertEqual(aggregated.total_predictions, 5)
        self.assertGreater(aggregated.fairness_score, 0)

class TestSelfImprovingSystem(unittest.TestCase):
    """Test self-improving system patterns."""
    
    def setUp(self):
        self.system = SelfImprovingSystem()
    
    def test_adaptive_cache(self):
        """Test adaptive cache functionality."""
        cache = self.system.get_cache()
        
        # Test cache operations
        cache.put("key1", "value1")
        result = cache.get("key1")
        self.assertEqual(result, "value1")
        
        # Test cache miss
        result = cache.get("nonexistent_key")
        self.assertIsNone(result)
        
        # Test cache statistics
        stats = cache.get_stats()
        self.assertIn("hit_rate", stats)
        self.assertIn("current_size", stats)
    
    def test_circuit_breaker(self):
        """Test circuit breaker functionality."""
        def failing_function():
            raise Exception("Test failure")
        
        def success_function():
            return "success"
        
        # Test successful execution
        result = self.system.execute_with_circuit_breaker(success_function)
        self.assertEqual(result, "success")
        
        # Test failure handling
        with self.assertRaises(Exception):
            self.system.execute_with_circuit_breaker(failing_function)
        
        # Get circuit breaker state
        state = self.system.circuit_breaker.get_state()
        self.assertIn("state", state)
        self.assertIn("failure_count", state)
    
    def test_system_statistics(self):
        """Test system statistics collection."""
        stats = self.system.get_system_stats()
        
        self.assertIn("cache_stats", stats)
        self.assertIn("circuit_breaker", stats) 
        self.assertIn("auto_scaler", stats)
        self.assertIn("monitoring", stats)

class TestRobustErrorHandling(unittest.TestCase):
    """Test robust error handling system."""
    
    def setUp(self):
        self.handler = RobustErrorHandler()
    
    def test_error_classification(self):
        """Test error severity and category classification."""
        # Test ValueError classification
        severity = self.handler._classify_severity(ValueError("test"))
        category = self.handler._classify_category(ValueError("test"))
        
        self.assertEqual(severity, ErrorSeverity.MEDIUM)
        self.assertEqual(category, ErrorCategory.VALIDATION)
        
        # Test PermissionError classification
        severity = self.handler._classify_severity(PermissionError("test"))
        category = self.handler._classify_category(PermissionError("test"))
        
        self.assertEqual(severity, ErrorSeverity.HIGH)
        self.assertEqual(category, ErrorCategory.AUTHORIZATION)
    
    def test_error_handling_with_recovery(self):
        """Test error handling with recovery attempts."""
        exception = ValueError("Test validation error")
        
        recovery_success, recovery_result = self.handler.handle_error(
            exception,
            attempt_recovery=True
        )
        
        # For validation errors, should fail fast (no recovery)
        self.assertFalse(recovery_success)
    
    def test_error_statistics(self):
        """Test error statistics collection."""
        # Generate some errors
        for i in range(5):
            self.handler.handle_error(ValueError(f"Error {i}"))
        
        stats = self.handler.get_error_statistics()
        
        self.assertGreater(stats["total_errors"], 0)
        self.assertIn("severity_distribution", stats)
        self.assertIn("category_distribution", stats)
    
    def test_robust_validator(self):
        """Test input validation functions."""
        # Test successful validation
        RobustValidator.validate_required("test_value", "test_field")
        RobustValidator.validate_type(123, int, "test_field")
        RobustValidator.validate_range(50, 0, 100, "test_field")
        
        # Test validation failures
        with self.assertRaises(ValueError):
            RobustValidator.validate_required(None, "test_field")
        
        with self.assertRaises(TypeError):
            RobustValidator.validate_type("string", int, "test_field")
        
        with self.assertRaises(ValueError):
            RobustValidator.validate_range(150, 0, 100, "test_field")
        
        # Test credit score validation
        RobustValidator.validate_credit_score(750.0)
        
        with self.assertRaises(ValueError):
            RobustValidator.validate_credit_score(200.0)  # Too low
        
        with self.assertRaises(ValueError):
            RobustValidator.validate_credit_score(900.0)  # Too high

class TestComprehensiveLogging(unittest.TestCase):
    """Test comprehensive logging system."""
    
    def setUp(self):
        # Use temporary directory for test logs
        self.logger = ComprehensiveLogger(
            app_name="test_app",
            log_dir="/tmp/test_logs"
        )
    
    def test_logging_levels(self):
        """Test different logging levels."""
        self.logger.debug("Debug message")
        self.logger.info("Info message")
        self.logger.warning("Warning message")
        self.logger.error("Error message")
        self.logger.critical("Critical message")
        
        # Should not raise exceptions
        self.assertTrue(True)
    
    def test_structured_logging(self):
        """Test structured logging with metadata."""
        self.logger.info(
            "Test structured log",
            category=LogCategory.PERFORMANCE,
            metadata={"key": "value", "number": 42}
        )
        
        # Test performance logging
        self.logger.performance("test_operation", 0.5)
        
        # Test audit logging
        self.logger.audit("user_login", metadata={"user_id": "123"})
    
    def test_logging_context(self):
        """Test logging context management."""
        self.logger.set_context(user_id="test_user", session_id="test_session")
        
        # Use context manager
        with self.logger.LoggingContext(self.logger, operation="test_op"):
            self.logger.info("Message with context")
        
        self.assertTrue(True)
    
    def test_logging_statistics(self):
        """Test logging statistics."""
        # Generate some logs
        for i in range(10):
            self.logger.info(f"Test message {i}")
        
        stats = self.logger.get_statistics()
        
        self.assertGreater(stats["total_logs"], 0)
        self.assertIn("uptime_seconds", stats)

class TestScalablePerformanceEngine(unittest.TestCase):
    """Test scalable performance engine."""
    
    def setUp(self):
        self.engine = ScalablePerformanceEngine()
    
    def test_intelligent_cache(self):
        """Test intelligent cache functionality."""
        cache = self.engine.get_cache()
        
        # Test cache operations
        cache.put("test_key", "test_value")
        result = cache.get("test_key")
        self.assertEqual(result, "test_value")
        
        # Test cache with compute function
        def compute_func():
            return "computed_value"
        
        result = cache.get("new_key", compute_func)
        self.assertEqual(result, "computed_value")
        
        # Test cache statistics
        stats = cache.get_statistics()
        self.assertIn("hit_rate", stats)
        self.assertIn("l1_items", stats)
    
    def test_resource_pools(self):
        """Test resource pool management."""
        # Test CPU pool
        cpu_stats = self.engine.cpu_pool.get_statistics()
        self.assertIn("current_workers", cpu_stats)
        self.assertIn("resource_type", cpu_stats)
        
        # Test I/O pool
        io_stats = self.engine.io_pool.get_statistics()
        self.assertIn("current_workers", io_stats)
        self.assertIn("resource_type", io_stats)
    
    def test_performance_optimization(self):
        """Test performance optimization."""
        metrics_data = {
            "cpu_usage": 75.0,
            "memory_usage": 60.0,
            "response_time": 200.0
        }
        
        optimizations = self.engine.optimize_performance(metrics_data)
        self.assertIsInstance(optimizations, dict)
    
    def test_comprehensive_statistics(self):
        """Test comprehensive statistics."""
        stats = self.engine.get_comprehensive_statistics()
        
        self.assertIn("cache", stats)
        self.assertIn("cpu_pool", stats)
        self.assertIn("io_pool", stats)
        self.assertIn("configuration", stats)

class TestIntegration(unittest.TestCase):
    """Integration tests for all components working together."""
    
    def test_complete_system_integration(self):
        """Test complete system integration."""
        # Initialize all components
        metrics_tracker = UsageMetricsTracker(storage_path=":memory:")
        self_improving = SelfImprovingSystem()
        error_handler = RobustErrorHandler()
        logger = ComprehensiveLogger(app_name="integration_test", log_dir="/tmp/integration_logs")
        performance_engine = ScalablePerformanceEngine()
        
        # Test component interactions
        # 1. Track a metric
        metric_id = metrics_tracker.track_prediction(
            "test_model",
            0.85,
            protected_attributes={"group": "test"}
        )
        
        # 2. Use adaptive cache
        cache = self_improving.get_cache()
        cache.put("model_result", {"prediction": 0.85})
        
        # 3. Handle an error
        try:
            raise ValueError("Integration test error")
        except Exception as e:
            error_handler.handle_error(e)
        
        # 4. Log the integration
        logger.info(
            "Integration test completed",
            category=LogCategory.SYSTEM,
            metadata={"metric_id": metric_id}
        )
        
        # 5. Get performance statistics
        perf_stats = performance_engine.get_comprehensive_statistics()
        
        # Verify all components are working
        self.assertIsNotNone(metric_id)
        self.assertIsNotNone(cache.get("model_result"))
        self.assertGreater(error_handler.get_error_statistics()["total_errors"], 0)
        self.assertGreater(logger.get_statistics()["total_logs"], 0)
        self.assertIn("cache", perf_stats)

if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)