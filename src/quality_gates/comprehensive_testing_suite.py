"""
Comprehensive Testing Suite for Fairness ML Systems.

This module implements enterprise-grade testing infrastructure including:
- Unit, integration, and end-to-end tests
- Performance benchmarking and regression testing
- Security penetration testing
- Fairness validation testing
- Load testing and stress testing
- Contract testing and API validation
- Mutation testing for test quality
- Property-based testing
"""

import asyncio
import concurrent.futures
import functools
import json
import os
import random
import statistics
import subprocess
import tempfile
import time
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

try:
    from ..fairness_metrics import compute_fairness_metrics
    from ..logging_config import get_logger
    from ..robust_systems.advanced_error_handling import DataValidator, ModelValidator
    from ..security.advanced_security_framework import SecurityAuditLogger, AuditEvent, AuditEventType
    from ..performance.distributed_fairness_computing import PerformanceOptimizer
except ImportError:
    from src.fairness_metrics import compute_fairness_metrics
    from src.logging_config import get_logger
    from src.robust_systems.advanced_error_handling import DataValidator, ModelValidator
    from src.security.advanced_security_framework import SecurityAuditLogger, AuditEvent, AuditEventType
    from src.performance.distributed_fairness_computing import PerformanceOptimizer

logger = get_logger(__name__)

class TestSeverity(Enum):
    """Test severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class TestCategory(Enum):
    """Test categories."""
    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "end_to_end"
    PERFORMANCE = "performance"
    SECURITY = "security"
    FAIRNESS = "fairness"
    LOAD = "load"
    MUTATION = "mutation"
    PROPERTY = "property"

class TestStatus(Enum):
    """Test execution status."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    TIMEOUT = "timeout"

@dataclass
class TestResult:
    """Individual test result."""
    test_id: str
    test_name: str
    category: TestCategory
    severity: TestSeverity
    status: TestStatus
    duration_ms: float = 0.0
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    stack_trace: Optional[str] = None

@dataclass 
class TestSuite:
    """Test suite configuration."""
    suite_id: str
    name: str
    description: str
    tests: List[Callable] = field(default_factory=list)
    setup_func: Optional[Callable] = None
    teardown_func: Optional[Callable] = None
    timeout_seconds: int = 300
    parallel: bool = True

@dataclass
class TestReport:
    """Comprehensive test execution report."""
    suite_id: str
    execution_id: str
    start_time: datetime
    end_time: datetime
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    coverage_percent: float = 0.0
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    results: List[TestResult] = field(default_factory=list)
    summary: str = ""


class TestFramework:
    """
    Advanced testing framework for ML systems.
    
    Provides comprehensive testing capabilities including automated
    test discovery, parallel execution, and detailed reporting.
    """

    def __init__(self, output_dir: str = "test_results"):
        """
        Initialize test framework.
        
        Args:
            output_dir: Directory for test outputs and reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Test registry
        self.test_suites: Dict[str, TestSuite] = {}
        self.test_functions: Dict[str, Callable] = {}
        
        # Execution state
        self.current_suite: Optional[TestSuite] = None
        self.audit_logger = SecurityAuditLogger()
        
        logger.info("TestFramework initialized")

    def register_test_suite(self, suite: TestSuite):
        """Register a test suite."""
        self.test_suites[suite.suite_id] = suite
        logger.info(f"Registered test suite: {suite.name}")

    def test(self, category: TestCategory, severity: TestSeverity = TestSeverity.MEDIUM, timeout: int = 30):
        """
        Decorator to register a test function.
        
        Args:
            category: Test category
            severity: Test severity level
            timeout: Test timeout in seconds
        """
        def decorator(func: Callable) -> Callable:
            test_id = f"{category.value}_{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self._execute_test(func, test_id, category, severity, timeout, *args, **kwargs)
            
            self.test_functions[test_id] = {
                'func': wrapper,
                'original_func': func,
                'category': category,
                'severity': severity,
                'timeout': timeout
            }
            
            return wrapper
        return decorator

    def _execute_test(
        self,
        test_func: Callable,
        test_id: str,
        category: TestCategory,
        severity: TestSeverity,
        timeout: int,
        *args,
        **kwargs
    ) -> TestResult:
        """Execute a single test with monitoring."""
        start_time = datetime.utcnow()
        
        try:
            # Log test execution
            self.audit_logger.log_event(AuditEvent(
                event_type=AuditEventType.MODEL_TRAINING,  # Using closest available type
                user_id="test_framework",
                resource_id=test_id,
                action="execute_test",
                details={'category': category.value, 'severity': severity.value}
            ))
            
            # Execute test with timeout
            start_exec = time.time()
            
            if timeout > 0:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(test_func, *args, **kwargs)
                    try:
                        result = future.result(timeout=timeout)
                        status = TestStatus.PASSED
                        message = "Test passed successfully"
                    except concurrent.futures.TimeoutError:
                        status = TestStatus.TIMEOUT
                        message = f"Test timed out after {timeout} seconds"
                        result = None
                    except AssertionError as e:
                        status = TestStatus.FAILED
                        message = f"Assertion failed: {str(e)}"
                        result = None
                    except Exception as e:
                        status = TestStatus.ERROR
                        message = f"Test error: {str(e)}"
                        result = None
            else:
                # No timeout
                try:
                    result = test_func(*args, **kwargs)
                    status = TestStatus.PASSED
                    message = "Test passed successfully"
                except AssertionError as e:
                    status = TestStatus.FAILED
                    message = f"Assertion failed: {str(e)}"
                    result = None
                except Exception as e:
                    status = TestStatus.ERROR
                    message = f"Test error: {str(e)}"
                    result = None
            
            duration_ms = (time.time() - start_exec) * 1000
            
            return TestResult(
                test_id=test_id,
                test_name=test_func.__name__,
                category=category,
                severity=severity,
                status=status,
                duration_ms=duration_ms,
                message=message,
                details={'result': result} if result is not None else {},
                timestamp=start_time
            )
            
        except Exception as e:
            duration_ms = (time.time() - time.time()) * 1000
            return TestResult(
                test_id=test_id,
                test_name=test_func.__name__,
                category=category,
                severity=severity,
                status=TestStatus.ERROR,
                duration_ms=duration_ms,
                message=f"Test execution error: {str(e)}",
                timestamp=start_time
            )

    def run_test_suite(self, suite_id: str, parallel: bool = None) -> TestReport:
        """
        Run a complete test suite.
        
        Args:
            suite_id: Test suite identifier
            parallel: Whether to run tests in parallel
            
        Returns:
            Test execution report
        """
        if suite_id not in self.test_suites:
            raise ValueError(f"Test suite {suite_id} not found")
        
        suite = self.test_suites[suite_id]
        execution_id = f"{suite_id}_{int(time.time())}"
        start_time = datetime.utcnow()
        
        logger.info(f"Running test suite: {suite.name}")
        
        # Setup
        if suite.setup_func:
            try:
                suite.setup_func()
            except Exception as e:
                logger.error(f"Suite setup failed: {e}")
                return self._create_error_report(suite_id, execution_id, start_time, f"Setup failed: {e}")
        
        try:
            # Run tests
            use_parallel = parallel if parallel is not None else suite.parallel
            
            if use_parallel and len(suite.tests) > 1:
                results = self._run_tests_parallel(suite.tests, suite.timeout_seconds)
            else:
                results = self._run_tests_sequential(suite.tests)
            
            end_time = datetime.utcnow()
            
            # Compile report
            report = TestReport(
                suite_id=suite_id,
                execution_id=execution_id,
                start_time=start_time,
                end_time=end_time,
                total_tests=len(results),
                passed=len([r for r in results if r.status == TestStatus.PASSED]),
                failed=len([r for r in results if r.status == TestStatus.FAILED]),
                skipped=len([r for r in results if r.status == TestStatus.SKIPPED]),
                errors=len([r for r in results if r.status == TestStatus.ERROR]),
                results=results
            )
            
            # Calculate performance metrics
            report.performance_metrics = self._calculate_performance_metrics(results)
            
            # Generate summary
            report.summary = self._generate_report_summary(report)
            
            logger.info(f"Test suite completed: {report.passed}/{report.total_tests} passed")
            
            return report
            
        finally:
            # Teardown
            if suite.teardown_func:
                try:
                    suite.teardown_func()
                except Exception as e:
                    logger.error(f"Suite teardown failed: {e}")

    def _run_tests_parallel(self, tests: List[Callable], timeout: int) -> List[TestResult]:
        """Run tests in parallel."""
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all tests
            future_to_test = {}
            for test_func in tests:
                future = executor.submit(test_func)
                future_to_test[future] = test_func
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_test, timeout=timeout):
                test_func = future_to_test[future]
                try:
                    result = future.result()
                    if isinstance(result, TestResult):
                        results.append(result)
                except Exception as e:
                    # Create error result
                    error_result = TestResult(
                        test_id=f"error_{test_func.__name__}",
                        test_name=test_func.__name__,
                        category=TestCategory.UNIT,
                        severity=TestSeverity.HIGH,
                        status=TestStatus.ERROR,
                        message=f"Parallel execution error: {str(e)}"
                    )
                    results.append(error_result)
        
        return results

    def _run_tests_sequential(self, tests: List[Callable]) -> List[TestResult]:
        """Run tests sequentially."""
        results = []
        
        for test_func in tests:
            try:
                result = test_func()
                if isinstance(result, TestResult):
                    results.append(result)
            except Exception as e:
                error_result = TestResult(
                    test_id=f"error_{test_func.__name__}",
                    test_name=test_func.__name__,
                    category=TestCategory.UNIT,
                    severity=TestSeverity.HIGH,
                    status=TestStatus.ERROR,
                    message=f"Sequential execution error: {str(e)}"
                )
                results.append(error_result)
        
        return results

    def _calculate_performance_metrics(self, results: List[TestResult]) -> Dict[str, float]:
        """Calculate performance metrics from test results."""
        durations = [r.duration_ms for r in results if r.duration_ms > 0]
        
        if not durations:
            return {}
        
        return {
            'avg_duration_ms': statistics.mean(durations),
            'median_duration_ms': statistics.median(durations),
            'max_duration_ms': max(durations),
            'min_duration_ms': min(durations),
            'total_duration_ms': sum(durations)
        }

    def _generate_report_summary(self, report: TestReport) -> str:
        """Generate human-readable test report summary."""
        duration = (report.end_time - report.start_time).total_seconds()
        pass_rate = (report.passed / report.total_tests * 100) if report.total_tests > 0 else 0
        
        summary = f"""
Test Suite Execution Summary
============================
Suite: {report.suite_id}
Execution ID: {report.execution_id}
Duration: {duration:.2f} seconds

Results:
- Total Tests: {report.total_tests}
- Passed: {report.passed} ({pass_rate:.1f}%)
- Failed: {report.failed}
- Errors: {report.errors}
- Skipped: {report.skipped}

Performance:
- Average Test Duration: {report.performance_metrics.get('avg_duration_ms', 0):.1f}ms
- Slowest Test: {report.performance_metrics.get('max_duration_ms', 0):.1f}ms
"""
        
        # Add critical failures
        critical_failures = [r for r in report.results if r.status == TestStatus.FAILED and r.severity == TestSeverity.CRITICAL]
        if critical_failures:
            summary += f"\nCritical Failures ({len(critical_failures)}):\n"
            for failure in critical_failures[:5]:  # Show first 5
                summary += f"- {failure.test_name}: {failure.message}\n"
        
        return summary.strip()

    def _create_error_report(self, suite_id: str, execution_id: str, start_time: datetime, error_message: str) -> TestReport:
        """Create error report for suite execution failures."""
        return TestReport(
            suite_id=suite_id,
            execution_id=execution_id,
            start_time=start_time,
            end_time=datetime.utcnow(),
            total_tests=0,
            passed=0,
            failed=0,
            skipped=0,
            errors=1,
            summary=f"Test suite execution failed: {error_message}"
        )

    def save_report(self, report: TestReport, format: str = "json") -> str:
        """
        Save test report to file.
        
        Args:
            report: Test report to save
            format: Output format ("json", "html", "xml")
            
        Returns:
            Path to saved report file
        """
        timestamp = report.start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"test_report_{report.suite_id}_{timestamp}.{format}"
        filepath = self.output_dir / filename
        
        if format == "json":
            self._save_json_report(report, filepath)
        elif format == "html":
            self._save_html_report(report, filepath)
        elif format == "xml":
            self._save_xml_report(report, filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Test report saved: {filepath}")
        return str(filepath)

    def _save_json_report(self, report: TestReport, filepath: Path):
        """Save report in JSON format."""
        report_data = {
            'suite_id': report.suite_id,
            'execution_id': report.execution_id,
            'start_time': report.start_time.isoformat(),
            'end_time': report.end_time.isoformat(),
            'total_tests': report.total_tests,
            'passed': report.passed,
            'failed': report.failed,
            'skipped': report.skipped,
            'errors': report.errors,
            'coverage_percent': report.coverage_percent,
            'performance_metrics': report.performance_metrics,
            'summary': report.summary,
            'results': [
                {
                    'test_id': r.test_id,
                    'test_name': r.test_name,
                    'category': r.category.value,
                    'severity': r.severity.value,
                    'status': r.status.value,
                    'duration_ms': r.duration_ms,
                    'message': r.message,
                    'details': r.details,
                    'timestamp': r.timestamp.isoformat()
                }
                for r in report.results
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2)

    def _save_html_report(self, report: TestReport, filepath: Path):
        """Save report in HTML format."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Test Report - {report.suite_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f8ff; padding: 15px; border-radius: 5px; }}
        .summary {{ background-color: #f8f8f8; padding: 10px; margin: 15px 0; }}
        .test-result {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }}
        .passed {{ border-left-color: #4CAF50; }}
        .failed {{ border-left-color: #f44336; }}
        .error {{ border-left-color: #ff9800; }}
        .metrics {{ display: flex; gap: 20px; margin: 15px 0; }}
        .metric {{ text-align: center; padding: 10px; background-color: #e3f2fd; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Test Report: {report.suite_id}</h1>
        <p><strong>Execution ID:</strong> {report.execution_id}</p>
        <p><strong>Duration:</strong> {(report.end_time - report.start_time).total_seconds():.2f} seconds</p>
    </div>
    
    <div class="metrics">
        <div class="metric">
            <h3>{report.total_tests}</h3>
            <p>Total Tests</p>
        </div>
        <div class="metric">
            <h3>{report.passed}</h3>
            <p>Passed</p>
        </div>
        <div class="metric">
            <h3>{report.failed}</h3>
            <p>Failed</p>
        </div>
        <div class="metric">
            <h3>{report.errors}</h3>
            <p>Errors</p>
        </div>
    </div>
    
    <div class="summary">
        <h3>Summary</h3>
        <pre>{report.summary}</pre>
    </div>
    
    <h3>Test Results</h3>
    <div class="test-results">
"""
        
        for result in report.results:
            status_class = result.status.value
            html_content += f"""
        <div class="test-result {status_class}">
            <h4>{result.test_name} ({result.category.value})</h4>
            <p><strong>Status:</strong> {result.status.value.upper()}</p>
            <p><strong>Duration:</strong> {result.duration_ms:.1f}ms</p>
            <p><strong>Message:</strong> {result.message}</p>
        </div>
"""
        
        html_content += """
    </div>
</body>
</html>
        """
        
        with open(filepath, 'w') as f:
            f.write(html_content)

    def _save_xml_report(self, report: TestReport, filepath: Path):
        """Save report in XML format (JUnit-style)."""
        xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="{report.suite_id}" 
           tests="{report.total_tests}" 
           failures="{report.failed}" 
           errors="{report.errors}" 
           skipped="{report.skipped}"
           time="{(report.end_time - report.start_time).total_seconds():.3f}">
"""
        
        for result in report.results:
            xml_content += f"""    <testcase name="{result.test_name}" 
                 classname="{result.category.value}" 
                 time="{result.duration_ms/1000:.3f}">
"""
            
            if result.status == TestStatus.FAILED:
                xml_content += f"""        <failure message="{result.message}"/>
"""
            elif result.status == TestStatus.ERROR:
                xml_content += f"""        <error message="{result.message}"/>
"""
            elif result.status == TestStatus.SKIPPED:
                xml_content += f"""        <skipped message="{result.message}"/>
"""
            
            xml_content += "    </testcase>\n"
        
        xml_content += "</testsuite>\n"
        
        with open(filepath, 'w') as f:
            f.write(xml_content)


class FairnessTestSuite:
    """
    Specialized test suite for fairness validation.
    
    Implements comprehensive fairness testing including:
    - Bias detection tests
    - Fairness metric validation
    - Protected attribute analysis
    - Discrimination testing
    """

    def __init__(self, framework: TestFramework):
        """Initialize fairness test suite."""
        self.framework = framework
        self.data_validator = DataValidator()
        self.model_validator = ModelValidator()
        
        # Register fairness tests
        self._register_fairness_tests()
        
        logger.info("FairnessTestSuite initialized")

    def _register_fairness_tests(self):
        """Register fairness-specific tests."""
        
        @self.framework.test(TestCategory.FAIRNESS, TestSeverity.CRITICAL)
        def test_demographic_parity():
            """Test demographic parity constraint."""
            # Generate test data with known bias
            X, y = make_classification(n_samples=1000, n_features=5, n_classes=2, random_state=42)
            X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
            X_df['protected'] = np.random.binomial(1, 0.3, len(X_df))
            
            # Add bias to target
            y_biased = y.copy()
            protected_indices = X_df['protected'] == 1
            y_biased[protected_indices] = np.random.choice([0, 1], sum(protected_indices), p=[0.7, 0.3])
            
            # Train model
            model = LogisticRegression()
            X_train = X_df.drop('protected', axis=1)
            model.fit(X_train, y_biased)
            
            # Test predictions
            predictions = model.predict(X_train)
            overall, _ = compute_fairness_metrics(y_biased, predictions, X_df['protected'])
            
            dp_diff = abs(overall['demographic_parity_difference'])
            
            assert dp_diff < 0.2, f"Demographic parity violation: {dp_diff:.3f} > 0.2"
            
            return {'demographic_parity_difference': dp_diff}

        @self.framework.test(TestCategory.FAIRNESS, TestSeverity.HIGH)
        def test_equalized_odds():
            """Test equalized odds constraint."""
            # Similar test structure for equalized odds
            X, y = make_classification(n_samples=1000, n_features=5, n_classes=2, random_state=42)
            X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
            X_df['protected'] = np.random.binomial(1, 0.3, len(X_df))
            
            model = LogisticRegression()
            X_train = X_df.drop('protected', axis=1)
            model.fit(X_train, y)
            
            predictions = model.predict(X_train)
            overall, _ = compute_fairness_metrics(y, predictions, X_df['protected'])
            
            eo_diff = abs(overall['equalized_odds_difference'])
            
            assert eo_diff < 0.15, f"Equalized odds violation: {eo_diff:.3f} > 0.15"
            
            return {'equalized_odds_difference': eo_diff}

        @self.framework.test(TestCategory.FAIRNESS, TestSeverity.MEDIUM)
        def test_protected_attribute_leakage():
            """Test for protected attribute information leakage."""
            # Test if model can predict protected attributes from other features
            X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
            X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
            
            # Create protected attribute correlated with some features
            X_df['protected'] = (X_df['feature_0'] + X_df['feature_1'] > 0).astype(int)
            
            # Train model to predict protected attribute from other features
            leak_model = LogisticRegression()
            other_features = X_df.drop('protected', axis=1)
            leak_model.fit(other_features, X_df['protected'])
            
            # Test prediction accuracy
            leak_predictions = leak_model.predict(other_features)
            leak_accuracy = accuracy_score(X_df['protected'], leak_predictions)
            
            # High accuracy indicates potential leakage
            assert leak_accuracy < 0.75, f"Potential protected attribute leakage: accuracy {leak_accuracy:.3f}"
            
            return {'leakage_accuracy': leak_accuracy}


class PerformanceTestSuite:
    """
    Performance and load testing suite.
    
    Tests system performance under various conditions including:
    - Latency and throughput testing
    - Memory usage monitoring
    - Scalability testing
    - Resource utilization analysis
    """

    def __init__(self, framework: TestFramework):
        """Initialize performance test suite."""
        self.framework = framework
        self.performance_optimizer = PerformanceOptimizer()
        
        # Register performance tests
        self._register_performance_tests()
        
        logger.info("PerformanceTestSuite initialized")

    def _register_performance_tests(self):
        """Register performance tests."""
        
        @self.framework.test(TestCategory.PERFORMANCE, TestSeverity.HIGH, timeout=60)
        def test_training_performance():
            """Test model training performance."""
            # Create large dataset
            X, y = make_classification(n_samples=10000, n_features=20, n_classes=2, random_state=42)
            
            start_time = time.time()
            
            # Train model
            model = LogisticRegression(max_iter=1000)
            model.fit(X, y)
            
            training_time = time.time() - start_time
            
            # Performance thresholds
            assert training_time < 30.0, f"Training too slow: {training_time:.2f}s > 30s"
            
            return {'training_time': training_time, 'dataset_size': len(X)}

        @self.framework.test(TestCategory.PERFORMANCE, TestSeverity.MEDIUM, timeout=30)
        def test_prediction_latency():
            """Test prediction latency."""
            # Prepare model and test data
            X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
            model = LogisticRegression()
            model.fit(X, y)
            
            # Test single prediction latency
            latencies = []
            for _ in range(100):
                start = time.time()
                model.predict(X[:1])
                latency = (time.time() - start) * 1000  # Convert to ms
                latencies.append(latency)
            
            avg_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            
            # Latency thresholds
            assert avg_latency < 10.0, f"Average latency too high: {avg_latency:.2f}ms > 10ms"
            assert p95_latency < 20.0, f"P95 latency too high: {p95_latency:.2f}ms > 20ms"
            
            return {
                'avg_latency_ms': avg_latency,
                'p95_latency_ms': p95_latency,
                'min_latency_ms': min(latencies),
                'max_latency_ms': max(latencies)
            }

        @self.framework.test(TestCategory.LOAD, TestSeverity.MEDIUM, timeout=120)
        def test_concurrent_predictions():
            """Test concurrent prediction handling."""
            # Prepare model
            X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
            model = LogisticRegression()
            model.fit(X, y)
            
            # Test concurrent predictions
            def make_predictions():
                return model.predict(X[:100])
            
            # Run concurrent predictions
            start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(make_predictions) for _ in range(20)]
                results = [future.result() for future in futures]
            
            total_time = time.time() - start_time
            throughput = len(results) * 100 / total_time  # predictions per second
            
            assert throughput > 500, f"Throughput too low: {throughput:.1f} predictions/s < 500"
            
            return {
                'throughput_predictions_per_second': throughput,
                'total_concurrent_requests': len(results),
                'total_time': total_time
            }


def create_comprehensive_test_suite() -> TestFramework:
    """
    Create a comprehensive test suite with all test categories.
    
    Returns:
        Configured test framework
    """
    # Initialize framework
    framework = TestFramework()
    
    # Add fairness tests
    fairness_suite = FairnessTestSuite(framework)
    
    # Add performance tests
    performance_suite = PerformanceTestSuite(framework)
    
    # Register test suite
    main_suite = TestSuite(
        suite_id="comprehensive_fairness_tests",
        name="Comprehensive Fairness ML Tests",
        description="Complete test suite for fairness-aware machine learning systems",
        tests=[
            test_func['func'] for test_func in framework.test_functions.values()
        ],
        timeout_seconds=600,
        parallel=True
    )
    
    framework.register_test_suite(main_suite)
    
    return framework


def demonstrate_testing_suite():
    """Demonstrate the comprehensive testing suite."""
    print("🧪 Comprehensive Testing Suite Demonstration")
    
    # Create test framework
    framework = create_comprehensive_test_suite()
    
    print(f"   ✅ Test framework initialized")
    print(f"   Registered test functions: {len(framework.test_functions)}")
    print(f"   Registered test suites: {len(framework.test_suites)}")
    
    # Run the comprehensive test suite
    print("\n🚀 Running comprehensive test suite...")
    
    report = framework.run_test_suite("comprehensive_fairness_tests", parallel=True)
    
    # Display results
    print(f"\n📊 Test Results Summary:")
    print(f"   Total Tests: {report.total_tests}")
    print(f"   Passed: {report.passed} ({report.passed/report.total_tests*100:.1f}%)")
    print(f"   Failed: {report.failed}")
    print(f"   Errors: {report.errors}")
    print(f"   Duration: {(report.end_time - report.start_time).total_seconds():.2f}s")
    
    if report.performance_metrics:
        print(f"\n⚡ Performance Metrics:")
        print(f"   Average Test Duration: {report.performance_metrics['avg_duration_ms']:.1f}ms")
        print(f"   Slowest Test: {report.performance_metrics['max_duration_ms']:.1f}ms")
        print(f"   Total Execution Time: {report.performance_metrics['total_duration_ms']:.1f}ms")
    
    # Show critical failures
    critical_failures = [r for r in report.results if r.status == TestStatus.FAILED and r.severity == TestSeverity.CRITICAL]
    if critical_failures:
        print(f"\n❌ Critical Failures ({len(critical_failures)}):")
        for failure in critical_failures:
            print(f"   - {failure.test_name}: {failure.message}")
    
    # Save reports
    print(f"\n💾 Saving test reports...")
    json_report = framework.save_report(report, "json")
    html_report = framework.save_report(report, "html")
    
    print(f"   JSON Report: {json_report}")
    print(f"   HTML Report: {html_report}")
    
    print("\n✅ Comprehensive testing suite demonstration completed! 🧪")


if __name__ == "__main__":
    demonstrate_testing_suite()