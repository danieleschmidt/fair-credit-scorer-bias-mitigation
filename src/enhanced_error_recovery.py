"""
Enhanced Error Recovery and Resilience Framework.

Provides comprehensive error handling, recovery mechanisms, and system resilience
for production fairness-aware ML systems.

Features:
- Circuit breaker pattern for external dependencies
- Exponential backoff and retry mechanisms
- Graceful degradation strategies
- Error classification and routing
- Health monitoring and auto-recovery
- Fallback model serving
"""

import asyncio
import functools
import json
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from .logging_config import get_logger

logger = get_logger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    DATA_ERROR = "data_error"
    MODEL_ERROR = "model_error"
    SYSTEM_ERROR = "system_error"
    NETWORK_ERROR = "network_error"
    VALIDATION_ERROR = "validation_error"
    FAIRNESS_ERROR = "fairness_error"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Recovery strategy types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    FAIL_FAST = "fail_fast"


@dataclass
class ErrorEvent:
    """Error event record."""
    timestamp: datetime
    error_type: str
    error_message: str
    severity: ErrorSeverity
    category: ErrorCategory
    context: Dict[str, Any]
    stack_trace: Optional[str] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False
    resolution_time: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'error_type': self.error_type,
            'error_message': self.error_message,
            'severity': self.severity.value,
            'category': self.category.value,
            'context': self.context,
            'stack_trace': self.stack_trace,
            'recovery_attempted': self.recovery_attempted,
            'recovery_successful': self.recovery_successful,
            'resolution_time': self.resolution_time.isoformat() if self.resolution_time else None
        }


@dataclass
class CircuitBreakerState:
    """Circuit breaker state tracking."""
    state: str = "closed"  # closed, open, half_open
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    next_attempt_time: Optional[datetime] = None


class ErrorClassifier:
    """Classify errors by type and severity."""

    def __init__(self):
        self.classification_rules = self._build_classification_rules()

    def _build_classification_rules(self) -> Dict[str, Dict[str, Any]]:
        """Build error classification rules."""
        return {
            # Data errors
            'ValueError': {
                'category': ErrorCategory.DATA_ERROR,
                'severity': ErrorSeverity.MEDIUM,
                'keywords': ['data', 'value', 'shape', 'dtype']
            },
            'KeyError': {
                'category': ErrorCategory.DATA_ERROR,
                'severity': ErrorSeverity.HIGH,
                'keywords': ['key', 'column', 'feature']
            },
            'FileNotFoundError': {
                'category': ErrorCategory.SYSTEM_ERROR,
                'severity': ErrorSeverity.HIGH,
                'keywords': ['file', 'path', 'directory']
            },
            
            # Model errors
            'NotFittedError': {
                'category': ErrorCategory.MODEL_ERROR,
                'severity': ErrorSeverity.HIGH,
                'keywords': ['fitted', 'trained', 'model']
            },
            'ConvergenceWarning': {
                'category': ErrorCategory.MODEL_ERROR,
                'severity': ErrorSeverity.MEDIUM,
                'keywords': ['convergence', 'iteration', 'optimization']
            },
            
            # Network errors
            'ConnectionError': {
                'category': ErrorCategory.NETWORK_ERROR,
                'severity': ErrorSeverity.HIGH,
                'keywords': ['connection', 'network', 'timeout']
            },
            'TimeoutError': {
                'category': ErrorCategory.NETWORK_ERROR,
                'severity': ErrorSeverity.MEDIUM,
                'keywords': ['timeout', 'slow', 'response']
            },
            
            # System errors
            'MemoryError': {
                'category': ErrorCategory.SYSTEM_ERROR,
                'severity': ErrorSeverity.CRITICAL,
                'keywords': ['memory', 'allocation', 'ram']
            },
            'SystemError': {
                'category': ErrorCategory.SYSTEM_ERROR,
                'severity': ErrorSeverity.CRITICAL,
                'keywords': ['system', 'internal', 'kernel']
            }
        }

    def classify_error(self, error: Exception, context: Dict[str, Any] = None) -> Tuple[ErrorCategory, ErrorSeverity]:
        """
        Classify error by category and severity.
        
        Args:
            error: Exception to classify
            context: Additional context for classification
            
        Returns:
            Tuple of (category, severity)
        """
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Check direct mapping
        if error_type in self.classification_rules:
            rule = self.classification_rules[error_type]
            return rule['category'], rule['severity']
        
        # Check message content
        for rule_name, rule in self.classification_rules.items():
            if any(keyword in error_message for keyword in rule['keywords']):
                return rule['category'], rule['severity']
        
        # Check context for additional hints
        if context:
            if 'fairness' in str(context).lower():
                return ErrorCategory.FAIRNESS_ERROR, ErrorSeverity.HIGH
            if 'validation' in str(context).lower():
                return ErrorCategory.VALIDATION_ERROR, ErrorSeverity.MEDIUM
        
        # Default classification
        return ErrorCategory.UNKNOWN, ErrorSeverity.MEDIUM


class RetryConfig:
    """Configuration for retry mechanism."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add random jitter (±25%)
            import random
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)


class CircuitBreaker:
    """Circuit breaker implementation for external dependencies."""
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.state = CircuitBreakerState()
        
        logger.info(f"CircuitBreaker '{name}' initialized")

    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap function with circuit breaker."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self._call_with_circuit_breaker(func, *args, **kwargs)
        return wrapper

    def _call_with_circuit_breaker(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state.state == "open":
            if self._should_attempt_reset():
                self.state.state = "half_open"
                logger.info(f"CircuitBreaker '{self.name}' transitioning to half-open")
            else:
                raise Exception(f"CircuitBreaker '{self.name}' is open")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.state.next_attempt_time is None:
            return True
        return datetime.now() >= self.state.next_attempt_time

    def _on_success(self):
        """Handle successful function call."""
        if self.state.state == "half_open":
            self.state.state = "closed"
            logger.info(f"CircuitBreaker '{self.name}' closed after successful recovery")
        
        self.state.failure_count = 0
        self.state.success_count += 1

    def _on_failure(self):
        """Handle failed function call."""
        self.state.failure_count += 1
        self.state.last_failure_time = datetime.now()
        
        if self.state.failure_count >= self.failure_threshold:
            self.state.state = "open"
            self.state.next_attempt_time = datetime.now() + timedelta(seconds=self.recovery_timeout)
            logger.warning(f"CircuitBreaker '{self.name}' opened due to {self.state.failure_count} failures")


class FallbackModel:
    """Simple fallback model for graceful degradation."""
    
    def __init__(self, name: str = "fallback"):
        self.name = name
        self.fitted = False
        self.fallback_prediction = 0.5  # Default neutral prediction
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit simple fallback model."""
        # Use simple majority class or mean prediction
        if y.dtype == 'object' or len(y.unique()) <= 10:
            # Classification: use majority class
            self.fallback_prediction = y.mode().iloc[0] if len(y.mode()) > 0 else 0
        else:
            # Regression: use mean
            self.fallback_prediction = y.mean()
        
        self.fitted = True
        logger.info(f"FallbackModel '{self.name}' fitted with prediction: {self.fallback_prediction}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make fallback predictions."""
        if not self.fitted:
            logger.warning(f"FallbackModel '{self.name}' not fitted, using default prediction")
        
        return np.full(len(X), self.fallback_prediction)


class HealthMonitor:
    """Monitor system health and performance."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = {
            'error_count': 0,
            'success_count': 0,
            'response_times': [],
            'memory_usage': [],
            'last_health_check': None
        }
        
    def record_success(self, response_time: float = None):
        """Record successful operation."""
        self.metrics['success_count'] += 1
        if response_time is not None:
            self.metrics['response_times'].append(response_time)
            self._trim_metrics('response_times')

    def record_error(self):
        """Record error occurrence."""
        self.metrics['error_count'] += 1

    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        total_ops = self.metrics['success_count'] + self.metrics['error_count']
        error_rate = self.metrics['error_count'] / max(1, total_ops)
        
        avg_response_time = (
            np.mean(self.metrics['response_times']) 
            if self.metrics['response_times'] else 0
        )
        
        health_score = max(0, 1 - error_rate)
        
        status = {
            'health_score': health_score,
            'error_rate': error_rate,
            'total_operations': total_ops,
            'average_response_time': avg_response_time,
            'status': 'healthy' if health_score > 0.8 else 'degraded' if health_score > 0.5 else 'unhealthy'
        }
        
        self.metrics['last_health_check'] = datetime.now()
        return status

    def _trim_metrics(self, metric_name: str):
        """Trim metrics to window size."""
        if len(self.metrics[metric_name]) > self.window_size:
            self.metrics[metric_name] = self.metrics[metric_name][-self.window_size:]


class ErrorRecoveryManager:
    """
    Comprehensive error recovery and resilience manager.
    
    Orchestrates various recovery strategies and maintains system resilience.
    """
    
    def __init__(
        self,
        enable_circuit_breaker: bool = True,
        enable_retry: bool = True,
        enable_fallback: bool = True,
        error_log_path: str = "error_log.json"
    ):
        self.enable_circuit_breaker = enable_circuit_breaker
        self.enable_retry = enable_retry
        self.enable_fallback = enable_fallback
        self.error_log_path = Path(error_log_path)
        
        # Components
        self.error_classifier = ErrorClassifier()
        self.health_monitor = HealthMonitor()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.fallback_models: Dict[str, FallbackModel] = {}
        self.error_events: List[ErrorEvent] = []
        
        # Load existing error log
        self._load_error_log()
        
        logger.info("ErrorRecoveryManager initialized")

    def _load_error_log(self):
        """Load existing error log from disk."""
        if self.error_log_path.exists():
            try:
                with open(self.error_log_path, 'r') as f:
                    error_data = json.load(f)
                
                for event_data in error_data:
                    event = ErrorEvent(
                        timestamp=datetime.fromisoformat(event_data['timestamp']),
                        error_type=event_data['error_type'],
                        error_message=event_data['error_message'],
                        severity=ErrorSeverity(event_data['severity']),
                        category=ErrorCategory(event_data['category']),
                        context=event_data['context'],
                        stack_trace=event_data.get('stack_trace'),
                        recovery_attempted=event_data.get('recovery_attempted', False),
                        recovery_successful=event_data.get('recovery_successful', False),
                        resolution_time=datetime.fromisoformat(event_data['resolution_time']) 
                                      if event_data.get('resolution_time') else None
                    )
                    self.error_events.append(event)
                
                logger.info(f"Loaded {len(self.error_events)} error events from log")
                
            except Exception as e:
                logger.error(f"Failed to load error log: {e}")

    def _save_error_log(self):
        """Save error log to disk."""
        try:
            error_data = [event.to_dict() for event in self.error_events]
            with open(self.error_log_path, 'w') as f:
                json.dump(error_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save error log: {e}")

    def register_circuit_breaker(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0
    ) -> CircuitBreaker:
        """Register a new circuit breaker."""
        circuit_breaker = CircuitBreaker(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout
        )
        self.circuit_breakers[name] = circuit_breaker
        return circuit_breaker

    def register_fallback_model(self, name: str, model: FallbackModel):
        """Register a fallback model."""
        self.fallback_models[name] = model
        logger.info(f"Registered fallback model '{name}'")

    def with_retry(
        self,
        retry_config: RetryConfig = None,
        retry_on: Tuple[Type[Exception], ...] = (Exception,)
    ):
        """Decorator for retry functionality."""
        if retry_config is None:
            retry_config = RetryConfig()
            
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self._execute_with_retry(func, retry_config, retry_on, *args, **kwargs)
            return wrapper
        return decorator

    def _execute_with_retry(
        self,
        func: Callable,
        retry_config: RetryConfig,
        retry_on: Tuple[Type[Exception], ...],
        *args,
        **kwargs
    ):
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(retry_config.max_attempts):
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                
                # Record success
                response_time = time.time() - start_time
                self.health_monitor.record_success(response_time)
                
                if attempt > 0:
                    logger.info(f"Function succeeded on attempt {attempt + 1}")
                
                return result
                
            except retry_on as e:
                last_exception = e
                self.health_monitor.record_error()
                
                # Classify and log error
                category, severity = self.error_classifier.classify_error(e)
                self._log_error_event(e, category, severity, {'attempt': attempt + 1})
                
                if attempt < retry_config.max_attempts - 1:
                    delay = retry_config.get_delay(attempt)
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s: {e}")
                    time.sleep(delay)
                else:
                    logger.error(f"All {retry_config.max_attempts} attempts failed: {e}")
            
            except Exception as e:
                # Non-retryable exception
                self.health_monitor.record_error()
                category, severity = self.error_classifier.classify_error(e)
                self._log_error_event(e, category, severity, {'attempt': attempt + 1, 'non_retryable': True})
                raise e
        
        # All retries exhausted
        raise last_exception

    def with_fallback(self, fallback_model_name: str = None, fallback_value: Any = None):
        """Decorator for fallback functionality."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Function failed, using fallback: {e}")
                    
                    # Log error
                    category, severity = self.error_classifier.classify_error(e)
                    self._log_error_event(e, category, severity, {'fallback_used': True})
                    
                    # Use fallback
                    if fallback_model_name and fallback_model_name in self.fallback_models:
                        # Try to extract input data for model prediction
                        if args and hasattr(args[0], '__len__'):
                            try:
                                X = args[0] if isinstance(args[0], pd.DataFrame) else pd.DataFrame(args[0])
                                return self.fallback_models[fallback_model_name].predict(X)
                            except:
                                pass
                    
                    # Use fallback value
                    if fallback_value is not None:
                        return fallback_value
                    
                    # If no fallback available, re-raise
                    raise e
                    
            return wrapper
        return decorator

    def _log_error_event(
        self,
        error: Exception,
        category: ErrorCategory,
        severity: ErrorSeverity,
        context: Dict[str, Any] = None
    ):
        """Log error event."""
        event = ErrorEvent(
            timestamp=datetime.now(),
            error_type=type(error).__name__,
            error_message=str(error),
            severity=severity,
            category=category,
            context=context or {},
            stack_trace=traceback.format_exc()
        )
        
        self.error_events.append(event)
        
        # Trim old events (keep last 1000)
        if len(self.error_events) > 1000:
            self.error_events = self.error_events[-1000:]
        
        # Save to disk periodically
        if len(self.error_events) % 10 == 0:
            self._save_error_log()

    def get_error_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get error statistics for the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_errors = [e for e in self.error_events if e.timestamp >= cutoff_time]
        
        if not recent_errors:
            return {
                'total_errors': 0,
                'error_rate': 0.0,
                'categories': {},
                'severities': {},
                'top_errors': []
            }
        
        # Count by category
        category_counts = {}
        for error in recent_errors:
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
        
        # Count by severity
        severity_counts = {}
        for error in recent_errors:
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
        
        # Top error types
        error_type_counts = {}
        for error in recent_errors:
            error_type_counts[error.error_type] = error_type_counts.get(error.error_type, 0) + 1
        
        top_errors = sorted(error_type_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_errors': len(recent_errors),
            'error_rate': len(recent_errors) / max(1, hours),
            'categories': category_counts,
            'severities': severity_counts,
            'top_errors': top_errors,
            'health_status': self.health_monitor.get_health_status()
        }

    def create_resilient_wrapper(
        self,
        name: str,
        retry_config: RetryConfig = None,
        enable_circuit_breaker: bool = None,
        enable_fallback: bool = None,
        fallback_model_name: str = None,
        fallback_value: Any = None
    ):
        """Create comprehensive resilient wrapper for functions."""
        if enable_circuit_breaker is None:
            enable_circuit_breaker = self.enable_circuit_breaker
        if enable_fallback is None:
            enable_fallback = self.enable_fallback
        if retry_config is None:
            retry_config = RetryConfig()
        
        def decorator(func: Callable) -> Callable:
            wrapped_func = func
            
            # Apply retry
            if self.enable_retry:
                wrapped_func = self.with_retry(retry_config)(wrapped_func)
            
            # Apply circuit breaker
            if enable_circuit_breaker:
                if name not in self.circuit_breakers:
                    self.register_circuit_breaker(name)
                wrapped_func = self.circuit_breakers[name](wrapped_func)
            
            # Apply fallback
            if enable_fallback:
                wrapped_func = self.with_fallback(fallback_model_name, fallback_value)(wrapped_func)
            
            return wrapped_func
        
        return decorator

    def generate_resilience_report(self, output_path: str = None) -> str:
        """Generate comprehensive resilience report."""
        stats = self.get_error_statistics()
        health = self.health_monitor.get_health_status()
        
        report = f"""
# System Resilience Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Health Status
- Overall Health Score: {health['health_score']:.2f}
- Status: {health['status'].upper()}
- Error Rate: {health['error_rate']:.2%}
- Total Operations: {health['total_operations']}
- Average Response Time: {health['average_response_time']:.3f}s

## Error Statistics (Last 24 Hours)
- Total Errors: {stats['total_errors']}
- Error Rate: {stats['error_rate']:.1f} errors/hour

### By Category
"""
        
        for category, count in stats['categories'].items():
            report += f"- {category}: {count}\n"
        
        report += "\n### By Severity\n"
        for severity, count in stats['severities'].items():
            report += f"- {severity}: {count}\n"
        
        report += "\n### Top Error Types\n"
        for error_type, count in stats['top_errors']:
            report += f"- {error_type}: {count}\n"
        
        report += f"\n## Circuit Breakers\n"
        for name, cb in self.circuit_breakers.items():
            report += f"- {name}: {cb.state.state} (failures: {cb.state.failure_count})\n"
        
        report += f"\n## Fallback Models\n"
        for name, model in self.fallback_models.items():
            report += f"- {name}: {'fitted' if model.fitted else 'not fitted'}\n"
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Resilience report saved to {output_path}")
        
        return report


# Example usage and demonstration
def main():
    """Demonstration of error recovery framework."""
    import argparse
    import random
    
    parser = argparse.ArgumentParser(description="Error Recovery Framework Demo")
    parser.add_argument("--demo", action="store_true", help="Run demonstration")
    parser.add_argument("--report", action="store_true", help="Generate resilience report")
    
    args = parser.parse_args()
    
    # Initialize recovery manager
    recovery_manager = ErrorRecoveryManager()
    
    if args.demo:
        print("Running Error Recovery Framework Demo...")
        
        # Create a fallback model
        fallback_model = FallbackModel("demo_fallback")
        
        # Simulate training data
        X_train = pd.DataFrame({'feature1': range(100), 'feature2': range(100, 200)})
        y_train = pd.Series([random.choice([0, 1]) for _ in range(100)])
        fallback_model.fit(X_train, y_train)
        
        recovery_manager.register_fallback_model("demo_fallback", fallback_model)
        
        # Demo function that fails sometimes
        @recovery_manager.create_resilient_wrapper(
            name="demo_function",
            retry_config=RetryConfig(max_attempts=3, base_delay=0.1),
            enable_circuit_breaker=True,
            enable_fallback=True,
            fallback_model_name="demo_fallback"
        )
        def unreliable_function(X):
            """Function that fails 30% of the time."""
            if random.random() < 0.3:
                raise ValueError("Random failure for demo")
            return np.random.random(len(X))
        
        # Test the resilient function
        test_data = pd.DataFrame({'feature1': range(10), 'feature2': range(10, 20)})
        
        print("Testing resilient function...")
        for i in range(10):
            try:
                result = unreliable_function(test_data)
                print(f"Call {i+1}: Success (result shape: {result.shape})")
            except Exception as e:
                print(f"Call {i+1}: Failed - {e}")
        
        # Show statistics
        stats = recovery_manager.get_error_statistics(hours=1)
        print(f"\nError Statistics:")
        print(f"- Total errors: {stats['total_errors']}")
        print(f"- Categories: {stats['categories']}")
        print(f"- Health status: {stats['health_status']['status']}")
    
    if args.report:
        print("Generating resilience report...")
        report = recovery_manager.generate_resilience_report("resilience_report.md")
        print("Report saved to resilience_report.md")
        print("\nReport preview:")
        print(report[:500] + "..." if len(report) > 500 else report)


if __name__ == "__main__":
    main()