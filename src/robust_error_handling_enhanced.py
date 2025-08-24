#!/usr/bin/env python3
"""
Enhanced Robust Error Handling System v2.0 - Generation 2: MAKE IT ROBUST

Comprehensive error handling, validation, and resilience patterns for the
autonomous SDLC system with progressive enhancement.

Features:
- Circuit breaker pattern for external service calls
- Exponential backoff with jitter for retries
- Comprehensive input validation with custom validators
- Structured error reporting and metrics
- Health monitoring and self-healing capabilities
- Security-first error handling (no sensitive data leakage)
"""

import asyncio
import functools
import json
import logging
import random
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for classification and handling."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better classification and handling."""
    VALIDATION = "validation"
    NETWORK = "network"
    SECURITY = "security"
    PERFORMANCE = "performance"
    SYSTEM = "system"
    BUSINESS_LOGIC = "business_logic"
    EXTERNAL_SERVICE = "external_service"


@dataclass
class ErrorContext:
    """Context information for error tracking and analysis."""
    timestamp: float = field(default_factory=time.time)
    function_name: str = ""
    module_name: str = ""
    error_id: str = ""
    user_context: Dict[str, Any] = field(default_factory=dict)
    system_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StructuredError:
    """Structured error representation for consistent handling."""
    message: str
    category: ErrorCategory
    severity: ErrorSeverity
    context: ErrorContext
    original_exception: Optional[Exception] = None
    suggested_actions: List[str] = field(default_factory=list)
    is_retryable: bool = False
    retry_after: Optional[float] = None


class CircuitBreakerState(Enum):
    """Circuit breaker states for external service resilience."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3
    timeout: float = 30.0


class CircuitBreaker:
    """Circuit breaker implementation for external service calls."""

    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.next_attempt_time = 0

    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == CircuitBreakerState.OPEN:
            if time.time() < self.next_attempt_time:
                raise StructuredError(
                    message=f"Circuit breaker {self.name} is OPEN",
                    category=ErrorCategory.EXTERNAL_SERVICE,
                    severity=ErrorSeverity.HIGH,
                    context=ErrorContext(
                        function_name=func.__name__,
                        system_context={"circuit_breaker_state": self.state.value}
                    ),
                    is_retryable=True,
                    retry_after=self.next_attempt_time - time.time()
                )
            else:
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise StructuredError(
                message=f"Circuit breaker {self.name} detected failure",
                category=ErrorCategory.EXTERNAL_SERVICE,
                severity=ErrorSeverity.MEDIUM,
                context=ErrorContext(
                    function_name=func.__name__,
                    system_context={"circuit_breaker_state": self.state.value}
                ),
                original_exception=e,
                is_retryable=True
            )

    def _on_success(self):
        """Handle successful operation."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0

    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            self.next_attempt_time = time.time() + self.config.recovery_timeout


class RetryConfig:
    """Configuration for retry behavior with exponential backoff."""

    def __init__(self,
                 max_attempts: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 exponential_base: float = 2.0,
                 jitter: bool = True):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt with exponential backoff and jitter."""
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            # Add jitter to prevent thundering herd
            delay *= (0.5 + random.random() * 0.5)

        return delay


def with_retry(config: RetryConfig):
    """Decorator for automatic retry with exponential backoff."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except StructuredError as e:
                    last_exception = e
                    if not e.is_retryable or attempt == config.max_attempts - 1:
                        raise

                    delay = config.get_delay(attempt)
                    logger.warning(f"Retry attempt {attempt + 1}/{config.max_attempts} "
                                 f"for {func.__name__} in {delay:.2f}s")
                    time.sleep(delay)
                except Exception as e:
                    last_exception = e
                    if attempt == config.max_attempts - 1:
                        raise StructuredError(
                            message=f"Failed after {config.max_attempts} attempts",
                            category=ErrorCategory.SYSTEM,
                            severity=ErrorSeverity.HIGH,
                            context=ErrorContext(function_name=func.__name__),
                            original_exception=e,
                            is_retryable=False
                        )

                    delay = config.get_delay(attempt)
                    logger.warning(f"Retry attempt {attempt + 1}/{config.max_attempts} "
                                 f"for {func.__name__} in {delay:.2f}s")
                    time.sleep(delay)

            raise last_exception
        return wrapper
    return decorator


class ValidationError(Exception):
    """Custom validation error with detailed context."""

    def __init__(self, field: str, value: Any, constraint: str, context: Dict[str, Any] = None):
        self.field = field
        self.value = value
        self.constraint = constraint
        self.context = context or {}
        super().__init__(f"Validation failed for field '{field}': {constraint}")


class InputValidator:
    """Comprehensive input validation with security considerations."""

    @staticmethod
    def validate_string(value: Any, field_name: str,
                       min_length: int = 0, max_length: int = 1000,
                       pattern: Optional[str] = None,
                       allowed_chars: Optional[str] = None) -> str:
        """Validate string input with length and character constraints."""
        if not isinstance(value, str):
            raise ValidationError(field_name, value, "must be a string")

        if len(value) < min_length:
            raise ValidationError(field_name, value, f"minimum length is {min_length}")

        if len(value) > max_length:
            raise ValidationError(field_name, value, f"maximum length is {max_length}")

        if pattern:
            import re
            if not re.match(pattern, value):
                raise ValidationError(field_name, value, f"must match pattern: {pattern}")

        if allowed_chars:
            invalid_chars = set(value) - set(allowed_chars)
            if invalid_chars:
                raise ValidationError(field_name, value,
                                    f"contains invalid characters: {invalid_chars}")

        return value

    @staticmethod
    def validate_number(value: Any, field_name: str,
                       min_value: Optional[float] = None,
                       max_value: Optional[float] = None,
                       integer_only: bool = False) -> Union[int, float]:
        """Validate numeric input with range constraints."""
        if not isinstance(value, (int, float)):
            try:
                value = float(value)
            except (ValueError, TypeError):
                raise ValidationError(field_name, value, "must be a number")

        if integer_only and not isinstance(value, int):
            if value != int(value):
                raise ValidationError(field_name, value, "must be an integer")
            value = int(value)

        if min_value is not None and value < min_value:
            raise ValidationError(field_name, value, f"minimum value is {min_value}")

        if max_value is not None and value > max_value:
            raise ValidationError(field_name, value, f"maximum value is {max_value}")

        return value

    @staticmethod
    def validate_enum(value: Any, field_name: str, enum_class: Type[Enum]) -> Enum:
        """Validate enum input."""
        if isinstance(value, enum_class):
            return value

        if isinstance(value, str):
            try:
                return enum_class(value)
            except ValueError:
                pass

        valid_values = [e.value for e in enum_class]
        raise ValidationError(field_name, value, f"must be one of: {valid_values}")


class ErrorTracker:
    """Track and analyze errors for system health monitoring."""

    def __init__(self):
        self.errors: List[StructuredError] = []
        self.error_counts: Dict[str, int] = {}
        self.last_cleanup = time.time()
        self.cleanup_interval = 3600  # 1 hour

    def record_error(self, error: StructuredError):
        """Record an error for tracking and analysis."""
        self.errors.append(error)

        # Update counts
        key = f"{error.category.value}:{error.severity.value}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1

        # Periodic cleanup
        if time.time() - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_errors()

    def _cleanup_old_errors(self):
        """Remove old errors to prevent memory bloat."""
        cutoff_time = time.time() - 86400  # 24 hours
        self.errors = [e for e in self.errors if e.context.timestamp > cutoff_time]
        self.last_cleanup = time.time()

    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary for health monitoring."""
        recent_errors = [e for e in self.errors
                        if e.context.timestamp > time.time() - 3600]  # Last hour

        return {
            "total_errors": len(self.errors),
            "recent_errors": len(recent_errors),
            "error_counts": self.error_counts.copy(),
            "critical_errors": len([e for e in recent_errors
                                  if e.severity == ErrorSeverity.CRITICAL]),
            "health_status": self._calculate_health_status(recent_errors)
        }

    def _calculate_health_status(self, recent_errors: List[StructuredError]) -> str:
        """Calculate overall system health status."""
        if not recent_errors:
            return "healthy"

        critical_count = len([e for e in recent_errors
                            if e.severity == ErrorSeverity.CRITICAL])
        high_count = len([e for e in recent_errors
                        if e.severity == ErrorSeverity.HIGH])

        if critical_count > 0:
            return "critical"
        elif high_count > 5:
            return "degraded"
        elif len(recent_errors) > 20:
            return "warning"
        else:
            return "healthy"


# Global error tracker instance
error_tracker = ErrorTracker()


@contextmanager
def error_handling_context(function_name: str, **context):
    """Context manager for structured error handling."""
    try:
        yield
    except StructuredError as e:
        # Already structured, just record and re-raise
        error_tracker.record_error(e)
        logger.error(f"Structured error in {function_name}: {e.message}")
        raise
    except ValidationError as e:
        # Convert validation error to structured error
        structured_error = StructuredError(
            message=str(e),
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            context=ErrorContext(
                function_name=function_name,
                user_context=context
            ),
            original_exception=e,
            suggested_actions=["Check input validation requirements"],
            is_retryable=False
        )
        error_tracker.record_error(structured_error)
        logger.error(f"Validation error in {function_name}: {e}")
        raise structured_error
    except Exception as e:
        # Convert generic exception to structured error
        structured_error = StructuredError(
            message=f"Unexpected error: {str(e)}",
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.HIGH,
            context=ErrorContext(
                function_name=function_name,
                system_context=context,
                user_context={"traceback": traceback.format_exc()}
            ),
            original_exception=e,
            suggested_actions=["Check logs for details", "Contact support if issue persists"],
            is_retryable=True
        )
        error_tracker.record_error(structured_error)
        logger.error(f"Unexpected error in {function_name}: {e}", exc_info=True)
        raise structured_error


def robust_function(category: ErrorCategory = ErrorCategory.SYSTEM,
                   retry_config: Optional[RetryConfig] = None):
    """Decorator for robust function execution with comprehensive error handling."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with error_handling_context(func.__name__,
                                      function_args=str(args)[:200],
                                      function_kwargs=str(kwargs)[:200]):
                if retry_config:
                    return with_retry(retry_config)(func)(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
        return wrapper
    return decorator


class HealthMonitor:
    """System health monitoring with self-healing capabilities."""

    def __init__(self):
        self.last_health_check = 0
        self.health_check_interval = 300  # 5 minutes
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}

    def register_circuit_breaker(self, name: str, config: CircuitBreakerConfig) -> CircuitBreaker:
        """Register a new circuit breaker for monitoring."""
        cb = CircuitBreaker(name, config)
        self.circuit_breakers[name] = cb
        return cb

    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        current_time = time.time()

        health_data = {
            "timestamp": current_time,
            "overall_status": "unknown",
            "error_summary": error_tracker.get_error_summary(),
            "circuit_breakers": {
                name: {
                    "state": cb.state.value,
                    "failure_count": cb.failure_count,
                    "success_count": cb.success_count
                }
                for name, cb in self.circuit_breakers.items()
            },
            "recommendations": []
        }

        # Determine overall health status
        error_summary = health_data["error_summary"]
        error_health = error_summary["health_status"]

        circuit_breaker_issues = sum(1 for cb in self.circuit_breakers.values()
                                   if cb.state != CircuitBreakerState.CLOSED)

        if error_health == "critical" or circuit_breaker_issues > 2:
            health_data["overall_status"] = "critical"
            health_data["recommendations"].extend([
                "Immediate attention required",
                "Check error logs and circuit breaker status",
                "Consider scaling resources or restarting services"
            ])
        elif error_health == "degraded" or circuit_breaker_issues > 0:
            health_data["overall_status"] = "degraded"
            health_data["recommendations"].extend([
                "Monitor system closely",
                "Check for resource constraints",
                "Review recent changes"
            ])
        elif error_health == "warning":
            health_data["overall_status"] = "warning"
            health_data["recommendations"].append("Continue monitoring")
        else:
            health_data["overall_status"] = "healthy"
            health_data["recommendations"].append("System operating normally")

        return health_data

    async def run_health_checks(self):
        """Run periodic health checks and self-healing actions."""
        while True:
            try:
                health_data = self.get_system_health()

                if health_data["overall_status"] in ["critical", "degraded"]:
                    await self._trigger_self_healing(health_data)

                # Log health status
                logger.info(f"System health: {health_data['overall_status']}")

                await asyncio.sleep(self.health_check_interval)

            except Exception as e:
                logger.error(f"Health check failed: {e}")
                await asyncio.sleep(60)  # Shorter interval on failure

    async def _trigger_self_healing(self, health_data: Dict[str, Any]):
        """Trigger self-healing actions based on health status."""
        logger.warning(f"Triggering self-healing for status: {health_data['overall_status']}")

        # Reset circuit breakers that have been open for too long
        for name, cb in self.circuit_breakers.items():
            if (cb.state == CircuitBreakerState.OPEN and
                time.time() - cb.last_failure_time > cb.config.recovery_timeout * 2):
                cb.state = CircuitBreakerState.HALF_OPEN
                cb.success_count = 0
                logger.info(f"Reset circuit breaker: {name}")

        # Additional self-healing actions can be added here
        # - Clear caches
        # - Restart background tasks
        # - Scale resources
        # - Notify administrators


# Global health monitor instance
health_monitor = HealthMonitor()


def save_error_report(output_file: str = "error_analysis_report.json"):
    """Save comprehensive error analysis report."""
    health_data = health_monitor.get_system_health()

    report = {
        "robust_error_handling": {
            "version": "2.0",
            "generation": "make_it_robust",
            "health_status": health_data,
            "features_enabled": {
                "circuit_breakers": True,
                "retry_with_backoff": True,
                "input_validation": True,
                "structured_errors": True,
                "health_monitoring": True,
                "self_healing": True
            },
            "error_handling_capabilities": {
                "error_categories": [e.value for e in ErrorCategory],
                "severity_levels": [e.value for e in ErrorSeverity],
                "validation_types": ["string", "number", "enum"],
                "circuit_breaker_states": [s.value for s in CircuitBreakerState]
            }
        }
    }

    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"Error analysis report saved to {output_file}")


if __name__ == "__main__":
    # Example usage and testing
    @robust_function(category=ErrorCategory.VALIDATION)
    def example_function(name: str, age: int):
        """Example function with robust error handling."""
        name = InputValidator.validate_string(name, "name", min_length=1, max_length=50)
        age = InputValidator.validate_number(age, "age", min_value=0, max_value=150, integer_only=True)

        return f"Hello {name}, you are {age} years old"

    # Test the robust error handling
    try:
        result = example_function("Terry", 25)
        print(result)
    except StructuredError as e:
        print(f"Structured error: {e.message}")

    # Generate health report
    save_error_report()
