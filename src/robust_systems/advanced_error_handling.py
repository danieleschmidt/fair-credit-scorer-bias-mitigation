"""
Advanced Error Handling and Recovery System.

This module implements comprehensive error handling, validation, and recovery
mechanisms for robust fairness-aware machine learning systems.

Features:
- Circuit breaker pattern for fault tolerance
- Advanced input validation with schema enforcement
- Automatic retry mechanisms with exponential backoff
- Error recovery strategies and fallback models
- Comprehensive logging and monitoring integration
- Security validation and sanitization
"""

import functools
import logging
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union
import traceback
import hashlib
import json
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning

try:
    from ..logging_config import get_logger
except ImportError:
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing if service recovered

class ValidationResult(Enum):
    """Validation result types."""
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    SANITIZED = "sanitized"

@dataclass
class ErrorContext:
    """Context information for errors."""
    timestamp: float = field(default_factory=time.time)
    function_name: str = ""
    input_data: Optional[Dict[str, Any]] = None
    stack_trace: str = ""
    error_id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:8])
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    recovery_attempted: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class ValidationReport:
    """Report from validation process."""
    is_valid: bool
    result_type: ValidationResult
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sanitizations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class FairnessError(Exception):
    """Base exception for fairness-related errors."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.context = context or ErrorContext()
        self.original_error = original_error
        self.message = message

class DataValidationError(FairnessError):
    """Error in data validation."""
    pass

class ModelTrainingError(FairnessError):
    """Error during model training."""
    pass

class BiasDetectionError(FairnessError):
    """Error in bias detection."""
    pass

class SecurityViolationError(FairnessError):
    """Security validation violation."""
    pass

class PerformanceDegradationError(FairnessError):
    """Performance degradation detected."""
    pass


class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance.
    
    Prevents cascading failures by temporarily stopping calls to failing services
    and providing fallback mechanisms.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception,
        fallback_function: Optional[Callable] = None
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before attempting recovery
            expected_exception: Exception type that triggers circuit breaking
            fallback_function: Function to call when circuit is open
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.fallback_function = fallback_function
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self.lock = threading.RLock()
        
        logger.info(f"CircuitBreaker initialized with threshold={failure_threshold}")

    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply circuit breaker to a function."""
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self.lock:
                # Check circuit state
                if self.state == CircuitBreakerState.OPEN:
                    if self._should_attempt_reset():
                        self.state = CircuitBreakerState.HALF_OPEN
                        logger.info("Circuit breaker moving to HALF_OPEN state")
                    else:
                        logger.warning(f"Circuit breaker OPEN, calling fallback for {func.__name__}")
                        return self._call_fallback(*args, **kwargs)
                
                try:
                    # Attempt the function call
                    result = func(*args, **kwargs)
                    
                    # Success - reset failure count
                    if self.state == CircuitBreakerState.HALF_OPEN:
                        self._reset()
                        logger.info("Circuit breaker CLOSED after successful call")
                    
                    return result
                    
                except self.expected_exception as e:
                    return self._handle_failure(func.__name__, e, *args, **kwargs)
        
        return wrapper

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.recovery_timeout

    def _handle_failure(self, func_name: str, exception: Exception, *args, **kwargs):
        """Handle function failure."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        logger.warning(f"Circuit breaker failure {self.failure_count}/{self.failure_threshold} for {func_name}: {exception}")
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.error(f"Circuit breaker OPENED for {func_name}")
        
        # If in HALF_OPEN and failed, go back to OPEN
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            logger.warning("Circuit breaker returned to OPEN state after failure in HALF_OPEN")
        
        return self._call_fallback(*args, **kwargs)

    def _call_fallback(self, *args, **kwargs):
        """Call fallback function or raise error."""
        if self.fallback_function:
            try:
                return self.fallback_function(*args, **kwargs)
            except Exception as e:
                logger.error(f"Fallback function also failed: {e}")
                raise FairnessError(f"Both primary function and fallback failed", 
                                   original_error=e)
        else:
            raise FairnessError("Circuit breaker OPEN and no fallback available")

    def _reset(self):
        """Reset circuit breaker to closed state."""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED

    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'failure_threshold': self.failure_threshold,
            'last_failure_time': self.last_failure_time,
            'time_since_failure': time.time() - self.last_failure_time if self.last_failure_time else None
        }


class RetryHandler:
    """
    Advanced retry mechanism with exponential backoff.
    
    Provides intelligent retry logic with configurable backoff strategies
    and exception filtering.
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_multiplier: float = 2.0,
        retryable_exceptions: List[Type[Exception]] = None,
        jitter: bool = True
    ):
        """
        Initialize retry handler.
        
        Args:
            max_attempts: Maximum number of retry attempts
            base_delay: Base delay between retries (seconds)
            max_delay: Maximum delay between retries (seconds)
            backoff_multiplier: Multiplier for exponential backoff
            retryable_exceptions: List of exception types to retry on
            jitter: Add random jitter to prevent thundering herd
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.retryable_exceptions = retryable_exceptions or [Exception]
        self.jitter = jitter
        
        logger.info(f"RetryHandler initialized with {max_attempts} max attempts")

    def __call__(self, func: Callable) -> Callable:
        """Decorator to add retry logic to a function."""
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(self.max_attempts):
                try:
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    last_exception = e
                    
                    # Check if this exception should trigger a retry
                    if not self._should_retry(e):
                        logger.info(f"Exception {type(e).__name__} not retryable, failing immediately")
                        raise
                    
                    if attempt == self.max_attempts - 1:
                        logger.error(f"All {self.max_attempts} attempts failed for {func.__name__}")
                        raise FairnessError(
                            f"Function {func.__name__} failed after {self.max_attempts} attempts",
                            original_error=e
                        )
                    
                    # Calculate delay for next attempt
                    delay = self._calculate_delay(attempt)
                    logger.warning(f"Attempt {attempt + 1}/{self.max_attempts} failed for {func.__name__}, "
                                 f"retrying in {delay:.2f}s: {e}")
                    
                    time.sleep(delay)
            
            # Should never reach here, but just in case
            raise FairnessError("Unexpected retry handler state", original_error=last_exception)
        
        return wrapper

    def _should_retry(self, exception: Exception) -> bool:
        """Check if exception should trigger a retry."""
        return any(isinstance(exception, exc_type) for exc_type in self.retryable_exceptions)

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt."""
        delay = self.base_delay * (self.backoff_multiplier ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add random jitter ±20%
            jitter_range = delay * 0.2
            jitter = np.random.uniform(-jitter_range, jitter_range)
            delay += jitter
        
        return max(0.1, delay)  # Minimum 0.1s delay


class DataValidator:
    """
    Advanced data validation with schema enforcement and security checks.
    
    Provides comprehensive validation for ML datasets including:
    - Schema validation
    - Statistical anomaly detection
    - Security vulnerability checks
    - Data quality assessment
    """

    def __init__(self, strict_mode: bool = True, auto_sanitize: bool = True):
        """
        Initialize data validator.
        
        Args:
            strict_mode: Whether to fail on warnings
            auto_sanitize: Whether to automatically sanitize input data
        """
        self.strict_mode = strict_mode
        self.auto_sanitize = auto_sanitize
        
        # Security patterns to detect
        self.security_patterns = [
            (r'<script.*?>', "Script injection attempt"),
            (r'javascript:', "JavaScript protocol"),
            (r'eval\s*\(', "Eval function call"),
            (r'exec\s*\(', "Exec function call"),
            (r'__import__', "Import injection"),
            (r'\.\./', "Path traversal"),
            (r'DROP\s+TABLE', "SQL injection"),
            (r'UNION\s+SELECT', "SQL union injection"),
        ]
        
        logger.info("DataValidator initialized")

    def validate_dataframe(self, df: pd.DataFrame, schema: Optional[Dict[str, Any]] = None) -> ValidationReport:
        """
        Comprehensive DataFrame validation.
        
        Args:
            df: DataFrame to validate
            schema: Expected schema (optional)
            
        Returns:
            Validation report
        """
        report = ValidationReport(is_valid=True, result_type=ValidationResult.VALID)
        
        try:
            # Basic structure validation
            self._validate_structure(df, report)
            
            # Schema validation if provided
            if schema:
                self._validate_schema(df, schema, report)
            
            # Security validation
            self._validate_security(df, report)
            
            # Statistical validation
            self._validate_statistics(df, report)
            
            # Data quality validation
            self._validate_quality(df, report)
            
            # Determine final validation result
            if report.errors:
                report.is_valid = False
                report.result_type = ValidationResult.INVALID
            elif report.warnings and self.strict_mode:
                report.is_valid = False
                report.result_type = ValidationResult.WARNING
            elif report.sanitizations:
                report.result_type = ValidationResult.SANITIZED
            
        except Exception as e:
            report.is_valid = False
            report.result_type = ValidationResult.INVALID
            report.errors.append(f"Validation failed with exception: {str(e)}")
            logger.error(f"Data validation failed: {e}")
        
        return report

    def _validate_structure(self, df: pd.DataFrame, report: ValidationReport):
        """Validate basic DataFrame structure."""
        if df.empty:
            report.errors.append("DataFrame is empty")
            return
        
        if df.shape[0] < 10:
            report.warnings.append(f"DataFrame has very few rows: {df.shape[0]}")
        
        if df.shape[1] == 0:
            report.errors.append("DataFrame has no columns")
        
        # Check for duplicate columns
        duplicate_columns = df.columns[df.columns.duplicated()].tolist()
        if duplicate_columns:
            report.errors.append(f"Duplicate columns detected: {duplicate_columns}")
        
        report.metadata['shape'] = df.shape
        report.metadata['memory_usage'] = df.memory_usage().sum()

    def _validate_schema(self, df: pd.DataFrame, schema: Dict[str, Any], report: ValidationReport):
        """Validate DataFrame against expected schema."""
        required_columns = schema.get('required_columns', [])
        column_types = schema.get('column_types', {})
        value_ranges = schema.get('value_ranges', {})
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            report.errors.append(f"Missing required columns: {missing_columns}")
        
        # Check column types
        for col, expected_type in column_types.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if not self._is_compatible_type(actual_type, expected_type):
                    report.warnings.append(f"Column {col} has type {actual_type}, expected {expected_type}")
        
        # Check value ranges
        for col, value_range in value_ranges.items():
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                min_val, max_val = value_range
                col_min, col_max = df[col].min(), df[col].max()
                
                if col_min < min_val or col_max > max_val:
                    report.warnings.append(f"Column {col} values outside expected range [{min_val}, {max_val}], "
                                         f"actual range [{col_min}, {col_max}]")

    def _validate_security(self, df: pd.DataFrame, report: ValidationReport):
        """Validate DataFrame for security vulnerabilities."""
        security_issues = []
        
        for col in df.select_dtypes(include=['object']).columns:
            for idx, value in df[col].items():
                if pd.isna(value):
                    continue
                
                value_str = str(value).lower()
                
                # Check for security patterns
                for pattern, description in self.security_patterns:
                    if re.search(pattern, value_str, re.IGNORECASE):
                        security_issues.append(f"Security risk in {col}[{idx}]: {description}")
                        
                        if self.auto_sanitize:
                            # Simple sanitization - remove the pattern
                            df.at[idx, col] = re.sub(pattern, '', str(value), flags=re.IGNORECASE)
                            report.sanitizations.append(f"Sanitized {col}[{idx}]: removed {description}")
        
        if security_issues:
            if not self.auto_sanitize:
                report.errors.extend(security_issues)
            logger.warning(f"Detected {len(security_issues)} potential security issues")

    def _validate_statistics(self, df: pd.DataFrame, report: ValidationReport):
        """Validate statistical properties of the data."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Check for infinite values
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                report.warnings.append(f"Column {col} contains {inf_count} infinite values")
            
            # Check for extreme outliers (beyond 6 standard deviations)
            if df[col].std() > 0:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                extreme_outliers = (z_scores > 6).sum()
                if extreme_outliers > 0:
                    report.warnings.append(f"Column {col} has {extreme_outliers} extreme outliers")
            
            # Check for constant columns
            if df[col].nunique() <= 1:
                report.warnings.append(f"Column {col} has constant or single unique value")

    def _validate_quality(self, df: pd.DataFrame, report: ValidationReport):
        """Validate data quality metrics."""
        quality_metrics = {}
        
        for col in df.columns:
            # Missing value percentage
            missing_pct = df[col].isnull().sum() / len(df) * 100
            quality_metrics[f'{col}_missing_pct'] = missing_pct
            
            if missing_pct > 50:
                report.errors.append(f"Column {col} has {missing_pct:.1f}% missing values")
            elif missing_pct > 20:
                report.warnings.append(f"Column {col} has {missing_pct:.1f}% missing values")
            
            # Unique value percentage for categorical columns
            if df[col].dtype == 'object':
                unique_pct = df[col].nunique() / len(df) * 100
                quality_metrics[f'{col}_unique_pct'] = unique_pct
                
                if unique_pct > 95:
                    report.warnings.append(f"Column {col} has {unique_pct:.1f}% unique values (high cardinality)")
        
        report.metadata['quality_metrics'] = quality_metrics

    def _is_compatible_type(self, actual_type: str, expected_type: str) -> bool:
        """Check if actual type is compatible with expected type."""
        type_mappings = {
            'int': ['int64', 'int32', 'int16', 'int8'],
            'float': ['float64', 'float32', 'int64', 'int32'],
            'string': ['object'],
            'bool': ['bool'],
            'datetime': ['datetime64']
        }
        
        expected_compatible = type_mappings.get(expected_type, [expected_type])
        return any(compatible in actual_type for compatible in expected_compatible)

    def validate_model_input(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None) -> ValidationReport:
        """
        Validate model input data.
        
        Args:
            X: Feature matrix
            y: Target vector (optional)
            
        Returns:
            Validation report
        """
        report = ValidationReport(is_valid=True, result_type=ValidationResult.VALID)
        
        try:
            # Convert to DataFrame if needed
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
            
            # Validate features
            feature_report = self.validate_dataframe(X)
            report.errors.extend(feature_report.errors)
            report.warnings.extend(feature_report.warnings)
            report.sanitizations.extend(feature_report.sanitizations)
            
            # Validate target if provided
            if y is not None:
                if isinstance(y, np.ndarray):
                    y = pd.Series(y, name='target')
                elif isinstance(y, pd.Series):
                    pass
                else:
                    report.errors.append(f"Unsupported target type: {type(y)}")
                    return report
                
                # Check target properties
                if len(y) != len(X):
                    report.errors.append(f"Feature matrix length {len(X)} doesn't match target length {len(y)}")
                
                if y.isnull().sum() > 0:
                    report.warnings.append(f"Target contains {y.isnull().sum()} null values")
                
                # For classification, check class balance
                if y.dtype == 'object' or y.nunique() <= 10:
                    class_counts = y.value_counts()
                    min_class_size = class_counts.min()
                    max_class_size = class_counts.max()
                    
                    if min_class_size < 5:
                        report.warnings.append(f"Some classes have very few samples (min: {min_class_size})")
                    
                    if max_class_size / min_class_size > 10:
                        report.warnings.append(f"Severe class imbalance detected (ratio: {max_class_size/min_class_size:.1f})")
            
            # Final validation result
            if report.errors:
                report.is_valid = False
                report.result_type = ValidationResult.INVALID
            elif report.warnings and self.strict_mode:
                report.is_valid = False
                report.result_type = ValidationResult.WARNING
            
        except Exception as e:
            report.is_valid = False
            report.result_type = ValidationResult.INVALID
            report.errors.append(f"Model input validation failed: {str(e)}")
            logger.error(f"Model input validation failed: {e}")
        
        return report


class ModelValidator:
    """
    Advanced model validation and monitoring.
    
    Validates trained models for performance, fairness, and robustness
    before deployment.
    """

    def __init__(self, performance_threshold: float = 0.7, fairness_threshold: float = 0.1):
        """
        Initialize model validator.
        
        Args:
            performance_threshold: Minimum acceptable performance
            fairness_threshold: Maximum acceptable fairness violation
        """
        self.performance_threshold = performance_threshold
        self.fairness_threshold = fairness_threshold
        
        logger.info("ModelValidator initialized")

    def validate_model(self, model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series, 
                      protected_attrs: Optional[pd.Series] = None) -> ValidationReport:
        """
        Comprehensive model validation.
        
        Args:
            model: Trained model to validate
            X_test: Test features
            y_test: Test targets
            protected_attrs: Protected attributes for fairness testing
            
        Returns:
            Validation report
        """
        report = ValidationReport(is_valid=True, result_type=ValidationResult.VALID)
        
        try:
            # Performance validation
            self._validate_performance(model, X_test, y_test, report)
            
            # Fairness validation
            if protected_attrs is not None:
                self._validate_fairness(model, X_test, y_test, protected_attrs, report)
            
            # Robustness validation
            self._validate_robustness(model, X_test, y_test, report)
            
            # Model properties validation
            self._validate_model_properties(model, report)
            
            # Final validation result
            if report.errors:
                report.is_valid = False
                report.result_type = ValidationResult.INVALID
            elif report.warnings:
                report.result_type = ValidationResult.WARNING
        
        except Exception as e:
            report.is_valid = False
            report.result_type = ValidationResult.INVALID
            report.errors.append(f"Model validation failed: {str(e)}")
            logger.error(f"Model validation failed: {e}")
        
        return report

    def _validate_performance(self, model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series, report: ValidationReport):
        """Validate model performance metrics."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        try:
            predictions = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
            recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
            f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
            
            report.metadata['performance'] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            # Check performance thresholds
            if accuracy < self.performance_threshold:
                report.errors.append(f"Model accuracy {accuracy:.3f} below threshold {self.performance_threshold}")
            
            if f1 < self.performance_threshold * 0.9:  # Slightly lower threshold for F1
                report.warnings.append(f"Model F1 score {f1:.3f} is relatively low")
            
        except Exception as e:
            report.warnings.append(f"Could not validate performance: {str(e)}")

    def _validate_fairness(self, model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series, 
                          protected_attrs: pd.Series, report: ValidationReport):
        """Validate model fairness metrics."""
        try:
            predictions = model.predict(X_test)
            
            # Simple fairness check - demographic parity
            protected_groups = protected_attrs.unique()
            if len(protected_groups) == 2:
                group_0_rate = predictions[protected_attrs == protected_groups[0]].mean()
                group_1_rate = predictions[protected_attrs == protected_groups[1]].mean()
                
                demographic_parity_diff = abs(group_0_rate - group_1_rate)
                
                report.metadata['fairness'] = {
                    'demographic_parity_difference': demographic_parity_diff,
                    'group_0_positive_rate': group_0_rate,
                    'group_1_positive_rate': group_1_rate
                }
                
                if demographic_parity_diff > self.fairness_threshold:
                    report.errors.append(f"Demographic parity violation: {demographic_parity_diff:.3f} > {self.fairness_threshold}")
                elif demographic_parity_diff > self.fairness_threshold * 0.5:
                    report.warnings.append(f"Potential fairness concern: demographic parity diff {demographic_parity_diff:.3f}")
            
        except Exception as e:
            report.warnings.append(f"Could not validate fairness: {str(e)}")

    def _validate_robustness(self, model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series, report: ValidationReport):
        """Validate model robustness."""
        try:
            # Test with slightly perturbed data
            X_perturbed = X_test.copy()
            
            # Add small amount of noise to numeric columns
            numeric_cols = X_test.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                noise_level = 0.01  # 1% noise
                for col in numeric_cols[:min(5, len(numeric_cols))]:  # Test up to 5 columns
                    col_std = X_test[col].std()
                    if col_std > 0:
                        noise = np.random.normal(0, col_std * noise_level, len(X_test))
                        X_perturbed[col] = X_test[col] + noise
                
                # Compare predictions
                original_preds = model.predict(X_test)
                perturbed_preds = model.predict(X_perturbed)
                
                # Calculate prediction stability
                stability = (original_preds == perturbed_preds).mean()
                
                report.metadata['robustness'] = {
                    'prediction_stability': stability,
                    'noise_level': noise_level
                }
                
                if stability < 0.9:
                    report.warnings.append(f"Model predictions unstable with noise: {stability:.3f} stability")
        
        except Exception as e:
            report.warnings.append(f"Could not validate robustness: {str(e)}")

    def _validate_model_properties(self, model: BaseEstimator, report: ValidationReport):
        """Validate model properties and configuration."""
        try:
            model_info = {
                'model_type': type(model).__name__,
                'has_predict': hasattr(model, 'predict'),
                'has_predict_proba': hasattr(model, 'predict_proba'),
                'is_fitted': hasattr(model, 'classes_') or hasattr(model, 'coef_') or hasattr(model, 'feature_importances_')
            }
            
            report.metadata['model_properties'] = model_info
            
            if not model_info['has_predict']:
                report.errors.append("Model does not have predict method")
            
            if not model_info['is_fitted']:
                report.errors.append("Model does not appear to be fitted")
        
        except Exception as e:
            report.warnings.append(f"Could not validate model properties: {str(e)}")


def create_robust_training_pipeline():
    """
    Create a robust training pipeline with error handling and validation.
    
    Returns:
        Configured pipeline with error handling decorators
    """
    # Circuit breaker for model training
    training_circuit_breaker = CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=300.0,  # 5 minutes
        expected_exception=ModelTrainingError
    )
    
    # Retry handler for data loading
    data_loading_retry = RetryHandler(
        max_attempts=3,
        base_delay=2.0,
        retryable_exceptions=[IOError, ConnectionError, TimeoutError]
    )
    
    # Data validator
    data_validator = DataValidator(strict_mode=False, auto_sanitize=True)
    
    # Model validator
    model_validator = ModelValidator(
        performance_threshold=0.7,
        fairness_threshold=0.1
    )
    
    return {
        'circuit_breaker': training_circuit_breaker,
        'retry_handler': data_loading_retry,
        'data_validator': data_validator,
        'model_validator': model_validator
    }


# Example usage and testing
def demonstrate_robust_systems():
    """Demonstrate robust error handling and validation systems."""
    print("🛡️ Robust Systems Demonstration")
    
    # Create sample data with issues
    np.random.seed(42)
    
    # Create problematic data
    data = {
        'feature1': [1, 2, np.inf, 4, 5],  # Contains infinity
        'feature2': [1.0, 2.0, 3.0, None, 5.0],  # Contains null
        'feature3': ['safe', 'also_safe', '<script>alert("xss")</script>', 'DROP TABLE users', 'normal'],  # Security issues
        'protected': [0, 1, 0, 1, 0],
        'target': [0, 1, 1, 0, 1]
    }
    
    df = pd.DataFrame(data)
    
    print(f"\n📊 Test dataset: {df.shape}")
    print(f"   Contains: infinity, nulls, security patterns")
    
    # Test data validator
    print("\n🔍 Testing Data Validator...")
    validator = DataValidator(strict_mode=False, auto_sanitize=True)
    
    schema = {
        'required_columns': ['feature1', 'feature2', 'target'],
        'column_types': {'feature1': 'int', 'feature2': 'float'},
        'value_ranges': {'feature1': (0, 10), 'feature2': (0, 10)}
    }
    
    validation_report = validator.validate_dataframe(df, schema)
    
    print(f"   Validation result: {validation_report.result_type.value}")
    print(f"   Errors: {len(validation_report.errors)}")
    print(f"   Warnings: {len(validation_report.warnings)}")
    print(f"   Sanitizations: {len(validation_report.sanitizations)}")
    
    if validation_report.errors:
        print("   Error details:")
        for error in validation_report.errors[:3]:  # Show first 3
            print(f"     - {error}")
    
    if validation_report.sanitizations:
        print("   Sanitization details:")
        for sanit in validation_report.sanitizations[:3]:
            print(f"     - {sanit}")
    
    # Test circuit breaker
    print("\n⚡ Testing Circuit Breaker...")
    
    @CircuitBreaker(failure_threshold=2, recovery_timeout=1.0)
    def unreliable_function(should_fail=True):
        if should_fail:
            raise ValueError("Simulated failure")
        return "Success!"
    
    # Test circuit breaker behavior
    for i in range(5):
        try:
            result = unreliable_function(should_fail=(i < 3))  # First 3 calls fail
            print(f"   Call {i+1}: {result}")
        except Exception as e:
            print(f"   Call {i+1}: Failed - {type(e).__name__}")
    
    # Test retry handler
    print("\n🔄 Testing Retry Handler...")
    
    attempt_count = 0
    
    @RetryHandler(max_attempts=3, base_delay=0.1)
    def flaky_function():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ConnectionError(f"Connection failed on attempt {attempt_count}")
        return f"Success after {attempt_count} attempts!"
    
    try:
        result = flaky_function()
        print(f"   Result: {result}")
    except Exception as e:
        print(f"   Failed: {e}")
    
    # Test model validator (simplified)
    print("\n🎯 Testing Model Validator...")
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    
    # Create clean data for model testing
    X_clean = df[['feature1', 'feature2']].fillna(0)
    X_clean.loc[2, 'feature1'] = 3  # Fix infinity
    y_clean = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.4, random_state=42)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    model_validator = ModelValidator(performance_threshold=0.6, fairness_threshold=0.2)
    model_report = model_validator.validate_model(model, X_test, y_test, df['protected'].iloc[X_test.index])
    
    print(f"   Model validation result: {model_report.result_type.value}")
    print(f"   Errors: {len(model_report.errors)}")
    print(f"   Warnings: {len(model_report.warnings)}")
    
    if 'performance' in model_report.metadata:
        perf = model_report.metadata['performance']
        print(f"   Model accuracy: {perf['accuracy']:.3f}")
    
    if 'fairness' in model_report.metadata:
        fair = model_report.metadata['fairness']
        print(f"   Demographic parity diff: {fair['demographic_parity_difference']:.3f}")
    
    print("\n✅ Robust systems demonstration completed! 🛡️")


if __name__ == "__main__":
    demonstrate_robust_systems()