"""
Data validation and quality checking modules.

This module provides comprehensive data validation capabilities including
schema validation, data quality checks, and fairness-specific validations.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np
from scipy import stats

from ..logging_config import get_logger

logger = get_logger(__name__)


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Container for validation results."""
    rule_name: str
    severity: ValidationSeverity
    passed: bool
    message: str
    details: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'rule_name': self.rule_name,
            'severity': self.severity.value,
            'passed': self.passed,
            'message': self.message,
            'details': self.details or {}
        }


class ValidationRule:
    """Base class for validation rules."""
    
    def __init__(self, name: str, severity: ValidationSeverity = ValidationSeverity.ERROR):
        """
        Initialize validation rule.
        
        Args:
            name: Name of the validation rule
            severity: Severity level for failures
        """
        self.name = name
        self.severity = severity
    
    def validate(self, data: pd.DataFrame, **kwargs) -> ValidationResult:
        """
        Validate data against this rule.
        
        Args:
            data: DataFrame to validate
            **kwargs: Additional validation parameters
            
        Returns:
            ValidationResult
        """
        try:
            passed, message, details = self._check(data, **kwargs)
            
            return ValidationResult(
                rule_name=self.name,
                severity=self.severity,
                passed=passed,
                message=message,
                details=details
            )
            
        except Exception as e:
            logger.error(f"Validation rule {self.name} failed with error: {e}")
            return ValidationResult(
                rule_name=self.name,
                severity=ValidationSeverity.CRITICAL,
                passed=False,
                message=f"Validation error: {str(e)}",
                details={'exception': str(e)}
            )
    
    def _check(self, data: pd.DataFrame, **kwargs) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Implement the actual validation logic.
        
        Args:
            data: DataFrame to validate
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (passed, message, details)
        """
        raise NotImplementedError("Subclasses must implement _check method")


class SchemaValidationRule(ValidationRule):
    """Validate data schema (columns, types, etc.)."""
    
    def __init__(self, expected_columns: List[str], required_columns: Optional[List[str]] = None):
        """
        Initialize schema validation.
        
        Args:
            expected_columns: List of expected column names
            required_columns: List of required column names (subset of expected)
        """
        super().__init__("schema_validation")
        self.expected_columns = set(expected_columns)
        self.required_columns = set(required_columns or expected_columns)
    
    def _check(self, data: pd.DataFrame, **kwargs) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Check data schema."""
        actual_columns = set(data.columns)
        
        # Check required columns
        missing_required = self.required_columns - actual_columns
        if missing_required:
            return False, f"Missing required columns: {missing_required}", {
                'missing_required': list(missing_required),
                'actual_columns': list(actual_columns)
            }
        
        # Check for unexpected columns
        unexpected_columns = actual_columns - self.expected_columns
        if unexpected_columns:
            return False, f"Unexpected columns found: {unexpected_columns}", {
                'unexpected_columns': list(unexpected_columns),
                'expected_columns': list(self.expected_columns)
            }
        
        return True, "Schema validation passed", {
            'validated_columns': list(actual_columns),
            'expected_columns': list(self.expected_columns)
        }


class DataTypeValidationRule(ValidationRule):
    """Validate data types of columns."""
    
    def __init__(self, expected_types: Dict[str, Union[str, type, List[Union[str, type]]]]):
        """
        Initialize data type validation.
        
        Args:
            expected_types: Mapping of column names to expected types
        """
        super().__init__("data_type_validation")
        self.expected_types = expected_types
    
    def _check(self, data: pd.DataFrame, **kwargs) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Check data types."""
        type_issues = {}
        
        for column, expected_type in self.expected_types.items():
            if column not in data.columns:
                continue
            
            actual_type = data[column].dtype
            
            # Normalize expected types to a list
            if not isinstance(expected_type, list):
                expected_type = [expected_type]
            
            # Check if actual type matches any expected type
            type_match = False
            for exp_type in expected_type:
                if isinstance(exp_type, str):
                    if exp_type == str(actual_type) or exp_type in str(actual_type):
                        type_match = True
                        break
                elif actual_type == exp_type:
                    type_match = True
                    break
            
            if not type_match:
                type_issues[column] = {
                    'actual': str(actual_type),
                    'expected': [str(t) for t in expected_type]
                }
        
        if type_issues:
            return False, f"Data type mismatches found in columns: {list(type_issues.keys())}", {
                'type_issues': type_issues
            }
        
        return True, "Data type validation passed", {
            'validated_types': {col: str(data[col].dtype) for col in self.expected_types.keys() if col in data.columns}
        }


class MissingDataValidationRule(ValidationRule):
    """Validate missing data patterns."""
    
    def __init__(self, max_missing_ratio: float = 0.1, critical_columns: Optional[List[str]] = None):
        """
        Initialize missing data validation.
        
        Args:
            max_missing_ratio: Maximum allowed ratio of missing values per column
            critical_columns: Columns that cannot have any missing values
        """
        super().__init__("missing_data_validation")
        self.max_missing_ratio = max_missing_ratio
        self.critical_columns = critical_columns or []
    
    def _check(self, data: pd.DataFrame, **kwargs) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Check missing data patterns."""
        missing_stats = data.isnull().sum()
        total_rows = len(data)
        
        issues = {}
        
        # Check critical columns
        for column in self.critical_columns:
            if column in data.columns and missing_stats[column] > 0:
                issues[column] = {
                    'type': 'critical_missing',
                    'missing_count': int(missing_stats[column]),
                    'missing_ratio': float(missing_stats[column] / total_rows)
                }
        
        # Check missing ratio for all columns
        for column in data.columns:
            missing_ratio = missing_stats[column] / total_rows
            if missing_ratio > self.max_missing_ratio:
                issues[column] = {
                    'type': 'excessive_missing',
                    'missing_count': int(missing_stats[column]),
                    'missing_ratio': float(missing_ratio),
                    'threshold': self.max_missing_ratio
                }
        
        if issues:
            return False, f"Missing data issues found in {len(issues)} columns", {
                'missing_issues': issues,
                'total_rows': total_rows
            }
        
        return True, "Missing data validation passed", {
            'missing_summary': {col: int(count) for col, count in missing_stats.items() if count > 0}
        }


class OutlierValidationRule(ValidationRule):
    """Validate outlier patterns in data."""
    
    def __init__(self, method: str = "iqr", threshold: float = 1.5, max_outlier_ratio: float = 0.05):
        """
        Initialize outlier validation.
        
        Args:
            method: Outlier detection method ('iqr', 'zscore')
            threshold: Threshold for outlier detection
            max_outlier_ratio: Maximum allowed ratio of outliers
        """
        super().__init__("outlier_validation", ValidationSeverity.WARNING)
        self.method = method
        self.threshold = threshold
        self.max_outlier_ratio = max_outlier_ratio
    
    def _check(self, data: pd.DataFrame, **kwargs) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Check for outliers in numeric columns."""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        outlier_stats = {}
        
        for column in numeric_columns:
            series = data[column].dropna()
            if len(series) == 0:
                continue
            
            if self.method == "iqr":
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.threshold * IQR
                upper_bound = Q3 + self.threshold * IQR
                outliers = series[(series < lower_bound) | (series > upper_bound)]
            
            elif self.method == "zscore":
                z_scores = np.abs(stats.zscore(series))
                outliers = series[z_scores > self.threshold]
            
            else:
                continue
            
            outlier_ratio = len(outliers) / len(series)
            
            if outlier_ratio > self.max_outlier_ratio:
                outlier_stats[column] = {
                    'outlier_count': len(outliers),
                    'outlier_ratio': float(outlier_ratio),
                    'threshold': self.max_outlier_ratio,
                    'method': self.method
                }
        
        if outlier_stats:
            return False, f"Excessive outliers found in {len(outlier_stats)} columns", {
                'outlier_stats': outlier_stats
            }
        
        return True, "Outlier validation passed", {
            'checked_columns': list(numeric_columns)
        }


class DataValidator:
    """
    Comprehensive data validator with configurable rules.
    
    Provides a framework for validating datasets against multiple rules
    with detailed reporting and severity-based filtering.
    """
    
    def __init__(self, rules: Optional[List[ValidationRule]] = None):
        """
        Initialize data validator.
        
        Args:
            rules: List of validation rules to apply
        """
        self.rules = rules or []
        self.validation_history = []
    
    def add_rule(self, rule: ValidationRule):
        """Add a validation rule."""
        self.rules.append(rule)
    
    def validate(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Validate data against all rules.
        
        Args:
            data: DataFrame to validate
            **kwargs: Additional validation parameters
            
        Returns:
            Validation report
        """
        logger.info(f"Validating data with {len(self.rules)} rules")
        
        results = []
        start_time = datetime.utcnow()
        
        # Run all validation rules
        for rule in self.rules:
            try:
                result = rule.validate(data, **kwargs)
                results.append(result)
                
                if not result.passed:
                    logger.warning(f"Validation rule '{rule.name}' failed: {result.message}")
                
            except Exception as e:
                logger.error(f"Error running validation rule '{rule.name}': {e}")
                results.append(ValidationResult(
                    rule_name=rule.name,
                    severity=ValidationSeverity.CRITICAL,
                    passed=False,
                    message=f"Rule execution failed: {str(e)}"
                ))
        
        # Compile report
        report = self._compile_report(data, results, start_time)
        self.validation_history.append(report)
        
        logger.info(f"Validation completed: {report['summary']['passed']}/{report['summary']['total']} rules passed")
        
        return report
    
    def _compile_report(self, data: pd.DataFrame, results: List[ValidationResult], start_time: datetime) -> Dict[str, Any]:
        """Compile validation results into a report."""
        passed_results = [r for r in results if r.passed]
        failed_results = [r for r in results if not r.passed]
        
        # Group by severity
        severity_counts = {}
        for severity in ValidationSeverity:
            severity_counts[severity.value] = len([r for r in failed_results if r.severity == severity])
        
        # Determine overall status
        has_critical = any(r.severity == ValidationSeverity.CRITICAL for r in failed_results)
        has_errors = any(r.severity == ValidationSeverity.ERROR for r in failed_results)
        
        if has_critical:
            overall_status = "critical"
        elif has_errors:
            overall_status = "failed"
        elif failed_results:
            overall_status = "warning"
        else:
            overall_status = "passed"
        
        return {
            "timestamp": start_time.isoformat(),
            "duration_seconds": (datetime.utcnow() - start_time).total_seconds(),
            "data_info": {
                "shape": data.shape,
                "columns": list(data.columns),
                "memory_usage_mb": data.memory_usage(deep=True).sum() / 1024**2
            },
            "summary": {
                "total": len(results),
                "passed": len(passed_results),
                "failed": len(failed_results),
                "overall_status": overall_status
            },
            "severity_breakdown": severity_counts,
            "results": [r.to_dict() for r in results],
            "failed_rules": [r.to_dict() for r in failed_results],
            "recommendations": self._generate_recommendations(failed_results)
        }
    
    def _generate_recommendations(self, failed_results: List[ValidationResult]) -> List[str]:
        """Generate recommendations based on failed validations."""
        recommendations = []
        
        # Analyze patterns in failures
        critical_failures = [r for r in failed_results if r.severity == ValidationSeverity.CRITICAL]
        if critical_failures:
            recommendations.append("CRITICAL: Address critical validation failures before proceeding")
        
        error_failures = [r for r in failed_results if r.severity == ValidationSeverity.ERROR]
        if error_failures:
            recommendations.append("Fix data quality issues to improve model reliability")
        
        # Specific recommendations based on rule types
        rule_types = {r.rule_name for r in failed_results}
        
        if "schema_validation" in rule_types:
            recommendations.append("Review data schema and ensure all required columns are present")
        
        if "missing_data_validation" in rule_types:
            recommendations.append("Implement missing data handling strategies (imputation, deletion)")
        
        if "outlier_validation" in rule_types:
            recommendations.append("Investigate outliers and consider robust preprocessing techniques")
        
        if "data_type_validation" in rule_types:
            recommendations.append("Check data loading process and type conversion logic")
        
        return recommendations
    
    def get_validation_history(self) -> List[Dict[str, Any]]:
        """Get history of validation runs."""
        return self.validation_history.copy()


class FairnessValidator(DataValidator):
    """
    Specialized validator for fairness-related data quality checks.
    
    Extends DataValidator with fairness-specific validation rules.
    """
    
    def __init__(self, protected_attributes: List[str], target_column: str):
        """
        Initialize fairness validator.
        
        Args:
            protected_attributes: List of protected attribute column names
            target_column: Name of the target column
        """
        super().__init__()
        self.protected_attributes = protected_attributes
        self.target_column = target_column
        
        # Add fairness-specific rules
        self._add_fairness_rules()
    
    def _add_fairness_rules(self):
        """Add fairness-specific validation rules."""
        # Protected attributes presence rule
        self.add_rule(ProtectedAttributeValidationRule(self.protected_attributes))
        
        # Target column validation rule
        self.add_rule(TargetValidationRule(self.target_column))
        
        # Bias detection rule
        self.add_rule(BiasDetectionRule(self.protected_attributes, self.target_column))
        
        # Representation balance rule
        self.add_rule(RepresentationBalanceRule(self.protected_attributes))


class ProtectedAttributeValidationRule(ValidationRule):
    """Validate protected attributes."""
    
    def __init__(self, protected_attributes: List[str]):
        """Initialize protected attribute validation."""
        super().__init__("protected_attribute_validation")
        self.protected_attributes = protected_attributes
    
    def _check(self, data: pd.DataFrame, **kwargs) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Check protected attributes."""
        missing_attributes = [attr for attr in self.protected_attributes if attr not in data.columns]
        
        if missing_attributes:
            return False, f"Missing protected attributes: {missing_attributes}", {
                'missing_attributes': missing_attributes,
                'available_columns': list(data.columns)
            }
        
        # Check for valid values in protected attributes
        attribute_stats = {}
        for attr in self.protected_attributes:
            if attr in data.columns:
                unique_values = data[attr].nunique()
                null_count = data[attr].isnull().sum()
                
                attribute_stats[attr] = {
                    'unique_values': unique_values,
                    'null_count': int(null_count),
                    'sample_values': list(data[attr].dropna().unique()[:5])
                }
                
                # Check for reasonable number of categories
                if unique_values > 20:
                    logger.warning(f"Protected attribute {attr} has {unique_values} unique values - consider grouping")
        
        return True, "Protected attribute validation passed", {
            'attribute_stats': attribute_stats
        }


class TargetValidationRule(ValidationRule):
    """Validate target column."""
    
    def __init__(self, target_column: str):
        """Initialize target validation."""
        super().__init__("target_validation")
        self.target_column = target_column
    
    def _check(self, data: pd.DataFrame, **kwargs) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Check target column."""
        if self.target_column not in data.columns:
            return False, f"Target column '{self.target_column}' not found", {
                'available_columns': list(data.columns)
            }
        
        target_series = data[self.target_column]
        
        # Check for missing values
        null_count = target_series.isnull().sum()
        if null_count > 0:
            return False, f"Target column has {null_count} missing values", {
                'null_count': int(null_count),
                'total_rows': len(data)
            }
        
        # Check value distribution
        value_counts = target_series.value_counts()
        unique_values = len(value_counts)
        
        target_stats = {
            'unique_values': unique_values,
            'value_distribution': value_counts.to_dict(),
            'is_binary': unique_values == 2
        }
        
        # Check for severe class imbalance
        if unique_values == 2:
            min_class_ratio = value_counts.min() / len(data)
            if min_class_ratio < 0.05:  # Less than 5% minority class
                return False, f"Severe class imbalance detected: minority class = {min_class_ratio:.1%}", {
                    'target_stats': target_stats,
                    'min_class_ratio': min_class_ratio
                }
        
        return True, "Target validation passed", {
            'target_stats': target_stats
        }


class BiasDetectionRule(ValidationRule):
    """Detect potential bias in data."""
    
    def __init__(self, protected_attributes: List[str], target_column: str):
        """Initialize bias detection."""
        super().__init__("bias_detection", ValidationSeverity.WARNING)
        self.protected_attributes = protected_attributes
        self.target_column = target_column
    
    def _check(self, data: pd.DataFrame, **kwargs) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Check for potential bias indicators."""
        if self.target_column not in data.columns:
            return True, "Cannot check bias - target column missing", {}
        
        bias_indicators = {}
        overall_target_rate = data[self.target_column].mean()
        
        for attr in self.protected_attributes:
            if attr not in data.columns:
                continue
            
            # Calculate target rates by protected group
            group_stats = data.groupby(attr)[self.target_column].agg(['count', 'mean']).round(4)
            
            # Calculate demographic parity difference
            group_rates = group_stats['mean']
            max_rate = group_rates.max()
            min_rate = group_rates.min()
            dp_difference = max_rate - min_rate
            
            bias_indicators[attr] = {
                'demographic_parity_difference': float(dp_difference),
                'group_target_rates': group_rates.to_dict(),
                'group_counts': group_stats['count'].to_dict(),
                'overall_target_rate': float(overall_target_rate)
            }
        
        # Check for significant bias
        significant_bias = any(
            stats['demographic_parity_difference'] > 0.1
            for stats in bias_indicators.values()
        )
        
        if significant_bias:
            return False, "Potential bias detected in data", {
                'bias_indicators': bias_indicators,
                'threshold': 0.1
            }
        
        return True, "No significant bias detected", {
            'bias_indicators': bias_indicators
        }


class RepresentationBalanceRule(ValidationRule):
    """Check representation balance across protected groups."""
    
    def __init__(self, protected_attributes: List[str], min_group_size: int = 100):
        """Initialize representation balance check."""
        super().__init__("representation_balance", ValidationSeverity.WARNING)
        self.protected_attributes = protected_attributes
        self.min_group_size = min_group_size
    
    def _check(self, data: pd.DataFrame, **kwargs) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Check representation balance."""
        representation_stats = {}
        
        for attr in self.protected_attributes:
            if attr not in data.columns:
                continue
            
            value_counts = data[attr].value_counts()
            total_count = len(data)
            
            # Check for small groups
            small_groups = value_counts[value_counts < self.min_group_size]
            
            representation_stats[attr] = {
                'group_counts': value_counts.to_dict(),
                'group_proportions': (value_counts / total_count).to_dict(),
                'small_groups': small_groups.index.tolist() if len(small_groups) > 0 else [],
                'min_group_size': self.min_group_size
            }
        
        # Check if any groups are too small
        has_small_groups = any(
            len(stats['small_groups']) > 0
            for stats in representation_stats.values()
        )
        
        if has_small_groups:
            return False, "Some protected groups have insufficient representation", {
                'representation_stats': representation_stats
            }
        
        return True, "Representation balance check passed", {
            'representation_stats': representation_stats
        }


# CLI interface
def main():
    """CLI interface for data validation operations."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Data Validator CLI")
    parser.add_argument("data_path", help="Path to data file")
    parser.add_argument("--target", help="Target column name")
    parser.add_argument("--protected", nargs="+", help="Protected attribute column names")
    parser.add_argument("--output", help="Output file for validation report")
    parser.add_argument("--fairness", action="store_true", help="Run fairness validation")
    
    args = parser.parse_args()
    
    # Load data
    try:
        data = pd.read_csv(args.data_path)
        print(f"Loaded data with shape: {data.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Create validator
    if args.fairness and args.protected and args.target:
        validator = FairnessValidator(
            protected_attributes=args.protected,
            target_column=args.target
        )
        print("Using fairness validator")
    else:
        # Create basic validator with common rules
        validator = DataValidator()
        
        # Add basic validation rules
        validator.add_rule(SchemaValidationRule(list(data.columns)))
        validator.add_rule(MissingDataValidationRule(max_missing_ratio=0.1))
        validator.add_rule(OutlierValidationRule())
        
        print("Using basic validator")
    
    # Run validation
    print("Running validation...")
    report = validator.validate(data)
    
    # Print summary
    print(f"\nValidation Summary:")
    print(f"Overall Status: {report['summary']['overall_status']}")
    print(f"Rules Passed: {report['summary']['passed']}/{report['summary']['total']}")
    
    if report['failed_rules']:
        print(f"\nFailed Rules:")
        for result in report['failed_rules']:
            print(f"  - {result['rule_name']}: {result['message']}")
    
    if report['recommendations']:
        print(f"\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  - {rec}")
    
    # Save report if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nValidation report saved to {args.output}")


if __name__ == "__main__":
    main()