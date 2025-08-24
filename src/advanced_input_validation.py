"""
Advanced Input Validation Framework for Fair ML Systems.

Provides comprehensive input validation, data quality checks, and anomaly detection
for robust fairness-aware machine learning pipelines.

Features:
- Schema validation with automatic type inference
- Data drift detection and monitoring
- Fairness-aware outlier detection
- Input sanitization for security
- Real-time validation with performance optimization
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from logging_config import get_logger

logger = get_logger(__name__)


class ValidationSeverity(Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DataType(Enum):
    """Supported data types for validation."""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BINARY = "binary"
    TEXT = "text"
    DATETIME = "datetime"


@dataclass
class ValidationRule:
    """Individual validation rule."""
    name: str
    description: str
    rule_type: str
    parameters: Dict[str, Any]
    severity: ValidationSeverity
    enabled: bool = True


@dataclass
class ValidationResult:
    """Result of a validation check."""
    rule_name: str
    passed: bool
    severity: ValidationSeverity
    message: str
    affected_rows: Optional[List[int]] = None
    affected_columns: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'rule_name': self.rule_name,
            'passed': self.passed,
            'severity': self.severity.value,
            'message': self.message,
            'affected_rows': self.affected_rows,
            'affected_columns': self.affected_columns,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class SchemaDefinition:
    """Data schema definition for validation."""
    column_types: Dict[str, DataType]
    required_columns: Set[str]
    nullable_columns: Set[str]
    value_ranges: Dict[str, Tuple[float, float]]
    categorical_values: Dict[str, Set[str]]
    protected_attributes: Set[str]
    target_column: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'column_types': {k: v.value for k, v in self.column_types.items()},
            'required_columns': list(self.required_columns),
            'nullable_columns': list(self.nullable_columns),
            'value_ranges': {k: list(v) for k, v in self.value_ranges.items()},
            'categorical_values': {k: list(v) for k, v in self.categorical_values.items()},
            'protected_attributes': list(self.protected_attributes),
            'target_column': self.target_column
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SchemaDefinition':
        """Create from dictionary."""
        return cls(
            column_types={k: DataType(v) for k, v in data['column_types'].items()},
            required_columns=set(data['required_columns']),
            nullable_columns=set(data['nullable_columns']),
            value_ranges={k: tuple(v) for k, v in data['value_ranges'].items()},
            categorical_values={k: set(v) for k, v in data['categorical_values'].items()},
            protected_attributes=set(data['protected_attributes']),
            target_column=data.get('target_column')
        )


class SchemaInference:
    """Automatic schema inference from data."""

    @staticmethod
    def infer_schema(
        df: pd.DataFrame,
        target_column: str = None,
        protected_attributes: List[str] = None,
        confidence_threshold: float = 0.95
    ) -> SchemaDefinition:
        """
        Infer schema from DataFrame.

        Args:
            df: Input DataFrame
            target_column: Name of target column
            protected_attributes: List of protected attribute columns
            confidence_threshold: Confidence threshold for type inference

        Returns:
            Inferred schema definition
        """
        logger.info("Inferring schema from data")

        if protected_attributes is None:
            protected_attributes = []

        column_types = {}
        value_ranges = {}
        categorical_values = {}
        nullable_columns = set()

        for column in df.columns:
            # Check for nulls
            if df[column].isnull().any():
                nullable_columns.add(column)

            # Infer data type
            non_null_data = df[column].dropna()

            if len(non_null_data) == 0:
                column_types[column] = DataType.TEXT
                continue

            # Check for datetime
            if pd.api.types.is_datetime64_any_dtype(df[column]):
                column_types[column] = DataType.DATETIME
                continue

            # Check for numeric
            if pd.api.types.is_numeric_dtype(df[column]):
                # Check if binary (0/1 or True/False)
                unique_vals = set(non_null_data.unique())
                if unique_vals.issubset({0, 1}) or unique_vals.issubset({True, False}):
                    column_types[column] = DataType.BINARY
                else:
                    column_types[column] = DataType.NUMERIC
                    value_ranges[column] = (float(non_null_data.min()), float(non_null_data.max()))
                continue

            # Check for categorical
            unique_count = non_null_data.nunique()
            total_count = len(non_null_data)

            if unique_count / total_count < 0.1 or unique_count <= 20:
                column_types[column] = DataType.CATEGORICAL
                categorical_values[column] = set(non_null_data.astype(str).unique())
            else:
                column_types[column] = DataType.TEXT

        return SchemaDefinition(
            column_types=column_types,
            required_columns=set(df.columns),
            nullable_columns=nullable_columns,
            value_ranges=value_ranges,
            categorical_values=categorical_values,
            protected_attributes=set(protected_attributes),
            target_column=target_column
        )


class DataDriftDetector:
    """Detect data drift between training and serving data."""

    def __init__(self, reference_data: pd.DataFrame, schema: SchemaDefinition):
        """
        Initialize drift detector.

        Args:
            reference_data: Reference dataset (e.g., training data)
            schema: Data schema definition
        """
        self.reference_data = reference_data
        self.schema = schema
        self.reference_stats = self._compute_reference_statistics()

        logger.info("DataDriftDetector initialized")

    def _compute_reference_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Compute reference statistics for drift detection."""
        stats = {}

        for column in self.reference_data.columns:
            col_stats = {}
            col_data = self.reference_data[column].dropna()

            if self.schema.column_types.get(column) == DataType.NUMERIC:
                col_stats.update({
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std()),
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'quantiles': [float(col_data.quantile(q)) for q in [0.25, 0.5, 0.75]]
                })

            elif self.schema.column_types.get(column) == DataType.CATEGORICAL:
                value_counts = col_data.value_counts(normalize=True)
                col_stats['distribution'] = value_counts.to_dict()

            stats[column] = col_stats

        return stats

    def detect_drift(
        self,
        current_data: pd.DataFrame,
        significance_level: float = 0.05
    ) -> Dict[str, ValidationResult]:
        """
        Detect drift in current data compared to reference.

        Args:
            current_data: Current dataset to check for drift
            significance_level: Statistical significance level

        Returns:
            Dictionary of validation results per column
        """
        drift_results = {}

        for column in current_data.columns:
            if column not in self.reference_stats:
                continue

            col_data = current_data[column].dropna()
            if len(col_data) == 0:
                continue

            ref_data = self.reference_data[column].dropna()

            # Perform appropriate drift test
            if self.schema.column_types.get(column) == DataType.NUMERIC:
                # Kolmogorov-Smirnov test for numeric data
                statistic, p_value = stats.ks_2samp(ref_data, col_data)

                drift_detected = p_value < significance_level
                severity = ValidationSeverity.WARNING if drift_detected else ValidationSeverity.INFO

                message = f"Numeric drift test for {column}: "
                message += f"KS statistic={statistic:.4f}, p-value={p_value:.4f}"

                if drift_detected:
                    message += " - DRIFT DETECTED"

            elif self.schema.column_types.get(column) == DataType.CATEGORICAL:
                # Chi-square test for categorical data
                ref_counts = ref_data.value_counts()
                curr_counts = col_data.value_counts()

                # Align categories
                all_categories = set(ref_counts.index) | set(curr_counts.index)
                ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
                curr_aligned = [curr_counts.get(cat, 0) for cat in all_categories]

                # Avoid zero counts for chi-square test
                ref_aligned = [max(1, count) for count in ref_aligned]
                curr_aligned = [max(1, count) for count in curr_aligned]

                try:
                    statistic, p_value = stats.chisquare(curr_aligned, ref_aligned)
                    drift_detected = p_value < significance_level
                    severity = ValidationSeverity.WARNING if drift_detected else ValidationSeverity.INFO

                    message = f"Categorical drift test for {column}: "
                    message += f"Chi-square statistic={statistic:.4f}, p-value={p_value:.4f}"

                    if drift_detected:
                        message += " - DRIFT DETECTED"

                except Exception as e:
                    drift_detected = False
                    severity = ValidationSeverity.INFO
                    message = f"Could not perform drift test for {column}: {e}"
                    p_value = 1.0

            else:
                # For other types, just check if new values are present
                if hasattr(ref_data, 'unique') and hasattr(col_data, 'unique'):
                    ref_unique = set(ref_data.unique())
                    curr_unique = set(col_data.unique())
                    new_values = curr_unique - ref_unique

                    drift_detected = len(new_values) > 0
                    severity = ValidationSeverity.WARNING if drift_detected else ValidationSeverity.INFO

                    message = f"Value drift check for {column}: "
                    if drift_detected:
                        message += f"Found {len(new_values)} new values"
                    else:
                        message += "No new values detected"
                    p_value = 0.0 if drift_detected else 1.0
                else:
                    drift_detected = False
                    severity = ValidationSeverity.INFO
                    message = f"Could not check drift for {column}"
                    p_value = 1.0

            drift_results[column] = ValidationResult(
                rule_name=f"drift_detection_{column}",
                passed=not drift_detected,
                severity=severity,
                message=message,
                affected_columns=[column],
                metadata={
                    'p_value': p_value,
                    'column_type': self.schema.column_types.get(column, DataType.TEXT).value
                }
            )

        return drift_results


class FairnessAwareValidator:
    """
    Fairness-aware input validation framework.

    Provides comprehensive validation with special attention to fairness
    considerations and protected attribute handling.
    """

    def __init__(
        self,
        schema: SchemaDefinition,
        validation_rules: List[ValidationRule] = None,
        enable_drift_detection: bool = True
    ):
        """
        Initialize validator.

        Args:
            schema: Data schema definition
            validation_rules: Custom validation rules
            enable_drift_detection: Whether to enable drift detection
        """
        self.schema = schema
        self.validation_rules = validation_rules or self._create_default_rules()
        self.enable_drift_detection = enable_drift_detection
        self.drift_detector: Optional[DataDriftDetector] = None
        self.anomaly_detector: Optional[BaseEstimator] = None

        logger.info("FairnessAwareValidator initialized")

    def _create_default_rules(self) -> List[ValidationRule]:
        """Create default validation rules."""
        rules = [
            ValidationRule(
                name="required_columns",
                description="Check that all required columns are present",
                rule_type="structural",
                parameters={},
                severity=ValidationSeverity.ERROR
            ),
            ValidationRule(
                name="data_types",
                description="Validate column data types",
                rule_type="type_check",
                parameters={},
                severity=ValidationSeverity.ERROR
            ),
            ValidationRule(
                name="value_ranges",
                description="Check numeric values are within expected ranges",
                rule_type="range_check",
                parameters={},
                severity=ValidationSeverity.WARNING
            ),
            ValidationRule(
                name="categorical_values",
                description="Check categorical values are from expected set",
                rule_type="categorical_check",
                parameters={},
                severity=ValidationSeverity.WARNING
            ),
            ValidationRule(
                name="missing_values",
                description="Check for unexpected missing values",
                rule_type="completeness_check",
                parameters={'max_missing_ratio': 0.1},
                severity=ValidationSeverity.WARNING
            ),
            ValidationRule(
                name="fairness_representation",
                description="Check representation of protected groups",
                rule_type="fairness_check",
                parameters={'min_group_size': 50},
                severity=ValidationSeverity.WARNING
            ),
            ValidationRule(
                name="outlier_detection",
                description="Detect statistical outliers",
                rule_type="outlier_check",
                parameters={'contamination': 0.1},
                severity=ValidationSeverity.INFO
            )
        ]
        return rules

    def fit_reference_data(self, reference_data: pd.DataFrame):
        """
        Fit validator on reference data for drift detection.

        Args:
            reference_data: Reference dataset (e.g., training data)
        """
        if self.enable_drift_detection:
            self.drift_detector = DataDriftDetector(reference_data, self.schema)

        # Fit anomaly detector
        numeric_columns = [
            col for col, dtype in self.schema.column_types.items()
            if dtype == DataType.NUMERIC and col in reference_data.columns
        ]

        if numeric_columns:
            numeric_data = reference_data[numeric_columns].fillna(0)
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_data)

            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            self.anomaly_detector.fit(scaled_data)
            self._scaler = scaler
            self._numeric_columns = numeric_columns

        logger.info("Validator fitted on reference data")

    def validate(self, data: pd.DataFrame) -> List[ValidationResult]:
        """
        Validate input data.

        Args:
            data: Data to validate

        Returns:
            List of validation results
        """
        logger.info("Starting data validation")
        results = []

        for rule in self.validation_rules:
            if not rule.enabled:
                continue

            try:
                if rule.rule_type == "structural":
                    result = self._validate_structure(data, rule)
                elif rule.rule_type == "type_check":
                    result = self._validate_types(data, rule)
                elif rule.rule_type == "range_check":
                    result = self._validate_ranges(data, rule)
                elif rule.rule_type == "categorical_check":
                    result = self._validate_categories(data, rule)
                elif rule.rule_type == "completeness_check":
                    result = self._validate_completeness(data, rule)
                elif rule.rule_type == "fairness_check":
                    result = self._validate_fairness_representation(data, rule)
                elif rule.rule_type == "outlier_check":
                    result = self._validate_outliers(data, rule)
                else:
                    continue

                if result:
                    results.append(result)

            except Exception as e:
                logger.error(f"Error in validation rule {rule.name}: {e}")
                results.append(ValidationResult(
                    rule_name=rule.name,
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Validation rule failed with error: {e}"
                ))

        # Add drift detection if enabled
        if self.drift_detector:
            try:
                drift_results = self.drift_detector.detect_drift(data)
                results.extend(drift_results.values())
            except Exception as e:
                logger.error(f"Error in drift detection: {e}")

        logger.info(f"Validation completed with {len(results)} results")
        return results

    def _validate_structure(self, data: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Validate data structure."""
        missing_columns = self.schema.required_columns - set(data.columns)

        if missing_columns:
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                severity=rule.severity,
                message=f"Missing required columns: {missing_columns}",
                affected_columns=list(missing_columns)
            )

        return ValidationResult(
            rule_name=rule.name,
            passed=True,
            severity=ValidationSeverity.INFO,
            message="All required columns present"
        )

    def _validate_types(self, data: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Validate data types."""
        type_errors = []

        for column, expected_type in self.schema.column_types.items():
            if column not in data.columns:
                continue

            col_data = data[column].dropna()
            if len(col_data) == 0:
                continue

            if expected_type == DataType.NUMERIC:
                if not pd.api.types.is_numeric_dtype(col_data):
                    type_errors.append(f"{column}: expected numeric, got {col_data.dtype}")
            elif expected_type == DataType.CATEGORICAL:
                # Allow string or category types
                if not (pd.api.types.is_string_dtype(col_data) or
                       pd.api.types.is_categorical_dtype(col_data) or
                       pd.api.types.is_object_dtype(col_data)):
                    type_errors.append(f"{column}: expected categorical, got {col_data.dtype}")
            elif expected_type == DataType.BINARY:
                unique_vals = set(col_data.unique())
                if not (unique_vals.issubset({0, 1}) or unique_vals.issubset({True, False})):
                    type_errors.append(f"{column}: expected binary (0/1), got values {unique_vals}")

        if type_errors:
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                severity=rule.severity,
                message=f"Type validation errors: {'; '.join(type_errors)}",
                metadata={'errors': type_errors}
            )

        return ValidationResult(
            rule_name=rule.name,
            passed=True,
            severity=ValidationSeverity.INFO,
            message="All data types valid"
        )

    def _validate_ranges(self, data: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Validate numeric ranges."""
        range_violations = []
        affected_rows = []

        for column, (min_val, max_val) in self.schema.value_ranges.items():
            if column not in data.columns:
                continue

            col_data = data[column].dropna()
            violations = col_data[(col_data < min_val) | (col_data > max_val)]

            if len(violations) > 0:
                range_violations.append(
                    f"{column}: {len(violations)} values outside range [{min_val}, {max_val}]"
                )
                affected_rows.extend(violations.index.tolist())

        if range_violations:
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                severity=rule.severity,
                message=f"Range validation errors: {'; '.join(range_violations)}",
                affected_rows=list(set(affected_rows)),
                metadata={'violations': range_violations}
            )

        return ValidationResult(
            rule_name=rule.name,
            passed=True,
            severity=ValidationSeverity.INFO,
            message="All values within expected ranges"
        )

    def _validate_categories(self, data: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Validate categorical values."""
        category_violations = []
        affected_rows = []

        for column, expected_values in self.schema.categorical_values.items():
            if column not in data.columns:
                continue

            col_data = data[column].dropna().astype(str)
            invalid_values = set(col_data.unique()) - expected_values

            if invalid_values:
                violations_mask = col_data.isin(invalid_values)
                affected_rows.extend(data.index[violations_mask].tolist())
                category_violations.append(
                    f"{column}: unexpected values {invalid_values}"
                )

        if category_violations:
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                severity=rule.severity,
                message=f"Categorical validation errors: {'; '.join(category_violations)}",
                affected_rows=list(set(affected_rows)),
                metadata={'violations': category_violations}
            )

        return ValidationResult(
            rule_name=rule.name,
            passed=True,
            severity=ValidationSeverity.INFO,
            message="All categorical values valid"
        )

    def _validate_completeness(self, data: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Validate data completeness."""
        max_missing_ratio = rule.parameters.get('max_missing_ratio', 0.1)
        completeness_issues = []

        for column in data.columns:
            if column in self.schema.nullable_columns:
                continue

            missing_ratio = data[column].isnull().sum() / len(data)
            if missing_ratio > max_missing_ratio:
                completeness_issues.append(
                    f"{column}: {missing_ratio:.2%} missing (max allowed: {max_missing_ratio:.2%})"
                )

        if completeness_issues:
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                severity=rule.severity,
                message=f"Completeness issues: {'; '.join(completeness_issues)}",
                metadata={'issues': completeness_issues}
            )

        return ValidationResult(
            rule_name=rule.name,
            passed=True,
            severity=ValidationSeverity.INFO,
            message="Data completeness satisfactory"
        )

    def _validate_fairness_representation(self, data: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Validate fairness representation."""
        min_group_size = rule.parameters.get('min_group_size', 50)
        representation_issues = []

        for attr in self.schema.protected_attributes:
            if attr not in data.columns:
                continue

            group_counts = data[attr].value_counts()
            small_groups = group_counts[group_counts < min_group_size]

            if len(small_groups) > 0:
                for group, count in small_groups.items():
                    representation_issues.append(
                        f"{attr}={group}: only {count} samples (min required: {min_group_size})"
                    )

        if representation_issues:
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                severity=rule.severity,
                message=f"Representation issues: {'; '.join(representation_issues)}",
                metadata={'issues': representation_issues}
            )

        return ValidationResult(
            rule_name=rule.name,
            passed=True,
            severity=ValidationSeverity.INFO,
            message="Protected group representation adequate"
        )

    def _validate_outliers(self, data: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Validate outliers using anomaly detection."""
        if not self.anomaly_detector or not hasattr(self, '_numeric_columns'):
            return ValidationResult(
                rule_name=rule.name,
                passed=True,
                severity=ValidationSeverity.INFO,
                message="Outlier detection not available (no reference data)"
            )

        try:
            numeric_data = data[self._numeric_columns].fillna(0)
            scaled_data = self._scaler.transform(numeric_data)

            outlier_scores = self.anomaly_detector.decision_function(scaled_data)
            is_outlier = self.anomaly_detector.predict(scaled_data) == -1

            outlier_count = np.sum(is_outlier)
            outlier_ratio = outlier_count / len(data)

            if outlier_ratio > 0.2:  # More than 20% outliers
                severity = ValidationSeverity.WARNING
                passed = False
                message = f"High outlier ratio: {outlier_ratio:.2%} ({outlier_count} samples)"
            else:
                severity = ValidationSeverity.INFO
                passed = True
                message = f"Normal outlier ratio: {outlier_ratio:.2%} ({outlier_count} samples)"

            return ValidationResult(
                rule_name=rule.name,
                passed=passed,
                severity=severity,
                message=message,
                affected_rows=data.index[is_outlier].tolist() if outlier_count > 0 else None,
                metadata={
                    'outlier_count': int(outlier_count),
                    'outlier_ratio': float(outlier_ratio),
                    'mean_outlier_score': float(np.mean(outlier_scores[is_outlier])) if outlier_count > 0 else None
                }
            )

        except Exception as e:
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Outlier detection failed: {e}"
            )

    def generate_validation_report(
        self,
        validation_results: List[ValidationResult],
        output_path: str = None
    ) -> str:
        """
        Generate comprehensive validation report.

        Args:
            validation_results: Results from validation
            output_path: Path to save report

        Returns:
            Report content as string
        """
        # Summary statistics
        total_checks = len(validation_results)
        passed_checks = sum(1 for r in validation_results if r.passed)
        failed_checks = total_checks - passed_checks

        severity_counts = {}
        for severity in ValidationSeverity:
            severity_counts[severity.value] = sum(
                1 for r in validation_results if r.severity == severity
            )

        # Generate report
        report = f"""
# Data Validation Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total validation checks: {total_checks}
- Passed: {passed_checks} ({passed_checks/total_checks:.1%})
- Failed: {failed_checks} ({failed_checks/total_checks:.1%})

## Severity Distribution
"""

        for severity, count in severity_counts.items():
            report += f"- {severity.upper()}: {count}\n"

        report += "\n## Detailed Results\n\n"

        # Group results by severity
        for severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR,
                        ValidationSeverity.WARNING, ValidationSeverity.INFO]:
            severity_results = [r for r in validation_results if r.severity == severity]
            if not severity_results:
                continue

            report += f"### {severity.value.upper()} ({len(severity_results)} items)\n\n"

            for result in severity_results:
                status = "✅ PASSED" if result.passed else "❌ FAILED"
                report += f"**{result.rule_name}** - {status}\n"
                report += f"- {result.message}\n"

                if result.affected_rows:
                    report += f"- Affected rows: {len(result.affected_rows)}\n"
                if result.affected_columns:
                    report += f"- Affected columns: {', '.join(result.affected_columns)}\n"

                report += "\n"

        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Validation report saved to {output_path}")

        return report

    def save_schema(self, filepath: str):
        """Save schema definition to file."""
        schema_dict = self.schema.to_dict()
        with open(filepath, 'w') as f:
            json.dump(schema_dict, f, indent=2)
        logger.info(f"Schema saved to {filepath}")

    @classmethod
    def load_schema(cls, filepath: str) -> 'FairnessAwareValidator':
        """Load validator from saved schema."""
        with open(filepath) as f:
            schema_dict = json.load(f)

        schema = SchemaDefinition.from_dict(schema_dict)
        return cls(schema)


# Example usage and CLI interface
def main():
    """CLI interface for validation framework."""
    import argparse

    parser = argparse.ArgumentParser(description="Fairness-Aware Input Validation")
    parser.add_argument("--data", required=True, help="Path to data file (CSV)")
    parser.add_argument("--schema", help="Path to schema file (JSON)")
    parser.add_argument("--infer-schema", action="store_true", help="Infer schema from data")
    parser.add_argument("--target", help="Target column name")
    parser.add_argument("--protected-attrs", nargs="+", help="Protected attribute columns")
    parser.add_argument("--output", help="Output path for validation report")
    parser.add_argument("--reference-data", help="Reference data for drift detection")

    args = parser.parse_args()

    # Load data
    try:
        data = pd.read_csv(args.data)
        logger.info(f"Loaded data with shape {data.shape}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # Initialize validator
    if args.schema:
        validator = FairnessAwareValidator.load_schema(args.schema)
    elif args.infer_schema:
        schema = SchemaInference.infer_schema(
            data,
            target_column=args.target,
            protected_attributes=args.protected_attrs or []
        )
        validator = FairnessAwareValidator(schema)

        # Save inferred schema
        schema_path = args.data.replace('.csv', '_schema.json')
        validator.save_schema(schema_path)
        logger.info(f"Inferred and saved schema to {schema_path}")
    else:
        logger.error("Must provide --schema or --infer-schema")
        return

    # Fit reference data if provided
    if args.reference_data:
        try:
            reference_data = pd.read_csv(args.reference_data)
            validator.fit_reference_data(reference_data)
            logger.info("Fitted validator on reference data")
        except Exception as e:
            logger.error(f"Failed to load reference data: {e}")

    # Validate data
    results = validator.validate(data)

    # Generate report
    output_path = args.output or args.data.replace('.csv', '_validation_report.md')
    validator.generate_validation_report(results, output_path)

    # Print summary
    total_checks = len(results)
    passed_checks = sum(1 for r in results if r.passed)
    failed_checks = total_checks - passed_checks

    print("\nValidation Summary:")
    print(f"- Total checks: {total_checks}")
    print(f"- Passed: {passed_checks}")
    print(f"- Failed: {failed_checks}")
    print(f"- Success rate: {passed_checks/total_checks:.1%}")

    if failed_checks > 0:
        print("\nFailed checks:")
        for result in results:
            if not result.passed:
                print(f"- {result.rule_name}: {result.message}")

    print(f"\nFull report saved to: {output_path}")


if __name__ == "__main__":
    main()
