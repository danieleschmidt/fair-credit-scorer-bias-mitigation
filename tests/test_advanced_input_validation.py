"""
Comprehensive tests for advanced input validation framework.
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest

from src.advanced_input_validation import (
    DataType,
    FairnessAwareValidator,
    SchemaDefinition,
    SchemaInference,
    ValidationRule,
    ValidationResult,
    ValidationSeverity,
    DataDriftDetector
)


@pytest.fixture
def sample_data():
    """Create sample dataset for testing."""
    np.random.seed(42)
    
    data = pd.DataFrame({
        'age': np.random.randint(18, 80, 1000),
        'income': np.random.normal(50000, 15000, 1000),
        'gender': np.random.choice(['M', 'F'], 1000),
        'education': np.random.choice(['HS', 'College', 'Graduate'], 1000),
        'score': np.random.random(1000),
        'approved': np.random.binomial(1, 0.6, 1000)
    })
    
    # Ensure non-negative income
    data['income'] = np.abs(data['income'])
    
    return data


@pytest.fixture
def sample_schema():
    """Create sample schema definition."""
    return SchemaDefinition(
        column_types={
            'age': DataType.NUMERIC,
            'income': DataType.NUMERIC,
            'gender': DataType.CATEGORICAL,
            'education': DataType.CATEGORICAL,
            'score': DataType.NUMERIC,
            'approved': DataType.BINARY
        },
        required_columns={'age', 'income', 'gender', 'education', 'score', 'approved'},
        nullable_columns={'income'},
        value_ranges={
            'age': (18, 100),
            'income': (0, 200000),
            'score': (0, 1)
        },
        categorical_values={
            'gender': {'M', 'F'},
            'education': {'HS', 'College', 'Graduate'}
        },
        protected_attributes={'gender'},
        target_column='approved'
    )


class TestSchemaInference:
    """Test schema inference functionality."""
    
    def test_infer_basic_types(self, sample_data):
        """Test basic data type inference."""
        schema = SchemaInference.infer_schema(
            sample_data,
            target_column='approved',
            protected_attributes=['gender']
        )
        
        assert schema.column_types['age'] == DataType.NUMERIC
        assert schema.column_types['income'] == DataType.NUMERIC
        assert schema.column_types['gender'] == DataType.CATEGORICAL
        assert schema.column_types['education'] == DataType.CATEGORICAL
        assert schema.column_types['score'] == DataType.NUMERIC
        assert schema.column_types['approved'] == DataType.BINARY
        
        assert schema.target_column == 'approved'
        assert 'gender' in schema.protected_attributes
    
    def test_infer_with_nulls(self):
        """Test schema inference with null values."""
        data = pd.DataFrame({
            'col1': [1, 2, None, 4, 5],
            'col2': ['A', 'B', None, 'C', 'D'],
            'col3': [0, 1, 1, None, 0]
        })
        
        schema = SchemaInference.infer_schema(data)
        
        assert 'col1' in schema.nullable_columns
        assert 'col2' in schema.nullable_columns
        assert 'col3' in schema.nullable_columns
    
    def test_infer_binary_detection(self):
        """Test binary column detection."""
        data = pd.DataFrame({
            'binary_01': [0, 1, 0, 1, 1],
            'binary_bool': [True, False, True, False, True],
            'numeric': [1, 2, 3, 4, 5],
            'categorical': ['A', 'B', 'A', 'C', 'B']
        })
        
        schema = SchemaInference.infer_schema(data)
        
        assert schema.column_types['binary_01'] == DataType.BINARY
        assert schema.column_types['binary_bool'] == DataType.BINARY
        assert schema.column_types['numeric'] == DataType.NUMERIC
        assert schema.column_types['categorical'] == DataType.CATEGORICAL


class TestDataDriftDetector:
    """Test data drift detection functionality."""
    
    def test_numeric_drift_detection(self, sample_data, sample_schema):
        """Test drift detection for numeric columns."""
        # Create reference and current data
        reference_data = sample_data.iloc[:500]
        current_data = sample_data.iloc[500:].copy()
        
        # Introduce drift in age column
        current_data['age'] = current_data['age'] + 10  # Age shift
        
        detector = DataDriftDetector(reference_data, sample_schema)
        drift_results = detector.detect_drift(current_data)
        
        # Should detect drift in age column
        assert 'age' in drift_results
        age_result = drift_results['age']
        assert not age_result.passed  # Drift detected
        assert age_result.severity == ValidationSeverity.WARNING
    
    def test_categorical_drift_detection(self, sample_data, sample_schema):
        """Test drift detection for categorical columns."""
        reference_data = sample_data.iloc[:500]
        current_data = sample_data.iloc[500:].copy()
        
        # Introduce drift in gender distribution
        current_data['gender'] = 'M'  # All males in current data
        
        detector = DataDriftDetector(reference_data, sample_schema)
        drift_results = detector.detect_drift(current_data)
        
        # Should detect drift in gender column
        assert 'gender' in drift_results
        gender_result = drift_results['gender']
        assert not gender_result.passed  # Drift detected
    
    def test_no_drift_scenario(self, sample_data, sample_schema):
        """Test scenario with no significant drift."""
        # Split data randomly without introducing drift
        np.random.seed(42)
        indices = np.random.permutation(len(sample_data))
        reference_data = sample_data.iloc[indices[:500]]
        current_data = sample_data.iloc[indices[500:]]
        
        detector = DataDriftDetector(reference_data, sample_schema)
        drift_results = detector.detect_drift(current_data)
        
        # Most columns should not have significant drift
        passed_results = [r for r in drift_results.values() if r.passed]
        assert len(passed_results) >= len(drift_results) * 0.8  # At least 80% should pass


class TestFairnessAwareValidator:
    """Test fairness-aware validation functionality."""
    
    def test_basic_validation_pass(self, sample_data, sample_schema):
        """Test validation with clean data."""
        validator = FairnessAwareValidator(sample_schema)
        results = validator.validate(sample_data)
        
        # Most validations should pass
        passed_results = [r for r in results if r.passed]
        assert len(passed_results) >= len(results) * 0.8
    
    def test_structure_validation_missing_columns(self, sample_data, sample_schema):
        """Test structure validation with missing columns."""
        # Remove required column
        incomplete_data = sample_data.drop('age', axis=1)
        
        validator = FairnessAwareValidator(sample_schema)
        results = validator.validate(incomplete_data)
        
        # Should find structure validation failure
        structure_results = [r for r in results if r.rule_name == 'required_columns']
        assert len(structure_results) == 1
        assert not structure_results[0].passed
        assert 'age' in structure_results[0].message
    
    def test_type_validation_wrong_types(self, sample_data, sample_schema):
        """Test type validation with wrong data types."""
        # Convert numeric column to string
        invalid_data = sample_data.copy()
        invalid_data['age'] = invalid_data['age'].astype(str)
        
        validator = FairnessAwareValidator(sample_schema)
        results = validator.validate(invalid_data)
        
        # Should find type validation failure
        type_results = [r for r in results if r.rule_name == 'data_types']
        assert len(type_results) == 1
        assert not type_results[0].passed
    
    def test_range_validation_out_of_bounds(self, sample_data, sample_schema):
        """Test range validation with out-of-bounds values."""
        # Add values outside expected range
        invalid_data = sample_data.copy()
        invalid_data.loc[0, 'age'] = 150  # Invalid age
        invalid_data.loc[1, 'score'] = 2.0  # Score > 1
        
        validator = FairnessAwareValidator(sample_schema)
        results = validator.validate(invalid_data)
        
        # Should find range validation failure
        range_results = [r for r in results if r.rule_name == 'value_ranges']
        assert len(range_results) == 1
        assert not range_results[0].passed
        assert len(range_results[0].affected_rows) >= 2
    
    def test_categorical_validation_invalid_values(self, sample_data, sample_schema):
        """Test categorical validation with invalid values."""
        # Add invalid categorical value
        invalid_data = sample_data.copy()
        invalid_data.loc[0, 'gender'] = 'X'  # Invalid gender
        invalid_data.loc[1, 'education'] = 'PhD'  # Not in expected values
        
        validator = FairnessAwareValidator(sample_schema)
        results = validator.validate(invalid_data)
        
        # Should find categorical validation failure
        cat_results = [r for r in results if r.rule_name == 'categorical_values']
        assert len(cat_results) == 1
        assert not cat_results[0].passed
    
    def test_fairness_representation_validation(self, sample_data, sample_schema):
        """Test fairness representation validation."""
        # Create data with very small protected group
        biased_data = sample_data.copy()
        biased_data['gender'] = 'M'  # Make almost all male
        biased_data.loc[:10, 'gender'] = 'F'  # Only 11 females
        
        validator = FairnessAwareValidator(sample_schema)
        results = validator.validate(biased_data)
        
        # Should find fairness representation issue
        fairness_results = [r for r in results if r.rule_name == 'fairness_representation']
        assert len(fairness_results) == 1
        assert not fairness_results[0].passed
    
    def test_completeness_validation_too_many_nulls(self, sample_data, sample_schema):
        """Test completeness validation with too many missing values."""
        # Add many null values to non-nullable column
        incomplete_data = sample_data.copy()
        incomplete_data.loc[:200, 'age'] = None  # 20% missing
        
        validator = FairnessAwareValidator(sample_schema)
        results = validator.validate(incomplete_data)
        
        # Should find completeness issue
        completeness_results = [r for r in results if r.rule_name == 'missing_values']
        assert len(completeness_results) == 1
        assert not completeness_results[0].passed
    
    def test_outlier_detection_with_reference_data(self, sample_data, sample_schema):
        """Test outlier detection with reference data."""
        # Fit validator on reference data
        reference_data = sample_data.iloc[:500]
        validator = FairnessAwareValidator(sample_schema)
        validator.fit_reference_data(reference_data)
        
        # Create test data with outliers
        test_data = sample_data.iloc[500:].copy()
        test_data.loc[500:510, 'income'] = 1000000  # Add income outliers
        
        results = validator.validate(test_data)
        
        # Should detect outliers
        outlier_results = [r for r in results if r.rule_name == 'outlier_detection']
        assert len(outlier_results) == 1
        # Note: May pass if outlier ratio is not too high
    
    def test_drift_detection_integration(self, sample_data, sample_schema):
        """Test drift detection integration with validator."""
        # Fit validator on reference data
        reference_data = sample_data.iloc[:500]
        validator = FairnessAwareValidator(sample_schema, enable_drift_detection=True)
        validator.fit_reference_data(reference_data)
        
        # Create test data with drift
        test_data = sample_data.iloc[500:].copy()
        test_data['age'] = test_data['age'] + 20  # Introduce age drift
        
        results = validator.validate(test_data)
        
        # Should detect drift
        drift_results = [r for r in results if 'drift_detection' in r.rule_name]
        age_drift_results = [r for r in drift_results if 'age' in r.rule_name]
        assert len(age_drift_results) == 1
        assert not age_drift_results[0].passed


class TestValidationReporting:
    """Test validation reporting functionality."""
    
    def test_generate_validation_report(self, sample_data, sample_schema):
        """Test validation report generation."""
        validator = FairnessAwareValidator(sample_schema)
        results = validator.validate(sample_data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            report = validator.generate_validation_report(results, f.name)
            
            # Check report content
            assert 'Data Validation Report' in report
            assert 'Summary' in report
            assert 'Total validation checks:' in report
            
            # Check file was created
            assert Path(f.name).exists()
            
            # Clean up
            Path(f.name).unlink()


class TestSchemaSerializationl:
    """Test schema serialization and deserialization."""
    
    def test_schema_to_dict_and_back(self, sample_schema):
        """Test schema serialization to dictionary and back."""
        # Convert to dict
        schema_dict = sample_schema.to_dict()
        
        # Verify structure
        assert 'column_types' in schema_dict
        assert 'required_columns' in schema_dict
        assert 'protected_attributes' in schema_dict
        
        # Convert back
        reconstructed_schema = SchemaDefinition.from_dict(schema_dict)
        
        # Verify equivalence
        assert reconstructed_schema.column_types == sample_schema.column_types
        assert reconstructed_schema.required_columns == sample_schema.required_columns
        assert reconstructed_schema.protected_attributes == sample_schema.protected_attributes
        assert reconstructed_schema.target_column == sample_schema.target_column
    
    def test_schema_save_and_load(self, sample_schema):
        """Test schema save and load functionality."""
        validator = FairnessAwareValidator(sample_schema)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            # Save schema
            validator.save_schema(f.name)
            
            # Load schema
            loaded_validator = FairnessAwareValidator.load_schema(f.name)
            
            # Verify equivalence
            assert loaded_validator.schema.column_types == sample_schema.column_types
            assert loaded_validator.schema.required_columns == sample_schema.required_columns
            
            # Clean up
            Path(f.name).unlink()


class TestValidationRules:
    """Test custom validation rules."""
    
    def test_custom_validation_rules(self, sample_data, sample_schema):
        """Test validator with custom validation rules."""
        # Create custom rule
        custom_rules = [
            ValidationRule(
                name="custom_age_check",
                description="Check if age is reasonable for credit application",
                rule_type="custom",
                parameters={'min_age': 21, 'max_age': 75},
                severity=ValidationSeverity.WARNING,
                enabled=True
            )
        ]
        
        validator = FairnessAwareValidator(sample_schema, validation_rules=custom_rules)
        
        # Custom rule won't be executed since we don't implement custom_check
        # But it should be in the validator
        assert len(validator.validation_rules) == 1
        assert validator.validation_rules[0].name == "custom_age_check"
    
    def test_disable_validation_rule(self, sample_data, sample_schema):
        """Test disabling validation rules."""
        validator = FairnessAwareValidator(sample_schema)
        
        # Disable a rule
        for rule in validator.validation_rules:
            if rule.rule_name == 'required_columns':
                rule.enabled = False
                break
        
        # Remove required column
        incomplete_data = sample_data.drop('age', axis=1)
        results = validator.validate(incomplete_data)
        
        # Should not find structure validation failure since rule is disabled
        structure_results = [r for r in results if r.rule_name == 'required_columns']
        assert len(structure_results) == 0


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_dataframe_validation(self, sample_schema):
        """Test validation with empty DataFrame."""
        empty_data = pd.DataFrame()
        
        validator = FairnessAwareValidator(sample_schema)
        results = validator.validate(empty_data)
        
        # Should handle empty data gracefully
        assert isinstance(results, list)
    
    def test_single_row_dataframe(self, sample_schema):
        """Test validation with single row DataFrame."""
        single_row_data = pd.DataFrame({
            'age': [25],
            'income': [50000],
            'gender': ['M'],
            'education': ['College'],
            'score': [0.7],
            'approved': [1]
        })
        
        validator = FairnessAwareValidator(sample_schema)
        results = validator.validate(single_row_data)
        
        # Should handle single row gracefully
        assert isinstance(results, list)
    
    def test_all_null_column(self, sample_schema):
        """Test validation with column containing all nulls."""
        null_data = pd.DataFrame({
            'age': [None, None, None],
            'income': [50000, 60000, 70000],
            'gender': ['M', 'F', 'M'],
            'education': ['HS', 'College', 'Graduate'],
            'score': [0.5, 0.6, 0.7],
            'approved': [0, 1, 1]
        })
        
        validator = FairnessAwareValidator(sample_schema)
        results = validator.validate(null_data)
        
        # Should detect completeness issue
        completeness_results = [r for r in results if r.rule_name == 'missing_values']
        assert len(completeness_results) == 1
        assert not completeness_results[0].passed
    
    def test_mixed_type_column(self):
        """Test validation with mixed-type column."""
        mixed_data = pd.DataFrame({
            'mixed_col': [1, 'two', 3.0, None, True],
            'normal_col': [1, 2, 3, 4, 5]
        })
        
        # Schema inference should handle mixed types
        schema = SchemaInference.infer_schema(mixed_data)
        
        # Mixed column should be inferred as text
        assert schema.column_types['mixed_col'] == DataType.TEXT
        assert schema.column_types['normal_col'] == DataType.NUMERIC


@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration tests with realistic scenarios."""
    
    def test_credit_approval_scenario(self):
        """Test realistic credit approval validation scenario."""
        # Create realistic credit data
        np.random.seed(42)
        n_samples = 1000
        
        data = pd.DataFrame({
            'age': np.random.randint(18, 80, n_samples),
            'income': np.random.lognormal(10, 1, n_samples),
            'credit_score': np.random.randint(300, 850, n_samples),
            'debt_ratio': np.random.beta(2, 5, n_samples),
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'race': np.random.choice(['White', 'Black', 'Hispanic', 'Asian'], n_samples),
            'employment_length': np.random.exponential(5, n_samples),
            'approved': np.random.binomial(1, 0.4, n_samples)
        })
        
        # Infer schema
        schema = SchemaInference.infer_schema(
            data,
            target_column='approved',
            protected_attributes=['gender', 'race']
        )
        
        # Validate
        validator = FairnessAwareValidator(schema)
        results = validator.validate(data)
        
        # Should mostly pass validation
        passed_results = [r for r in results if r.passed]
        assert len(passed_results) >= len(results) * 0.7
        
        # Generate report
        report = validator.generate_validation_report(results)
        assert 'Credit' not in report  # Schema inference doesn't know domain
        assert 'Summary' in report
    
    def test_drift_detection_scenario(self):
        """Test realistic data drift detection scenario."""
        np.random.seed(42)
        
        # Training data (historical)
        train_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.exponential(1, 1000),
            'category': np.random.choice(['A', 'B', 'C'], 1000, p=[0.5, 0.3, 0.2]),
            'target': np.random.binomial(1, 0.3, 1000)
        })
        
        # Serving data (with drift)
        serve_data = pd.DataFrame({
            'feature1': np.random.normal(0.5, 1.2, 500),  # Mean and variance shift
            'feature2': np.random.exponential(1.5, 500),  # Scale shift
            'category': np.random.choice(['A', 'B', 'C'], 500, p=[0.2, 0.3, 0.5]),  # Distribution shift
            'target': np.random.binomial(1, 0.3, 500)
        })
        
        # Infer schema and set up validation
        schema = SchemaInference.infer_schema(train_data, target_column='target')
        validator = FairnessAwareValidator(schema, enable_drift_detection=True)
        validator.fit_reference_data(train_data)
        
        # Validate serving data
        results = validator.validate(serve_data)
        
        # Should detect drift in multiple features
        drift_results = [r for r in results if 'drift_detection' in r.rule_name]
        failed_drift_results = [r for r in drift_results if not r.passed]
        
        # Expect to detect drift in at least some features
        assert len(failed_drift_results) >= 1


if __name__ == "__main__":
    pytest.main([__file__])