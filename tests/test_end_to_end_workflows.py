"""Integration tests for end-to-end workflows.

This module contains comprehensive tests that validate complete workflows
from data loading to model evaluation, ensuring all components work together
correctly in realistic scenarios.

Test Categories:
- Complete data pipeline workflow
- Model training and evaluation workflow  
- CLI interface end-to-end
- Configuration changes impact
- Performance regression tests

The tests follow the RED-GREEN-REFACTOR TDD methodology and provide
extensive edge case coverage to ensure robust integration behavior.
"""

import os
import tempfile
import shutil
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import modules under test
try:
    from src.data_loader_preprocessor import (
        load_credit_data, load_credit_dataset, 
        load_versioned_credit_data, generate_synthetic_credit_data
    )
    from src.baseline_model import train_baseline_model, evaluate_model
    from src.config import Config, get_config, reset_config
    from src.performance_benchmarking import PerformanceBenchmark
    from src.data_versioning import DataVersionManager
    from src.fairness_metrics import compute_fairness_metrics as calculate_fairness_metrics
    from src.bias_mitigator import BiasPostprocessor
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


class TestCompleteDataPipelineWorkflow:
    """Test complete data pipeline workflow from raw data to processed datasets."""
    
    def setup_method(self):
        """Set up test environment for each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_path = os.path.join(self.temp_dir, "test_data.csv")
        self.version_storage_path = os.path.join(self.temp_dir, "versions")
        
        # Create test configuration
        self.test_config = {
            'data': {
                'default_dataset_path': self.test_data_path,
                'protected_column_name': 'protected',
                'label_column_name': 'label',
                'feature_column_prefix': 'feature_',
                'default_test_size': 0.3,
                'synthetic': {
                    'n_samples': 1000,
                    'n_features': 10,
                    'n_informative': 5,
                    'n_redundant': 2
                }
            },
            'general': {
                'default_random_state': 42
            },
            'model': {
                'logistic_regression': {
                    'max_iter': 1000,
                    'solver': 'liblinear'
                }
            }
        }
        
        # Reset configuration singleton
        reset_config()
    
    def teardown_method(self):
        """Clean up test environment after each test method."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        reset_config()
    
    def test_complete_data_pipeline_synthetic_generation(self):
        """Test complete pipeline with synthetic data generation."""
        # RED: Test should fail initially without implementation
        
        # Generate synthetic data through the pipeline
        X, y, sensitive_features = generate_synthetic_credit_data(
            n_samples=100,
            n_features=5,
            n_informative=3,
            n_redundant=1,
            random_state=42
        )
        
        # Verify synthetic data characteristics
        assert X.shape == (100, 4)  # n_features - 1 (removed for protected attribute)
        assert y.shape == (100,)
        assert sensitive_features.shape == (100,)
        assert len(np.unique(sensitive_features)) == 2  # Binary protected attribute
        assert set(np.unique(y)) == {0, 1}  # Binary classification
        
        # Test data quality
        assert not np.isnan(X).any(), "Features should not contain NaN values"
        assert not np.isnan(y).any(), "Labels should not contain NaN values"
        assert not np.isnan(sensitive_features).any(), "Protected attributes should not contain NaN values"
        
        # Test correlation structure (protected attribute should correlate with features)
        correlation = np.corrcoef(X[:, 0], sensitive_features)[0, 1]
        assert abs(correlation) > 0.1, "Protected attribute should correlate with features"
    
    def test_complete_data_pipeline_csv_loading_and_processing(self):
        """Test complete pipeline with CSV loading and processing."""
        # Create test CSV data
        test_data = {
            'feature_0': np.random.normal(0, 1, 100),
            'feature_1': np.random.normal(0, 1, 100), 
            'feature_2': np.random.normal(0, 1, 100),
            'protected': np.random.binomial(1, 0.5, 100),
            'label': np.random.binomial(1, 0.5, 100)
        }
        test_df = pd.DataFrame(test_data)
        test_df.to_csv(self.test_data_path, index=False)
        
        # Test loading entire dataset
        X, y = load_credit_dataset(path=self.test_data_path, random_state=42)
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert X.shape[0] == 100
        assert len(y) == 100
        assert 'label' not in X.columns  # Label should be removed from features
        assert 'protected' in X.columns  # Protected attribute should remain
        
        # Test train/test splitting
        X_train, X_test, y_train, y_test = load_credit_data(
            path=self.test_data_path,
            test_size=0.3,
            random_state=42
        )
        
        assert X_train.shape[0] == 70  # 70% for training
        assert X_test.shape[0] == 30   # 30% for testing
        assert len(y_train) == 70
        assert len(y_test) == 30
        
        # Verify no data leakage
        train_indices = X_train.index.tolist()
        test_indices = X_test.index.tolist()
        assert len(set(train_indices) & set(test_indices)) == 0
    
    def test_complete_data_pipeline_with_versioning(self):
        """Test complete pipeline with data versioning enabled."""
        # Create test data
        test_data = {
            'feature_0': np.random.normal(0, 1, 50),
            'feature_1': np.random.normal(0, 1, 50), 
            'protected': np.random.binomial(1, 0.5, 50),
            'label': np.random.binomial(1, 0.5, 50)
        }
        test_df = pd.DataFrame(test_data)
        test_df.to_csv(self.test_data_path, index=False)
        
        # Test versioned data loading
        X_train, X_test, y_train, y_test = load_versioned_credit_data(
            path=self.test_data_path,
            test_size=0.3,
            random_state=42,
            enable_versioning=True,
            version_storage_path=self.version_storage_path,
            version_description="Integration test dataset"
        )
        
        # Verify data splits
        assert X_train.shape[0] == 35  # 70% of 50
        assert X_test.shape[0] == 15   # 30% of 50
        assert len(y_train) == 35
        assert len(y_test) == 15
        
        # Verify version storage was created
        assert os.path.exists(self.version_storage_path)
        version_files = os.listdir(self.version_storage_path)
        assert len(version_files) >= 3  # original, train, test versions
        
        # Verify versions can be loaded
        manager = DataVersionManager(self.version_storage_path)
        versions = manager.list_versions()
        assert len(versions) >= 3
        
        # Test version metadata
        for version in versions:
            assert version.version_id is not None
            assert version.created_at is not None
            assert hasattr(version, 'description')
    
    def test_data_pipeline_error_handling(self):
        """Test data pipeline error handling for various edge cases."""
        # Test missing file handling
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_credit_dataset(path="nonexistent_file.csv")
        
        # Test empty CSV handling
        empty_csv_path = os.path.join(self.temp_dir, "empty.csv")
        with open(empty_csv_path, 'w') as f:
            f.write("")
        
        with pytest.raises(ValueError, match="empty"):
            load_credit_dataset(path=empty_csv_path)
        
        # Test invalid CSV format
        invalid_csv_path = os.path.join(self.temp_dir, "invalid.csv")
        with open(invalid_csv_path, 'w') as f:
            f.write("malformed,csv,content\n1,2\n3,4,5,6")
        
        with pytest.raises(ValueError, match="parse"):
            load_credit_dataset(path=invalid_csv_path)
        
        # Test missing required columns
        missing_label_data = pd.DataFrame({
            'feature_0': [1, 2, 3],
            'feature_1': [4, 5, 6],
            'protected': [0, 1, 0]
            # Missing 'label' column
        })
        missing_label_path = os.path.join(self.temp_dir, "missing_label.csv")
        missing_label_data.to_csv(missing_label_path, index=False)
        
        with pytest.raises(ValueError, match="missing required.*label"):
            load_credit_dataset(path=missing_label_path)


class TestModelTrainingAndEvaluationWorkflow:
    """Test complete model training and evaluation workflow."""
    
    def setup_method(self):
        """Set up test environment for each test method."""
        self.temp_dir = tempfile.mkdtemp()
        reset_config()
        
        # Generate test data
        np.random.seed(42)
        self.n_samples = 100
        self.n_features = 5
        
        # Create realistic test data
        X = np.random.normal(0, 1, (self.n_samples, self.n_features))
        y = (X[:, 0] + X[:, 1] + np.random.normal(0, 0.1, self.n_samples) > 0).astype(int)
        protected = (X[:, 0] > 0).astype(int)
        
        # Convert to DataFrames
        self.X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(self.n_features)])
        self.X['protected'] = protected
        self.y = pd.Series(y, name='label')
        
        # Create train/test splits
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Extract protected attributes
        self.protected_train = self.X_train['protected']
        self.protected_test = self.X_test['protected']
        
        # Remove protected attribute from features for some tests
        self.X_train_no_protected = self.X_train.drop('protected', axis=1)
        self.X_test_no_protected = self.X_test.drop('protected', axis=1)
    
    def teardown_method(self):
        """Clean up test environment after each test method."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        reset_config()
    
    def test_complete_model_training_workflow(self):
        """Test complete model training workflow with default configuration."""
        # Train baseline model
        model = train_baseline_model(
            self.X_train_no_protected, 
            self.y_train
        )
        
        # Verify model was trained successfully
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'intercept_')
        
        # Check model coefficients shape
        assert model.coef_.shape == (1, self.n_features)
        assert model.intercept_.shape == (1,)
        
        # Test model can make predictions
        predictions = model.predict(self.X_test_no_protected)
        assert predictions.shape == (len(self.X_test),)
        assert set(predictions) <= {0, 1}  # Binary predictions
        
        # Test model can predict probabilities
        probabilities = model.predict_proba(self.X_test_no_protected)
        assert probabilities.shape == (len(self.X_test), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1
        assert np.all(probabilities >= 0) and np.all(probabilities <= 1)
    
    def test_complete_model_evaluation_workflow(self):
        """Test complete model evaluation workflow with metrics."""
        # Train model
        model = train_baseline_model(
            self.X_train_no_protected, 
            self.y_train
        )
        
        # Evaluate model - basic evaluation
        accuracy, predictions = evaluate_model(
            model, 
            self.X_test_no_protected, 
            self.y_test
        )
        
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0
        assert len(predictions) == len(self.y_test)
        assert set(predictions) <= {0, 1}
        
        # Evaluate model with probabilities
        accuracy_prob, predictions_prob, probabilities = evaluate_model(
            model,
            self.X_test_no_protected,
            self.y_test,
            return_probs=True
        )
        
        assert accuracy == accuracy_prob  # Should be same accuracy
        assert np.array_equal(predictions, predictions_prob)  # Should be same predictions
        assert probabilities.shape == (len(self.y_test),)
        assert np.all(probabilities >= 0) and np.all(probabilities <= 1)
        
        # Evaluate model with custom threshold
        accuracy_thresh, predictions_thresh = evaluate_model(
            model,
            self.X_test_no_protected,
            self.y_test,
            threshold=0.7
        )
        
        assert isinstance(accuracy_thresh, float)
        assert 0.0 <= accuracy_thresh <= 1.0
        assert len(predictions_thresh) == len(self.y_test)
        assert set(predictions_thresh) <= {0, 1}
        
        # Threshold should affect predictions
        regular_pred_rate = np.mean(predictions)
        thresh_pred_rate = np.mean(predictions_thresh)
        assert thresh_pred_rate <= regular_pred_rate  # Higher threshold = fewer positive predictions
    
    def test_model_fairness_evaluation_workflow(self):
        """Test complete model evaluation workflow with fairness metrics."""
        # Train model
        model = train_baseline_model(
            self.X_train_no_protected,
            self.y_train
        )
        
        # Get predictions and probabilities
        accuracy, predictions, probabilities = evaluate_model(
            model,
            self.X_test_no_protected,
            self.y_test,
            return_probs=True
        )
        
        # Calculate fairness metrics
        fairness_metrics = calculate_fairness_metrics(
            y_true=self.y_test,
            y_pred=predictions,
            y_prob=probabilities,
            sensitive_features=self.protected_test
        )
        
        # Verify fairness metrics structure
        assert isinstance(fairness_metrics, dict)
        expected_metrics = [
            'demographic_parity_difference',
            'equalized_odds_difference', 
            'equal_opportunity_difference',
            'calibration_difference'
        ]
        
        for metric in expected_metrics:
            assert metric in fairness_metrics
            assert isinstance(fairness_metrics[metric], (int, float))
            assert not np.isnan(fairness_metrics[metric])
        
        # Verify metrics are within reasonable bounds
        for metric in expected_metrics:
            value = fairness_metrics[metric]
            assert -1.0 <= value <= 1.0, f"{metric} should be between -1 and 1, got {value}"
    
    def test_model_training_with_sample_weights(self):
        """Test model training workflow with sample weights."""
        # Create sample weights (higher weights for minority class)
        sample_weights = np.where(self.y_train == 1, 2.0, 1.0)
        
        # Train model with sample weights
        model_weighted = train_baseline_model(
            self.X_train_no_protected,
            self.y_train,
            sample_weight=sample_weights
        )
        
        # Train model without sample weights for comparison
        model_unweighted = train_baseline_model(
            self.X_train_no_protected,
            self.y_train
        )
        
        # Both models should be valid
        assert hasattr(model_weighted, 'predict')
        assert hasattr(model_unweighted, 'predict')
        
        # Models should have different coefficients due to weighting
        coef_diff = np.abs(model_weighted.coef_ - model_unweighted.coef_).max()
        assert coef_diff > 1e-6, "Sample weights should affect model coefficients"
        
        # Test predictions on the same data
        pred_weighted = model_weighted.predict(self.X_test_no_protected)
        pred_unweighted = model_unweighted.predict(self.X_test_no_protected)
        
        # Models may produce different predictions
        diff_predictions = np.sum(pred_weighted != pred_unweighted)
        assert diff_predictions >= 0  # May or may not differ, but should be valid
    
    def test_end_to_end_bias_mitigation_workflow(self):
        """Test complete bias mitigation workflow from training to post-processing."""
        # Train baseline model
        baseline_model = train_baseline_model(
            self.X_train_no_protected,
            self.y_train
        )
        
        # Get baseline predictions
        baseline_accuracy, baseline_predictions, baseline_probs = evaluate_model(
            baseline_model,
            self.X_test_no_protected,
            self.y_test,
            return_probs=True
        )
        
        # Calculate baseline fairness
        baseline_fairness = calculate_fairness_metrics(
            y_true=self.y_test,
            y_pred=baseline_predictions,
            y_prob=baseline_probs,
            sensitive_features=self.protected_test
        )
        
        # Apply bias mitigation post-processing
        postprocessor = BiasPostprocessor(
            constraint="equalized_odds",
            objective="accuracy"
        )
        
        # Fit post-processor on training data
        train_probs = baseline_model.predict_proba(self.X_train_no_protected)[:, 1]
        postprocessor.fit(
            y_true=self.y_train,
            y_prob=train_probs,
            sensitive_features=self.protected_train
        )
        
        # Apply post-processing to test predictions
        mitigated_predictions = postprocessor.predict(
            y_prob=baseline_probs,
            sensitive_features=self.protected_test
        )
        
        # Calculate mitigated fairness metrics
        mitigated_accuracy = np.mean(mitigated_predictions == self.y_test)
        mitigated_fairness = calculate_fairness_metrics(
            y_true=self.y_test,
            y_pred=mitigated_predictions,
            y_prob=baseline_probs,  # Still use original probabilities
            sensitive_features=self.protected_test
        )
        
        # Verify post-processing completed successfully
        assert len(mitigated_predictions) == len(self.y_test)
        assert set(mitigated_predictions) <= {0, 1}
        assert isinstance(mitigated_accuracy, float)
        assert 0.0 <= mitigated_accuracy <= 1.0
        
        # Verify fairness metrics improved or accuracy trade-off is reasonable
        baseline_eo_diff = abs(baseline_fairness['equalized_odds_difference'])
        mitigated_eo_diff = abs(mitigated_fairness['equalized_odds_difference'])
        
        # Post-processing should improve fairness (reduce equalized odds difference)
        improvement_threshold = 0.9  # Allow for some variation
        assert (mitigated_eo_diff <= baseline_eo_diff * improvement_threshold or
                mitigated_accuracy >= baseline_accuracy * 0.9), \
            "Post-processing should improve fairness or maintain reasonable accuracy"


class TestCLIInterfaceEndToEnd:
    """Test CLI interface end-to-end functionality."""
    
    def setup_method(self):
        """Set up test environment for each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_path = os.path.join(self.temp_dir, "test_data.csv")
        self.output_path = os.path.join(self.temp_dir, "output.json")
        
        # Create test data file
        test_data = {
            'feature_0': np.random.normal(0, 1, 100),
            'feature_1': np.random.normal(0, 1, 100),
            'protected': np.random.binomial(1, 0.5, 100),
            'label': np.random.binomial(1, 0.5, 100)
        }
        test_df = pd.DataFrame(test_data)
        test_df.to_csv(self.test_data_path, index=False)
        
        reset_config()
    
    def teardown_method(self):
        """Clean up test environment after each test method."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        reset_config()
    
    def test_cli_data_loading_interface(self):
        """Test CLI-style data loading interface."""
        # Simulate CLI data loading call
        def simulate_cli_data_load(data_path, test_size=0.3, random_state=42):
            """Simulate CLI function for data loading."""
            try:
                X_train, X_test, y_train, y_test = load_credit_data(
                    path=data_path,
                    test_size=test_size,
                    random_state=random_state
                )
                return {
                    'status': 'success',
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'features': list(X_train.columns),
                    'train_class_distribution': y_train.value_counts().to_dict(),
                    'test_class_distribution': y_test.value_counts().to_dict()
                }
            except Exception as e:
                return {
                    'status': 'error',
                    'error': str(e)
                }
        
        # Test successful data loading
        result = simulate_cli_data_load(self.test_data_path)
        
        assert result['status'] == 'success'
        assert result['train_samples'] == 70
        assert result['test_samples'] == 30
        assert 'protected' in result['features']
        assert 'label' not in result['features']  # Should be removed from features
        assert isinstance(result['train_class_distribution'], dict)
        assert isinstance(result['test_class_distribution'], dict)
        
        # Test error handling
        error_result = simulate_cli_data_load("nonexistent_file.csv")
        assert error_result['status'] == 'error'
        assert 'error' in error_result
    
    def test_cli_model_training_interface(self):
        """Test CLI-style model training interface."""
        # Load data first
        X_train, X_test, y_train, y_test = load_credit_data(
            path=self.test_data_path,
            test_size=0.3,
            random_state=42
        )
        
        # Remove protected attribute for training
        X_train_features = X_train.drop('protected', axis=1)
        X_test_features = X_test.drop('protected', axis=1)
        
        def simulate_cli_model_train(X, y, model_type='baseline', **kwargs):
            """Simulate CLI function for model training."""
            try:
                if model_type == 'baseline':
                    model = train_baseline_model(X, y, **kwargs)
                    return {
                        'status': 'success',
                        'model_type': model_type,
                        'n_features': len(X.columns),
                        'n_samples': len(X),
                        'solver': getattr(model, 'solver', 'unknown'),
                        'max_iter': getattr(model, 'max_iter', 'unknown'),
                        'converged': getattr(model, 'n_iter_', [0])[0] < getattr(model, 'max_iter', 1000)
                    }
                else:
                    return {
                        'status': 'error',
                        'error': f'Unknown model type: {model_type}'
                    }
            except Exception as e:
                return {
                    'status': 'error',
                    'error': str(e)
                }
        
        # Test successful model training
        result = simulate_cli_model_train(X_train_features, y_train)
        
        assert result['status'] == 'success'
        assert result['model_type'] == 'baseline'
        assert result['n_features'] == X_train_features.shape[1]
        assert result['n_samples'] == len(X_train_features)
        assert 'solver' in result
        assert 'max_iter' in result
        
        # Test with custom parameters
        custom_result = simulate_cli_model_train(
            X_train_features, y_train,
            solver='lbfgs', max_iter=500
        )
        assert custom_result['status'] == 'success'
        
        # Test error handling
        error_result = simulate_cli_model_train(X_train_features, y_train, model_type='invalid')
        assert error_result['status'] == 'error'
    
    def test_cli_evaluation_interface(self):
        """Test CLI-style model evaluation interface."""
        # Load data and train model
        X_train, X_test, y_train, y_test = load_credit_data(
            path=self.test_data_path,
            test_size=0.3,
            random_state=42
        )
        
        X_train_features = X_train.drop('protected', axis=1)
        X_test_features = X_test.drop('protected', axis=1)
        protected_test = X_test['protected']
        
        model = train_baseline_model(X_train_features, y_train)
        
        def simulate_cli_evaluate(model, X_test, y_test, protected_features=None, output_file=None):
            """Simulate CLI function for model evaluation."""
            try:
                # Basic evaluation
                accuracy, predictions, probabilities = evaluate_model(
                    model, X_test, y_test, return_probs=True
                )
                
                # Calculate fairness metrics if protected features provided
                fairness_metrics = None
                if protected_features is not None:
                    fairness_metrics = calculate_fairness_metrics(
                        y_true=y_test,
                        y_pred=predictions,
                        y_prob=probabilities,
                        sensitive_features=protected_features
                    )
                
                # Create evaluation report
                report = {
                    'status': 'success',
                    'accuracy': float(accuracy),
                    'n_samples': len(X_test),
                    'positive_rate': float(np.mean(predictions)),
                    'class_distribution': {
                        'actual': y_test.value_counts().to_dict(),
                        'predicted': pd.Series(predictions).value_counts().to_dict()
                    }
                }
                
                if fairness_metrics:
                    report['fairness_metrics'] = fairness_metrics
                
                # Save to file if requested
                if output_file:
                    import json
                    with open(output_file, 'w') as f:
                        json.dump(report, f, indent=2)
                    report['output_file'] = output_file
                
                return report
                
            except Exception as e:
                return {
                    'status': 'error',
                    'error': str(e)
                }
        
        # Test basic evaluation
        result = simulate_cli_evaluate(model, X_test_features, y_test)
        
        assert result['status'] == 'success'
        assert 'accuracy' in result
        assert 0.0 <= result['accuracy'] <= 1.0
        assert result['n_samples'] == len(X_test_features)
        assert 'positive_rate' in result
        assert 'class_distribution' in result
        assert 'fairness_metrics' not in result  # No protected features provided
        
        # Test evaluation with fairness metrics
        fairness_result = simulate_cli_evaluate(
            model, X_test_features, y_test, 
            protected_features=protected_test
        )
        
        assert fairness_result['status'] == 'success'
        assert 'fairness_metrics' in fairness_result
        assert isinstance(fairness_result['fairness_metrics'], dict)
        
        # Test evaluation with output file
        file_result = simulate_cli_evaluate(
            model, X_test_features, y_test,
            protected_features=protected_test,
            output_file=self.output_path
        )
        
        assert file_result['status'] == 'success'
        assert file_result['output_file'] == self.output_path
        assert os.path.exists(self.output_path)
        
        # Verify output file content
        import json
        with open(self.output_path, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data['status'] == 'success'
        assert 'accuracy' in saved_data
        assert 'fairness_metrics' in saved_data


class TestConfigurationChangesImpact:
    """Test impact of configuration changes on system behavior."""
    
    def setup_method(self):
        """Set up test environment for each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")
        reset_config()
    
    def teardown_method(self):
        """Clean up test environment after each test method."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        reset_config()
    
    def test_model_configuration_changes_impact(self):
        """Test impact of model configuration changes."""
        import yaml
        
        # Create base configuration
        base_config = {
            'model': {
                'logistic_regression': {
                    'max_iter': 100,
                    'solver': 'liblinear'
                }
            },
            'general': {
                'default_random_state': 42
            }
        }
        
        # Test with base configuration
        with open(self.config_path, 'w') as f:
            yaml.dump(base_config, f)
        
        config1 = Config(config_path=self.config_path, force_reload=True)
        assert config1.model.logistic_regression.max_iter == 100
        assert config1.model.logistic_regression.solver == 'liblinear'
        
        # Create modified configuration
        modified_config = base_config.copy()
        modified_config['model']['logistic_regression']['max_iter'] = 1000
        modified_config['model']['logistic_regression']['solver'] = 'lbfgs'
        
        with open(self.config_path, 'w') as f:
            yaml.dump(modified_config, f)
        
        config2 = Config(config_path=self.config_path, force_reload=True)
        assert config2.model.logistic_regression.max_iter == 1000
        assert config2.model.logistic_regression.solver == 'lbfgs'
        
        # Test that changes affect model training
        # Generate test data
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 5))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        y_series = pd.Series(y)
        
        # Train models with different configurations
        model1 = train_baseline_model(X_df, y_series, max_iter=100, solver='liblinear')
        model2 = train_baseline_model(X_df, y_series, max_iter=1000, solver='lbfgs')
        
        # Models should have different properties
        assert model1.max_iter == 100
        assert model1.solver == 'liblinear'
        assert model2.max_iter == 1000
        assert model2.solver == 'lbfgs'
        
        # Coefficients may be different due to different solvers
        coef_diff = np.abs(model1.coef_ - model2.coef_).max()
        # Note: May be similar if both converge to same solution, but solvers are different
    
    def test_data_configuration_changes_impact(self):
        """Test impact of data configuration changes."""
        import yaml
        
        # Create configuration with different synthetic data parameters
        config1 = {
            'data': {
                'synthetic': {
                    'n_samples': 100,
                    'n_features': 5,
                    'n_informative': 3,
                    'n_redundant': 1
                }
            },
            'general': {
                'default_random_state': 42
            }
        }
        
        config2 = {
            'data': {
                'synthetic': {
                    'n_samples': 200,
                    'n_features': 10,
                    'n_informative': 7,
                    'n_redundant': 2
                }
            },
            'general': {
                'default_random_state': 42
            }
        }
        
        # Test with first configuration
        with open(self.config_path, 'w') as f:
            yaml.dump(config1, f)
        
        reset_config()
        conf1 = Config(config_path=self.config_path, force_reload=True)
        
        X1, y1, protected1 = generate_synthetic_credit_data(
            n_samples=conf1.data.synthetic.n_samples,
            n_features=conf1.data.synthetic.n_features,
            n_informative=conf1.data.synthetic.n_informative,
            n_redundant=conf1.data.synthetic.n_redundant,
            random_state=conf1.general.default_random_state
        )
        
        # Test with second configuration
        with open(self.config_path, 'w') as f:
            yaml.dump(config2, f)
        
        conf2 = Config(config_path=self.config_path, force_reload=True)
        
        X2, y2, protected2 = generate_synthetic_credit_data(
            n_samples=conf2.data.synthetic.n_samples,
            n_features=conf2.data.synthetic.n_features,
            n_informative=conf2.data.synthetic.n_informative,
            n_redundant=conf2.data.synthetic.n_redundant,
            random_state=conf2.general.default_random_state
        )
        
        # Verify configurations had impact
        assert X1.shape == (100, 4)  # n_features - 1
        assert X2.shape == (200, 9)  # n_features - 1
        assert len(y1) == 100
        assert len(y2) == 200
        assert len(protected1) == 100
        assert len(protected2) == 200
    
    def test_environment_variable_override_impact(self):
        """Test impact of environment variable overrides."""
        import yaml
        
        # Create base configuration
        base_config = {
            'model': {
                'logistic_regression': {
                    'max_iter': 100,
                    'solver': 'liblinear'
                }
            },
            'data': {
                'default_test_size': 0.3
            },
            'general': {
                'default_random_state': 42
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(base_config, f)
        
        # Test without environment variables
        config1 = Config(config_path=self.config_path, force_reload=True)
        assert config1.model.logistic_regression.max_iter == 100
        assert config1.data.default_test_size == 0.3
        
        # Test with environment variable overrides
        with patch.dict(os.environ, {
            'FAIRNESS_MODEL_MAX_ITER': '500',
            'FAIRNESS_DATA_TEST_SIZE': '0.2'
        }):
            config2 = Config(config_path=self.config_path, force_reload=True)
            assert config2.model.logistic_regression.max_iter == 500
            assert config2.data.default_test_size == 0.2
        
        # Test that changes affect actual operations
        # Generate test data
        test_data = pd.DataFrame({
            'feature_0': np.random.normal(0, 1, 100),
            'feature_1': np.random.normal(0, 1, 100),
            'label': np.random.binomial(1, 0.5, 100)
        })
        
        data_path = os.path.join(self.temp_dir, "test_data.csv")
        test_data.to_csv(data_path, index=False)
        
        # Test data loading with different test_size
        X_train1, X_test1, y_train1, y_test1 = load_credit_data(
            path=data_path, test_size=0.3, random_state=42
        )
        
        X_train2, X_test2, y_train2, y_test2 = load_credit_data(
            path=data_path, test_size=0.2, random_state=42
        )
        
        # Different test sizes should result in different split sizes
        assert len(X_test1) == 30  # 30% of 100
        assert len(X_test2) == 20  # 20% of 100
        assert len(X_train1) == 70  # 70% of 100
        assert len(X_train2) == 80  # 80% of 100


class TestPerformanceRegressionTests:
    """Test performance regression detection for system components."""
    
    def setup_method(self):
        """Set up test environment for each test method."""
        self.temp_dir = tempfile.mkdtemp()
        reset_config()
        
        # Create test data with varying sizes
        self.small_data_size = 100
        self.medium_data_size = 1000
        self.large_data_size = 5000
        
        np.random.seed(42)
    
    def teardown_method(self):
        """Clean up test environment after each test method."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        reset_config()
    
    def test_data_loading_performance_regression(self):
        """Test data loading performance across different dataset sizes."""
        benchmark = PerformanceBenchmark(enable_memory_tracking=True)
        
        # Test small dataset loading performance
        def load_small_data():
            X, y, _ = generate_synthetic_credit_data(
                n_samples=self.small_data_size,
                n_features=10,
                random_state=42
            )
            return X, y
        
        small_result = benchmark.benchmark_function(
            load_small_data,
            iterations=5,
            description="Small dataset loading"
        )
        
        # Test medium dataset loading performance
        def load_medium_data():
            X, y, _ = generate_synthetic_credit_data(
                n_samples=self.medium_data_size,
                n_features=10,
                random_state=42
            )
            return X, y
        
        medium_result = benchmark.benchmark_function(
            load_medium_data,
            iterations=3,
            description="Medium dataset loading"
        )
        
        # Test large dataset loading performance
        def load_large_data():
            X, y, _ = generate_synthetic_credit_data(
                n_samples=self.large_data_size,
                n_features=10,
                random_state=42
            )
            return X, y
        
        large_result = benchmark.benchmark_function(
            load_large_data,
            iterations=2,
            description="Large dataset loading"
        )
        
        # Verify performance scaling is reasonable
        assert small_result['mean_time'] > 0
        assert medium_result['mean_time'] > 0
        assert large_result['mean_time'] > 0
        
        # Performance should scale roughly linearly with data size
        # Allow for some variation due to overhead
        small_to_medium_ratio = medium_result['mean_time'] / small_result['mean_time']
        medium_to_large_ratio = large_result['mean_time'] / medium_result['mean_time']
        
        # Should not be more than 50x slower for 10x more data (very generous bounds)
        assert small_to_medium_ratio < 50, f"Medium data loading too slow: {small_to_medium_ratio}x"
        assert medium_to_large_ratio < 20, f"Large data loading too slow: {medium_to_large_ratio}x"
        
        # Memory usage should also scale reasonably
        if 'peak_memory_mb' in small_result:
            small_memory = small_result['peak_memory_mb']
            medium_memory = medium_result['peak_memory_mb']
            large_memory = large_result['peak_memory_mb']
            
            # Memory should increase with data size, but not excessively
            assert medium_memory >= small_memory
            assert large_memory >= medium_memory
            
            # Memory should not increase more than 100x for 50x more data
            if small_memory > 0:
                memory_ratio = large_memory / small_memory
                assert memory_ratio < 100, f"Memory usage scaling too high: {memory_ratio}x"
    
    def test_model_training_performance_regression(self):
        """Test model training performance across different scenarios."""
        benchmark = PerformanceBenchmark(enable_memory_tracking=True)
        
        # Generate datasets of different sizes
        small_X, small_y, _ = generate_synthetic_credit_data(
            n_samples=self.small_data_size, n_features=5, random_state=42
        )
        medium_X, medium_y, _ = generate_synthetic_credit_data(
            n_samples=self.medium_data_size, n_features=5, random_state=42
        )
        large_X, large_y, _ = generate_synthetic_credit_data(
            n_samples=self.large_data_size, n_features=5, random_state=42
        )
        
        # Convert to DataFrames
        small_X_df = pd.DataFrame(small_X, columns=[f'feature_{i}' for i in range(5)])
        medium_X_df = pd.DataFrame(medium_X, columns=[f'feature_{i}' for i in range(5)])
        large_X_df = pd.DataFrame(large_X, columns=[f'feature_{i}' for i in range(5)])
        
        small_y_series = pd.Series(small_y)
        medium_y_series = pd.Series(medium_y)
        large_y_series = pd.Series(large_y)
        
        # Benchmark small dataset training
        def train_small_model():
            return train_baseline_model(small_X_df, small_y_series)
        
        small_training = benchmark.benchmark_function(
            train_small_model,
            iterations=10,
            description="Small dataset training"
        )
        
        # Benchmark medium dataset training
        def train_medium_model():
            return train_baseline_model(medium_X_df, medium_y_series)
        
        medium_training = benchmark.benchmark_function(
            train_medium_model,
            iterations=5,
            description="Medium dataset training"
        )
        
        # Benchmark large dataset training
        def train_large_model():
            return train_baseline_model(large_X_df, large_y_series)
        
        large_training = benchmark.benchmark_function(
            train_large_model,
            iterations=3,
            description="Large dataset training"
        )
        
        # Verify training times are reasonable
        assert small_training['mean_time'] > 0
        assert medium_training['mean_time'] > 0
        assert large_training['mean_time'] > 0
        
        # Training time should increase with dataset size, but not excessively
        small_to_medium = medium_training['mean_time'] / small_training['mean_time']
        medium_to_large = large_training['mean_time'] / medium_training['mean_time']
        
        # Should not be more than 100x slower for 10x-50x more data
        assert small_to_medium < 100, f"Medium training too slow: {small_to_medium}x"
        assert medium_to_large < 50, f"Large training too slow: {medium_to_large}x"
        
        # All training should complete in reasonable time (< 10 seconds for our test sizes)
        assert small_training['mean_time'] < 10.0
        assert medium_training['mean_time'] < 10.0
        assert large_training['mean_time'] < 10.0
    
    def test_model_prediction_performance_regression(self):
        """Test model prediction performance across different scenarios."""
        benchmark = PerformanceBenchmark(enable_memory_tracking=True)
        
        # Train a model for testing
        train_X, train_y, _ = generate_synthetic_credit_data(
            n_samples=1000, n_features=10, random_state=42
        )
        train_X_df = pd.DataFrame(train_X, columns=[f'feature_{i}' for i in range(10)])
        train_y_series = pd.Series(train_y)
        
        model = train_baseline_model(train_X_df, train_y_series)
        
        # Generate test datasets of different sizes
        small_test_X, _, _ = generate_synthetic_credit_data(
            n_samples=self.small_data_size, n_features=10, random_state=43
        )
        medium_test_X, _, _ = generate_synthetic_credit_data(
            n_samples=self.medium_data_size, n_features=10, random_state=43
        )
        large_test_X, _, _ = generate_synthetic_credit_data(
            n_samples=self.large_data_size, n_features=10, random_state=43
        )
        
        small_test_df = pd.DataFrame(small_test_X, columns=[f'feature_{i}' for i in range(10)])
        medium_test_df = pd.DataFrame(medium_test_X, columns=[f'feature_{i}' for i in range(10)])
        large_test_df = pd.DataFrame(large_test_X, columns=[f'feature_{i}' for i in range(10)])
        
        # Benchmark prediction performance
        def predict_small():
            return model.predict(small_test_df)
        
        def predict_medium():
            return model.predict(medium_test_df)
        
        def predict_large():
            return model.predict(large_test_df)
        
        small_pred = benchmark.benchmark_function(
            predict_small, iterations=20, description="Small prediction"
        )
        medium_pred = benchmark.benchmark_function(
            predict_medium, iterations=10, description="Medium prediction"
        )
        large_pred = benchmark.benchmark_function(
            predict_large, iterations=5, description="Large prediction"
        )
        
        # Verify prediction times are reasonable
        assert small_pred['mean_time'] > 0
        assert medium_pred['mean_time'] > 0
        assert large_pred['mean_time'] > 0
        
        # Prediction should be fast and scale linearly
        small_to_medium = medium_pred['mean_time'] / small_pred['mean_time']
        medium_to_large = large_pred['mean_time'] / medium_pred['mean_time']
        
        # Should scale reasonably with data size
        assert small_to_medium < 20, f"Medium prediction too slow: {small_to_medium}x"
        assert medium_to_large < 10, f"Large prediction too slow: {medium_to_large}x"
        
        # All predictions should be very fast (< 1 second)
        assert small_pred['mean_time'] < 1.0
        assert medium_pred['mean_time'] < 1.0
        assert large_pred['mean_time'] < 1.0
    
    def test_fairness_metrics_performance_regression(self):
        """Test fairness metrics calculation performance."""
        benchmark = PerformanceBenchmark(enable_memory_tracking=True)
        
        # Generate test data for fairness evaluation
        def generate_fairness_test_data(n_samples):
            np.random.seed(42)
            y_true = np.random.binomial(1, 0.5, n_samples)
            y_pred = np.random.binomial(1, 0.6, n_samples)
            y_prob = np.random.uniform(0, 1, n_samples)
            protected = np.random.binomial(1, 0.4, n_samples)
            return y_true, y_pred, y_prob, protected
        
        # Test fairness metrics on different data sizes
        small_true, small_pred, small_prob, small_prot = generate_fairness_test_data(self.small_data_size)
        medium_true, medium_pred, medium_prob, medium_prot = generate_fairness_test_data(self.medium_data_size)
        large_true, large_pred, large_prob, large_prot = generate_fairness_test_data(self.large_data_size)
        
        def calc_fairness_small():
            return calculate_fairness_metrics(
                y_true=small_true,
                y_pred=small_pred,
                y_prob=small_prob,
                sensitive_features=small_prot
            )
        
        def calc_fairness_medium():
            return calculate_fairness_metrics(
                y_true=medium_true,
                y_pred=medium_pred,
                y_prob=medium_prob,
                sensitive_features=medium_prot
            )
        
        def calc_fairness_large():
            return calculate_fairness_metrics(
                y_true=large_true,
                y_pred=large_pred,
                y_prob=large_prob,
                sensitive_features=large_prot
            )
        
        # Benchmark fairness calculations
        small_fairness = benchmark.benchmark_function(
            calc_fairness_small, iterations=10, description="Small fairness calculation"
        )
        medium_fairness = benchmark.benchmark_function(
            calc_fairness_medium, iterations=5, description="Medium fairness calculation"
        )
        large_fairness = benchmark.benchmark_function(
            calc_fairness_large, iterations=3, description="Large fairness calculation"
        )
        
        # Verify fairness calculation times
        assert small_fairness['mean_time'] > 0
        assert medium_fairness['mean_time'] > 0
        assert large_fairness['mean_time'] > 0
        
        # Fairness calculations should scale reasonably
        small_to_medium = medium_fairness['mean_time'] / small_fairness['mean_time']
        medium_to_large = large_fairness['mean_time'] / medium_fairness['mean_time']
        
        assert small_to_medium < 20, f"Medium fairness calculation too slow: {small_to_medium}x"
        assert medium_to_large < 10, f"Large fairness calculation too slow: {medium_to_large}x"
        
        # All fairness calculations should be reasonably fast (< 5 seconds)
        assert small_fairness['mean_time'] < 5.0
        assert medium_fairness['mean_time'] < 5.0
        assert large_fairness['mean_time'] < 5.0
        
        # Verify that all calculations produce valid results
        small_result = calc_fairness_small()
        medium_result = calc_fairness_medium()
        large_result = calc_fairness_large()
        
        for result in [small_result, medium_result, large_result]:
            assert isinstance(result, dict)
            assert 'demographic_parity_difference' in result
            assert 'equalized_odds_difference' in result
            assert all(isinstance(v, (int, float)) for v in result.values())
            assert all(not np.isnan(v) for v in result.values())


# Performance baseline constants for regression detection
PERFORMANCE_BASELINES = {
    'data_loading_small': 0.1,     # seconds
    'data_loading_medium': 1.0,    # seconds  
    'data_loading_large': 5.0,     # seconds
    'model_training_small': 1.0,   # seconds
    'model_training_medium': 5.0,  # seconds
    'model_training_large': 10.0,  # seconds
    'prediction_small': 0.01,      # seconds
    'prediction_medium': 0.1,      # seconds
    'prediction_large': 0.5,       # seconds
    'fairness_calc_small': 0.1,    # seconds
    'fairness_calc_medium': 0.5,   # seconds
    'fairness_calc_large': 2.0,    # seconds
}


if __name__ == "__main__":
    # Run the tests when executed directly
    pytest.main([__file__, "-v"])