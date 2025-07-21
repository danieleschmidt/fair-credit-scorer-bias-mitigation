"""Test input validation and error handling for data loading and pipeline functions."""

import pytest
import pandas as pd

from src.data_loader_preprocessor import load_credit_data, load_credit_dataset
from src.evaluate_fairness import run_pipeline, run_cross_validation, _save_metrics_json, _validate_common_parameters


class TestDataLoaderValidation:
    """Test validation in data_loader_preprocessor.py"""

    def test_load_credit_dataset_path_validation(self):
        """Test path parameter validation."""
        # Test non-string path
        with pytest.raises(TypeError, match="path must be a string"):
            load_credit_dataset(path=123)
        
        # Test empty path
        with pytest.raises(ValueError, match="path cannot be empty"):
            load_credit_dataset(path="")
        
        # Test whitespace-only path
        with pytest.raises(ValueError, match="path cannot be empty"):
            load_credit_dataset(path="   ")

    def test_load_credit_dataset_random_state_validation(self):
        """Test random_state parameter validation."""
        # Test non-integer random_state
        with pytest.raises(TypeError, match="random_state must be an integer"):
            load_credit_dataset(random_state=42.5)
        
        # Test negative random_state
        with pytest.raises(ValueError, match="random_state must be non-negative"):
            load_credit_dataset(random_state=-1)

    def test_load_credit_dataset_invalid_csv(self, tmp_path):
        """Test handling of invalid CSV files."""
        # Create an empty file
        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("")
        
        with pytest.raises((ValueError, RuntimeError), match="(contains no data|EmptyDataError)"):
            load_credit_dataset(str(empty_file))
        
        # Create a file with invalid CSV
        invalid_file = tmp_path / "invalid.csv"
        invalid_file.write_text("invalid,csv\ndata,more")
        
        with pytest.raises((ValueError, RuntimeError), match="missing required 'label' column"):
            load_credit_dataset(str(invalid_file))

    @pytest.mark.skip(reason="Permission tests don't work reliably as root")
    def test_load_credit_dataset_file_permissions(self, tmp_path):
        """Test handling of file permission errors."""
        # Create a read-only directory
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir(mode=0o444)
        readonly_file = readonly_dir / "data.csv"
        
        try:
            with pytest.raises((PermissionError, FileNotFoundError, RuntimeError)):
                load_credit_dataset(str(readonly_file))
        finally:
            # Cleanup - restore permissions
            readonly_dir.chmod(0o755)

    def test_load_credit_data_test_size_validation(self):
        """Test test_size parameter validation."""
        # Test non-numeric test_size
        with pytest.raises(TypeError, match="test_size must be a number"):
            load_credit_data(test_size="0.3")
        
        # Test test_size out of range
        with pytest.raises(ValueError, match="test_size must be between 0.0 and 1.0"):
            load_credit_data(test_size=0.0)
        
        with pytest.raises(ValueError, match="test_size must be between 0.0 and 1.0"):
            load_credit_data(test_size=1.0)
        
        with pytest.raises(ValueError, match="test_size must be between 0.0 and 1.0"):
            load_credit_data(test_size=1.5)

    def test_load_credit_data_small_dataset(self, tmp_path):
        """Test handling of datasets too small for splitting."""
        # Create a tiny dataset
        small_dataset = tmp_path / "small.csv"
        df = pd.DataFrame({"feature_0": [1], "protected": [0], "label": [1]})
        df.to_csv(small_dataset, index=False)
        
        with pytest.raises(ValueError, match="Dataset has only 1 samples"):
            load_credit_data(str(small_dataset))


class TestPipelineValidation:
    """Test validation in evaluate_fairness.py"""

    def test_run_pipeline_method_validation(self):
        """Test method parameter validation."""
        # Test non-string method
        with pytest.raises(TypeError, match="method must be a string"):
            run_pipeline(method=123)
        
        # Test invalid method
        with pytest.raises(ValueError, match="method must be one of"):
            run_pipeline(method="invalid_method")

    def test_run_pipeline_threshold_validation(self):
        """Test threshold parameter validation."""
        # Test non-numeric threshold
        with pytest.raises(TypeError, match="threshold must be a number"):
            run_pipeline(threshold="0.5")
        
        # Test threshold out of range
        with pytest.raises(ValueError, match="threshold must be between 0.0 and 1.0"):
            run_pipeline(threshold=-0.1)
        
        with pytest.raises(ValueError, match="threshold must be between 0.0 and 1.0"):
            run_pipeline(threshold=1.1)

    def test_run_pipeline_output_path_validation(self):
        """Test output_path parameter validation."""
        # Test non-string output_path
        with pytest.raises(TypeError, match="output_path must be a string"):
            run_pipeline(output_path=123)
        
        # Test empty output_path
        with pytest.raises(ValueError, match="output_path cannot be empty"):
            run_pipeline(output_path="")

    def test_run_cross_validation_cv_validation(self):
        """Test cv parameter validation."""
        # Test non-integer cv
        with pytest.raises(TypeError, match="cv must be an integer"):
            run_cross_validation(cv=5.5)
        
        # Test cv too small
        with pytest.raises(ValueError, match="cv must be at least 2"):
            run_cross_validation(cv=1)

    def test_run_cross_validation_method_validation(self):
        """Test method parameter validation in cross-validation."""
        # Test invalid method
        with pytest.raises(ValueError, match="method must be one of"):
            run_cross_validation(method="invalid")

    def test_run_cross_validation_threshold_validation(self):
        """Test threshold parameter validation in cross-validation."""
        # Test invalid threshold
        with pytest.raises(ValueError, match="threshold must be between 0.0 and 1.0"):
            run_cross_validation(threshold=2.0)


class TestFileIOValidation:
    """Test file I/O error handling"""

    def test_save_metrics_json_path_validation(self):
        """Test path validation in _save_metrics_json."""
        results = {"accuracy": 0.8, "overall": {}, "by_group": {}}
        
        # Test non-string path
        with pytest.raises(ValueError, match="path must be a non-empty string"):
            _save_metrics_json(results, 123)
        
        # Test empty path
        with pytest.raises(ValueError, match="path must be a non-empty string"):
            _save_metrics_json(results, "")

    def test_save_metrics_json_results_validation(self):
        """Test results validation in _save_metrics_json."""
        # Test non-dict results
        with pytest.raises(ValueError, match="results must be a dictionary"):
            _save_metrics_json("invalid", "test.json")

    @pytest.mark.skip(reason="Permission tests don't work reliably as root")
    def test_save_metrics_json_permission_error(self, tmp_path):
        """Test handling of permission errors when saving JSON."""
        # Create valid results with pandas objects (like real usage)
        import pandas as pd
        results = {
            "accuracy": 0.8,
            "overall": pd.Series({"precision": 0.8, "recall": 0.75}),
            "by_group": pd.DataFrame({"precision": [0.82, 0.78]})
        }
        
        # Create a read-only directory
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir(mode=0o444)
        readonly_file = readonly_dir / "metrics.json"
        
        try:
            with pytest.raises((PermissionError, OSError, RuntimeError)):
                _save_metrics_json(results, str(readonly_file))
        finally:
            # Cleanup
            readonly_dir.chmod(0o755)

    def test_save_metrics_json_success(self, tmp_path):
        """Test successful JSON saving with directory creation."""
        # Create valid results with pandas objects (like real usage)
        import pandas as pd
        results = {
            "accuracy": 0.85,
            "overall": pd.Series({"precision": 0.8, "recall": 0.75}),
            "by_group": pd.DataFrame({
                "precision": [0.82, 0.78]
            })
        }
        
        # Test saving to nested directory that doesn't exist
        nested_file = tmp_path / "nested" / "dir" / "metrics.json"
        _save_metrics_json(results, str(nested_file))
        
        assert nested_file.exists()
        assert nested_file.parent.exists()


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_valid_threshold_boundaries(self):
        """Test that boundary threshold values are accepted."""
        # These should not raise exceptions
        run_pipeline(threshold=0.0, test_size=0.1)
        run_pipeline(threshold=1.0, test_size=0.1)

    def test_valid_test_size_boundaries(self, tmp_path):
        """Test that boundary test_size values work correctly."""
        # Create a small but valid dataset
        dataset_path = tmp_path / "test.csv"
        df = pd.DataFrame({
            "feature_0": [1, 2, 3, 4],
            "protected": [0, 1, 0, 1],
            "label": [0, 1, 0, 1]
        })
        df.to_csv(dataset_path, index=False)
        
        # Very small test size (but valid)
        X_train, X_test, y_train, y_test = load_credit_data(
            str(dataset_path), test_size=0.26  # Just above minimum for 4 samples
        )
        assert len(X_test) >= 1
        assert len(X_train) >= 1

    def test_minimum_cv_value(self):
        """Test that minimum valid cv value works."""
        # cv=2 should be the minimum acceptable value
        run_cross_validation(cv=2, data_path="data/credit_data.csv")

    def test_comprehensive_parameter_combinations(self):
        """Test various valid parameter combinations."""
        # Test all valid methods
        valid_methods = ["baseline", "reweight", "postprocess", "expgrad"]
        for method in valid_methods:
            run_pipeline(method=method, test_size=0.3, threshold=0.5)
            run_cross_validation(method=method, cv=2, threshold=0.5)


class TestValidationHelperFunction:
    """Test the centralized _validate_common_parameters helper function."""

    def test_valid_method_validation(self):
        """Test that valid methods pass validation."""
        valid_methods = ["baseline", "reweight", "postprocess", "expgrad"]
        for method in valid_methods:
            # Should not raise any exceptions
            _validate_common_parameters(method)

    def test_invalid_method_type(self):
        """Test that non-string methods are rejected."""
        with pytest.raises(TypeError, match="method must be a string"):
            _validate_common_parameters(123)
        
        with pytest.raises(TypeError, match="method must be a string"):
            _validate_common_parameters(None)

    def test_invalid_method_value(self):
        """Test that unsupported method names are rejected."""
        with pytest.raises(ValueError, match="method must be one of"):
            _validate_common_parameters("invalid_method")
        
        with pytest.raises(ValueError, match="method must be one of"):
            _validate_common_parameters("random_forest")

    def test_valid_threshold_validation(self):
        """Test that valid threshold values pass validation."""
        # Should not raise exceptions
        _validate_common_parameters("baseline", threshold=0.0)
        _validate_common_parameters("baseline", threshold=0.5)
        _validate_common_parameters("baseline", threshold=1.0)
        _validate_common_parameters("baseline", threshold=None)

    def test_invalid_threshold_type(self):
        """Test that non-numeric thresholds are rejected."""
        with pytest.raises(TypeError, match="threshold must be a number or None"):
            _validate_common_parameters("baseline", threshold="0.5")
        
        with pytest.raises(TypeError, match="threshold must be a number or None"):
            _validate_common_parameters("baseline", threshold=[0.5])

    def test_invalid_threshold_range(self):
        """Test that out-of-range thresholds are rejected."""
        with pytest.raises(ValueError, match="threshold must be between 0.0 and 1.0"):
            _validate_common_parameters("baseline", threshold=-0.1)
        
        with pytest.raises(ValueError, match="threshold must be between 0.0 and 1.0"):
            _validate_common_parameters("baseline", threshold=1.1)

    def test_valid_output_path_validation(self):
        """Test that valid output paths pass validation."""
        # Should not raise exceptions
        _validate_common_parameters("baseline", output_path=None)
        _validate_common_parameters("baseline", output_path="metrics.json")
        _validate_common_parameters("baseline", output_path="/path/to/file.json")

    def test_invalid_output_path_type(self):
        """Test that non-string output paths are rejected."""
        with pytest.raises(TypeError, match="output_path must be a string or None"):
            _validate_common_parameters("baseline", output_path=123)
        
        with pytest.raises(TypeError, match="output_path must be a string or None"):
            _validate_common_parameters("baseline", output_path=["path.json"])

    def test_invalid_output_path_empty(self):
        """Test that empty output paths are rejected."""
        with pytest.raises(ValueError, match="output_path cannot be empty"):
            _validate_common_parameters("baseline", output_path="")
        
        with pytest.raises(ValueError, match="output_path cannot be empty"):
            _validate_common_parameters("baseline", output_path="   ")

    def test_combined_parameter_validation(self):
        """Test validation with multiple parameters simultaneously."""
        # All valid parameters
        _validate_common_parameters("reweight", threshold=0.7, output_path="results.json")
        
        # Mix of valid and invalid should still fail
        with pytest.raises(ValueError, match="method must be one of"):
            _validate_common_parameters("invalid", threshold=0.5, output_path="valid.json")
        
        with pytest.raises(ValueError, match="threshold must be between"):
            _validate_common_parameters("baseline", threshold=2.0, output_path="valid.json")