"""Tests for improved error handling with specific exception types."""

import pytest
import pandas as pd
from unittest.mock import patch

from src.data_loader_preprocessor import load_credit_dataset, train_test_split_validated
from src.evaluate_fairness import _save_metrics_json


class TestSpecificErrorHandling:
    """Test that specific errors are raised instead of generic RuntimeError."""

    def test_load_credit_dataset_file_not_found_specific_error(self):
        """Test that FileNotFoundError is raised for missing files."""
        # When file doesn't exist, function generates synthetic data, so we test with mocked read_csv
        with patch("os.path.exists", return_value=True), \
             patch("pandas.read_csv", side_effect=FileNotFoundError("No such file")):
            with pytest.raises(FileNotFoundError) as exc_info:
                load_credit_dataset("nonexistent_file.csv")
            # The error is properly propagated, just check it's a FileNotFoundError
            assert "No such file" in str(exc_info.value)

    def test_load_credit_dataset_permission_error_specific(self):
        """Test that PermissionError is properly propagated."""
        with patch("pandas.read_csv", side_effect=PermissionError("Access denied")):
            with pytest.raises(PermissionError):
                load_credit_dataset("test.csv")

    def test_load_credit_dataset_invalid_csv_specific_error(self):
        """Test that pandas parsing errors are handled specifically."""
        with patch("pandas.read_csv", side_effect=pd.errors.EmptyDataError("No data")):
            with pytest.raises(ValueError) as exc_info:
                load_credit_dataset("test.csv")
            assert "contains no data" in str(exc_info.value)

    def test_load_credit_dataset_missing_label_column_specific(self):
        """Test that missing label column raises ValueError."""
        # Mock a DataFrame without 'label' column
        test_df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        with patch("pandas.read_csv", return_value=test_df):
            with pytest.raises(ValueError) as exc_info:
                load_credit_dataset("test.csv")
            assert "missing required 'label' column" in str(exc_info.value)

    def test_save_metrics_json_permission_error_specific(self):
        """Test that PermissionError is raised for write permission issues."""
        test_metrics = {"accuracy": 0.85, "overall": pd.Series(), "by_group": pd.DataFrame()}
        
        with patch("builtins.open", side_effect=PermissionError("Access denied")):
            with pytest.raises(PermissionError) as exc_info:
                _save_metrics_json(test_metrics, "restricted.json")
            assert "Permission denied writing to restricted.json" in str(exc_info.value)

    def test_save_metrics_json_os_error_specific(self):
        """Test that OSError is properly handled for disk/path issues."""
        test_metrics = {"accuracy": 0.85, "overall": pd.Series(), "by_group": pd.DataFrame()}
        
        with patch("builtins.open", side_effect=OSError("Disk full")):
            with pytest.raises(OSError) as exc_info:
                _save_metrics_json(test_metrics, "test.json")
            assert "Could not write to test.json" in str(exc_info.value)


class TestDataValidationErrors:
    """Test specific validation error handling."""

    def test_invalid_test_size_specific_error(self):
        """Test that invalid test_size values raise specific ValueError."""
        X = [[1, 2], [3, 4], [5, 6]]
        y = [0, 1, 0]
        
        with pytest.raises(ValueError) as exc_info:
            train_test_split_validated(X, y, test_size=1.5)
        assert "test_size must be between 0 and 1" in str(exc_info.value)

    def test_empty_dataset_specific_error(self):
        """Test that empty datasets raise specific ValueError."""
        with pytest.raises(ValueError) as exc_info:
            train_test_split_validated([], [], test_size=0.3)
        assert "Dataset cannot be empty" in str(exc_info.value)

    def test_mismatched_xy_lengths_specific_error(self):
        """Test that mismatched X,y lengths raise specific ValueError."""
        X = [[1, 2], [3, 4]]
        y = [0, 1, 0]  # Different length
        
        with pytest.raises(ValueError) as exc_info:
            train_test_split_validated(X, y, test_size=0.3)
        assert "Features and labels must have the same length" in str(exc_info.value)

    def test_invalid_test_size_type_error(self):
        """Test that invalid test_size type raises TypeError."""
        X = [[1, 2], [3, 4]]
        y = [0, 1]
        
        with pytest.raises(TypeError) as exc_info:
            train_test_split_validated(X, y, test_size="invalid")
        assert "test_size must be a number" in str(exc_info.value)