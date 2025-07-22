"""Tests for versioned data loading integration.

This module tests the integration between data loading and versioning systems,
ensuring proper tracking of data lineage and version creation.
"""

import pytest
import tempfile
import os
import pandas as pd
from unittest.mock import patch, MagicMock

from src.data_loader_preprocessor import load_versioned_credit_data, VERSIONING_AVAILABLE
from src.data_versioning import DataVersionManager


class TestVersionedDataLoading:
    """Test cases for versioned data loading functionality."""
    
    @pytest.mark.skipif(not VERSIONING_AVAILABLE, reason="Data versioning not available")
    def test_load_versioned_credit_data_with_versioning(self):
        """Test loading data with versioning enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a temporary data directory and version storage
            data_dir = os.path.join(temp_dir, "data")
            version_dir = os.path.join(temp_dir, "versions")
            os.makedirs(data_dir, exist_ok=True)
            
            # Create sample data file
            sample_data = pd.DataFrame({
                'feature1': [1, 2, 3, 4, 5, 6],
                'feature2': [10, 20, 30, 40, 50, 60],
                'protected': [0, 1, 0, 1, 0, 1],
                'label': [0, 1, 0, 1, 1, 0]
            })
            data_path = os.path.join(data_dir, "test_data.csv")
            sample_data.to_csv(data_path, index=False)
            
            # Load data with versioning
            X_train, X_test, y_train, y_test = load_versioned_credit_data(
                path=data_path,
                test_size=0.3,
                random_state=42,
                enable_versioning=True,
                version_storage_path=version_dir,
                version_description="Test dataset for versioning"
            )
            
            # Verify data was loaded correctly
            assert len(X_train) > 0
            assert len(X_test) > 0
            assert len(y_train) == len(X_train)
            assert len(y_test) == len(X_test)
            
            # Verify versions were created
            manager = DataVersionManager(version_dir)
            versions = manager.list_versions()
            
            # Should have original, train, and test versions
            assert len(versions) == 3
            
            version_types = set()
            for version in versions:
                if "original" in version.tags:
                    version_types.add("original")
                elif "train" in version.tags:
                    version_types.add("train")
                elif "test" in version.tags:
                    version_types.add("test")
            
            assert version_types == {"original", "train", "test"}
    
    def test_load_versioned_credit_data_without_versioning(self):
        """Test loading data with versioning disabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample data file
            data_dir = os.path.join(temp_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            
            sample_data = pd.DataFrame({
                'feature1': [1, 2, 3, 4],
                'feature2': [10, 20, 30, 40],
                'protected': [0, 1, 0, 1],
                'label': [0, 1, 0, 1]
            })
            data_path = os.path.join(data_dir, "test_data.csv")
            sample_data.to_csv(data_path, index=False)
            
            # Load data without versioning
            X_train, X_test, y_train, y_test = load_versioned_credit_data(
                path=data_path,
                test_size=0.3,
                random_state=42,
                enable_versioning=False
            )
            
            # Verify data was loaded correctly
            assert len(X_train) > 0
            assert len(X_test) > 0
            assert len(y_train) == len(X_train)
            assert len(y_test) == len(X_test)
    
    def test_versioning_unavailable_error(self):
        """Test error when versioning is requested but unavailable."""
        with patch('src.data_loader_preprocessor.VERSIONING_AVAILABLE', False):
            with pytest.raises(ImportError, match="Data versioning is enabled but.*not available"):
                load_versioned_credit_data(
                    enable_versioning=True,
                    path="dummy_path.csv"
                )
    
    @pytest.mark.skipif(not VERSIONING_AVAILABLE, reason="Data versioning not available")
    def test_versioning_failure_graceful_handling(self):
        """Test graceful handling when versioning fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample data file
            data_dir = os.path.join(temp_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            
            sample_data = pd.DataFrame({
                'feature1': [1, 2, 3, 4],
                'feature2': [10, 20, 30, 40],
                'protected': [0, 1, 0, 1],
                'label': [0, 1, 0, 1]
            })
            data_path = os.path.join(data_dir, "test_data.csv")
            sample_data.to_csv(data_path, index=False)
            
            # Mock DataVersionManager to raise an exception
            with patch('src.data_loader_preprocessor.DataVersionManager') as mock_manager:
                mock_manager.side_effect = Exception("Versioning failed")
                
                # Should not raise an exception, just warn and continue
                X_train, X_test, y_train, y_test = load_versioned_credit_data(
                    path=data_path,
                    test_size=0.3,
                    random_state=42,
                    enable_versioning=True
                )
                
                # Data should still be loaded correctly
                assert len(X_train) > 0
                assert len(X_test) > 0
    
    @pytest.mark.skipif(not VERSIONING_AVAILABLE, reason="Data versioning not available")
    def test_version_metadata_accuracy(self):
        """Test that version metadata accurately reflects the data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample data file
            data_dir = os.path.join(temp_dir, "data")
            version_dir = os.path.join(temp_dir, "versions")
            os.makedirs(data_dir, exist_ok=True)
            
            sample_data = pd.DataFrame({
                'feature1': [1, 2, 3, 4, 5, 6, 7, 8],
                'feature2': [10, 20, 30, 40, 50, 60, 70, 80],
                'protected': [0, 1, 0, 1, 0, 1, 0, 1],
                'label': [0, 1, 0, 1, 1, 0, 1, 0]
            })
            data_path = os.path.join(data_dir, "test_data.csv")
            sample_data.to_csv(data_path, index=False)
            
            # Load data with versioning
            X_train, X_test, y_train, y_test = load_versioned_credit_data(
                path=data_path,
                test_size=0.3,
                random_state=42,
                enable_versioning=True,
                version_storage_path=version_dir
            )
            
            # Check version metadata
            manager = DataVersionManager(version_dir)
            versions = manager.list_versions()
            
            original_version = next(v for v in versions if "original" in v.tags)
            train_version = next(v for v in versions if "train" in v.tags)
            test_version = next(v for v in versions if "test" in v.tags)
            
            # Original version should have all data
            assert original_version.metadata.rows == 8
            assert original_version.metadata.columns == 4
            
            # Train and test versions should sum to original
            total_split_rows = train_version.metadata.rows + test_version.metadata.rows
            assert total_split_rows == original_version.metadata.rows
            
            # Both splits should have same number of columns
            assert train_version.metadata.columns == original_version.metadata.columns
            assert test_version.metadata.columns == original_version.metadata.columns
    
    @pytest.mark.skipif(not VERSIONING_AVAILABLE, reason="Data versioning not available")
    def test_lineage_tracking(self):
        """Test that lineage is properly tracked for data transformations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample data file
            data_dir = os.path.join(temp_dir, "data")
            version_dir = os.path.join(temp_dir, "versions")
            os.makedirs(data_dir, exist_ok=True)
            
            sample_data = pd.DataFrame({
                'feature1': [1, 2, 3, 4, 5, 6],
                'feature2': [10, 20, 30, 40, 50, 60],
                'protected': [0, 1, 0, 1, 0, 1],
                'label': [0, 1, 0, 1, 1, 0]
            })
            data_path = os.path.join(data_dir, "test_data.csv")
            sample_data.to_csv(data_path, index=False)
            
            # Load data with versioning
            load_versioned_credit_data(
                path=data_path,
                test_size=0.3,
                random_state=42,
                enable_versioning=True,
                version_storage_path=version_dir
            )
            
            # Check lineage tracking
            manager = DataVersionManager(version_dir)
            versions = manager.list_versions()
            
            train_version = next(v for v in versions if "train" in v.tags)
            test_version = next(v for v in versions if "test" in v.tags)
            
            # Get lineage for train and test versions
            train_lineage = manager.get_lineage_history(train_version.version_id)
            test_lineage = manager.get_lineage_history(test_version.version_id)
            
            # Should have lineage records
            assert len(train_lineage) >= 1
            assert len(test_lineage) >= 1
            
            # Check transformation parameters
            train_transform = next(l for l in train_lineage if l.output_version == train_version.version_id)
            assert train_transform.transformation_type == "train_test_split"
            assert train_transform.parameters["test_size"] == 0.3
            assert train_transform.parameters["random_state"] == 42
            assert train_transform.parameters["split_type"] == "train"
    
    def test_parameter_validation(self):
        """Test parameter validation for versioned data loading."""
        # Test invalid test_size
        with pytest.raises(ValueError, match="test_size must be between 0.0 and 1.0"):
            load_versioned_credit_data(test_size=1.5, enable_versioning=False)
        
        # Test invalid random_state
        with pytest.raises(ValueError, match="random_state must be non-negative"):
            load_versioned_credit_data(random_state=-1, enable_versioning=False)
        
        # Test invalid path type - this will be caught by underlying load_credit_data
        with pytest.raises((TypeError, ValueError), match="path must be a string|Invalid data format"):
            load_versioned_credit_data(path=123, enable_versioning=False)


class TestVersionedDataLoadingIntegration:
    """Integration tests for versioned data loading."""
    
    @pytest.mark.skipif(not VERSIONING_AVAILABLE, reason="Data versioning not available")
    def test_multiple_loads_same_data(self):
        """Test loading the same data multiple times creates appropriate versions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample data file
            data_dir = os.path.join(temp_dir, "data")
            version_dir = os.path.join(temp_dir, "versions")
            os.makedirs(data_dir, exist_ok=True)
            
            sample_data = pd.DataFrame({
                'feature1': [1, 2, 3, 4, 5, 6],
                'feature2': [10, 20, 30, 40, 50, 60],
                'protected': [0, 1, 0, 1, 0, 1],
                'label': [0, 1, 0, 1, 1, 0]
            })
            data_path = os.path.join(data_dir, "test_data.csv")
            sample_data.to_csv(data_path, index=False)
            
            # Load data twice with same parameters
            load_versioned_credit_data(
                path=data_path,
                test_size=0.3,
                random_state=42,
                enable_versioning=True,
                version_storage_path=version_dir
            )
            
            load_versioned_credit_data(
                path=data_path,
                test_size=0.3,
                random_state=42,
                enable_versioning=True,
                version_storage_path=version_dir
            )
            
            # Check that versions were created 
            manager = DataVersionManager(version_dir)
            versions = manager.list_versions()
            
            # Since the same data is loaded twice with the same parameters,
            # the original dataset should be reused (same hash), but new split versions created
            # We should have at least 3 versions, possibly more depending on implementation
            assert len(versions) >= 3
    
    @pytest.mark.skipif(not VERSIONING_AVAILABLE, reason="Data versioning not available")
    def test_different_splits_different_versions(self):
        """Test that different split parameters create different versions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample data file
            data_dir = os.path.join(temp_dir, "data")
            version_dir = os.path.join(temp_dir, "versions")
            os.makedirs(data_dir, exist_ok=True)
            
            sample_data = pd.DataFrame({
                'feature1': list(range(10)),
                'feature2': list(range(10, 20)),
                'protected': [0, 1] * 5,
                'label': [0, 1, 0, 1, 1, 0, 1, 0, 1, 0]
            })
            data_path = os.path.join(data_dir, "test_data.csv")
            sample_data.to_csv(data_path, index=False)
            
            # Load data with different test sizes
            load_versioned_credit_data(
                path=data_path,
                test_size=0.2,
                random_state=42,
                enable_versioning=True,
                version_storage_path=version_dir
            )
            
            load_versioned_credit_data(
                path=data_path,
                test_size=0.3,
                random_state=42,
                enable_versioning=True,
                version_storage_path=version_dir
            )
            
            # Check that different versions were created
            manager = DataVersionManager(version_dir)
            versions = manager.list_versions()
            
            # Should have multiple versions (at least 3, possibly more)
            # The behavior depends on how version IDs are generated and data is deduplicated
            assert len(versions) >= 3
            
            # Check that we have train and test versions
            train_versions = [v for v in versions if "train" in v.tags]
            test_versions = [v for v in versions if "test" in v.tags]
            
            # Should have at least one train and one test version
            assert len(train_versions) >= 1
            assert len(test_versions) >= 1
            
            # The most recent versions should reflect the last split (test_size=0.3)
            if len(train_versions) >= 1:
                latest_train = sorted(train_versions, key=lambda v: v.timestamp)[-1]
                # With test_size=0.3, training should have 7 rows (10 * 0.7)
                assert latest_train.metadata.rows == 7


# Fixtures for common test data
@pytest.fixture
def sample_credit_data():
    """Fixture providing sample credit data for testing."""
    return pd.DataFrame({
        'age': [25, 30, 35, 40, 45, 50],
        'income': [30000, 45000, 60000, 75000, 90000, 105000],
        'credit_score': [650, 700, 750, 800, 750, 700],
        'protected': [0, 1, 0, 1, 0, 1],
        'target': [0, 1, 1, 1, 0, 1]
    })


@pytest.fixture
def temp_data_environment():
    """Fixture providing temporary directory structure for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = os.path.join(temp_dir, "data")
        version_dir = os.path.join(temp_dir, "versions")
        os.makedirs(data_dir, exist_ok=True)
        
        yield {
            "temp_dir": temp_dir,
            "data_dir": data_dir,
            "version_dir": version_dir
        }