"""Tests for data versioning and tracking functionality.

This module provides comprehensive tests for the data versioning system,
ensuring proper tracking of data lineage, versioning, and metadata management.
"""

import pytest
import json
import tempfile
import os
import hashlib
from datetime import datetime
from unittest.mock import patch, MagicMock
import pandas as pd

from src.data_versioning import (
    DataVersion,
    DataVersionManager,
    DataLineage,
    DataMetadata,
    create_data_version,
    track_data_transformation
)


class TestDataVersion:
    """Test cases for DataVersion class."""
    
    def test_data_version_creation(self):
        """Test basic creation and attributes of DataVersion."""
        metadata = DataMetadata(
            rows=1000,
            columns=10,
            size_bytes=50000,
            schema={"col1": "int64", "col2": "float64"},
            quality_score=1.0
        )
        version = DataVersion(
            version_id="v1.0.0",
            data_hash="abc123",
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
            source_path="/data/train.csv",
            metadata=metadata
        )
        
        assert version.version_id == "v1.0.0"
        assert version.data_hash == "abc123"
        assert version.source_path == "/data/train.csv"
        assert version.metadata.rows == 1000
        assert version.metadata.columns == 10
    
    def test_data_version_to_dict(self):
        """Test conversion to dictionary for serialization."""
        metadata = DataMetadata(
            rows=1000,
            columns=5,
            size_bytes=5000,
            schema={"col1": "int64"},
            quality_score=1.0
        )
        version = DataVersion(
            version_id="v1.0.0",
            data_hash="abc123",
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
            source_path="/data/train.csv",
            metadata=metadata
        )
        
        result_dict = version.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict["version_id"] == "v1.0.0"
        assert result_dict["data_hash"] == "abc123"
        assert result_dict["source_path"] == "/data/train.csv"
        assert "timestamp" in result_dict
        assert result_dict["metadata"]["rows"] == 1000
    
    def test_data_version_from_dict(self):
        """Test creation from dictionary."""
        data_dict = {
            "version_id": "v1.0.0",
            "data_hash": "abc123",
            "timestamp": "2025-01-01T12:00:00",
            "source_path": "/data/train.csv",
            "metadata": {
                "rows": 1000,
                "columns": 5,
                "size_bytes": 5000,
                "schema": {"col1": "int64"},
                "quality_score": 1.0,
                "missing_values": None,
                "statistics": None,
                "created_at": None
            }
        }
        
        version = DataVersion.from_dict(data_dict)
        
        assert version.version_id == "v1.0.0"
        assert version.data_hash == "abc123"
        assert version.source_path == "/data/train.csv"
        assert version.metadata.rows == 1000
    
    def test_data_version_equality(self):
        """Test equality comparison between DataVersion objects."""
        metadata1 = DataMetadata(rows=100, columns=5, size_bytes=1000, schema={"col1": "int64"})
        metadata2 = DataMetadata(rows=100, columns=5, size_bytes=1000, schema={"col1": "int64"})
        metadata3 = DataMetadata(rows=200, columns=5, size_bytes=2000, schema={"col1": "int64"})
        
        version1 = DataVersion(
            version_id="v1.0.0",
            data_hash="abc123",
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
            source_path="/data/train.csv",
            metadata=metadata1
        )
        
        version2 = DataVersion(
            version_id="v1.0.0",
            data_hash="abc123",
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
            source_path="/data/train.csv",
            metadata=metadata2
        )
        
        version3 = DataVersion(
            version_id="v1.0.1",
            data_hash="def456",
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
            source_path="/data/train.csv",
            metadata=metadata3
        )
        
        assert version1 == version2
        assert version1 != version3


class TestDataLineage:
    """Test cases for DataLineage class."""
    
    def test_data_lineage_creation(self):
        """Test basic creation of DataLineage."""
        lineage = DataLineage(
            transformation_id="preprocessing_1",
            input_versions=["v1.0.0"],
            output_version="v1.1.0",
            transformation_type="preprocessing",
            parameters={"test_size": 0.3}
        )
        
        assert lineage.transformation_id == "preprocessing_1"
        assert lineage.input_versions == ["v1.0.0"]
        assert lineage.output_version == "v1.1.0"
        assert lineage.transformation_type == "preprocessing"
        assert lineage.parameters["test_size"] == 0.3
    
    def test_data_lineage_with_multiple_inputs(self):
        """Test DataLineage with multiple input versions."""
        lineage = DataLineage(
            transformation_id="merge_1",
            input_versions=["v1.0.0", "v1.0.1"],
            output_version="v1.2.0",
            transformation_type="merge",
            parameters={"merge_key": "id"}
        )
        
        assert len(lineage.input_versions) == 2
        assert "v1.0.0" in lineage.input_versions
        assert "v1.0.1" in lineage.input_versions
    
    def test_data_lineage_to_dict(self):
        """Test DataLineage serialization."""
        lineage = DataLineage(
            transformation_id="preprocessing_1",
            input_versions=["v1.0.0"],
            output_version="v1.1.0",
            transformation_type="preprocessing",
            parameters={"test_size": 0.3}
        )
        
        result_dict = lineage.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict["transformation_id"] == "preprocessing_1"
        assert result_dict["input_versions"] == ["v1.0.0"]
        assert result_dict["output_version"] == "v1.1.0"
        assert "timestamp" in result_dict


class TestDataMetadata:
    """Test cases for DataMetadata class."""
    
    def test_data_metadata_creation(self):
        """Test creation of DataMetadata."""
        metadata = DataMetadata(
            rows=1000,
            columns=10,
            size_bytes=50000,
            schema={"feature1": "float64", "target": "int64"},
            quality_score=0.95
        )
        
        assert metadata.rows == 1000
        assert metadata.columns == 10
        assert metadata.size_bytes == 50000
        assert metadata.quality_score == 0.95
        assert metadata.schema["feature1"] == "float64"
    
    def test_data_metadata_from_dataframe(self):
        """Test creating DataMetadata from pandas DataFrame."""
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [4, 5, 6],
            'target': [0, 1, 0]
        })
        
        metadata = DataMetadata.from_dataframe(df)
        
        assert metadata.rows == 3
        assert metadata.columns == 3
        assert metadata.size_bytes > 0
        assert "feature1" in metadata.schema
        assert "target" in metadata.schema
    
    def test_data_metadata_quality_assessment(self):
        """Test quality score calculation."""
        # DataFrame with missing values
        df = pd.DataFrame({
            'feature1': [1.0, None, 3.0, 4.0],
            'feature2': [1, 2, 3, 4],
            'target': [0, 1, None, 1]
        })
        
        metadata = DataMetadata.from_dataframe(df, include_quality_assessment=True)
        
        # Quality score should reflect missing values
        assert 0.0 <= metadata.quality_score <= 1.0
        assert metadata.quality_score < 1.0  # Should be less than perfect due to missing values


class TestDataVersionManager:
    """Test cases for DataVersionManager class."""
    
    def test_version_manager_initialization(self):
        """Test DataVersionManager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DataVersionManager(storage_path=temp_dir)
            assert manager.storage_path == temp_dir
            assert os.path.exists(temp_dir)
    
    def test_create_version(self):
        """Test creating a new data version."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DataVersionManager(storage_path=temp_dir)
            
            # Create sample data
            df = pd.DataFrame({
                'feature1': [1, 2, 3],
                'target': [0, 1, 0]
            })
            
            version = manager.create_version(
                data=df,
                source_path="/data/test.csv",
                version_id="v1.0.0"
            )
            
            assert version.version_id == "v1.0.0"
            assert version.source_path == "/data/test.csv"
            assert len(version.data_hash) > 0
            assert version.metadata.rows == 3
            assert version.metadata.columns == 2
    
    def test_save_and_load_version(self):
        """Test saving and loading data versions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DataVersionManager(storage_path=temp_dir)
            
            df = pd.DataFrame({
                'feature1': [1, 2, 3],
                'target': [0, 1, 0]
            })
            
            # Create and save version
            version = manager.create_version(data=df, source_path="/data/test.csv")
            manager.save_version(version, df)
            
            # Load version
            loaded_version = manager.load_version(version.version_id)
            
            assert loaded_version.version_id == version.version_id
            assert loaded_version.data_hash == version.data_hash
            assert loaded_version.metadata.rows == version.metadata.rows
    
    def test_list_versions(self):
        """Test listing all available versions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DataVersionManager(storage_path=temp_dir)
            
            # Create multiple versions
            df1 = pd.DataFrame({'a': [1, 2], 'b': [0, 1]})
            df2 = pd.DataFrame({'a': [3, 4], 'b': [1, 0]})
            
            version1 = manager.create_version(data=df1, source_path="/data/v1.csv", version_id="v1.0.0")
            version2 = manager.create_version(data=df2, source_path="/data/v2.csv", version_id="v2.0.0")
            
            manager.save_version(version1, df1)
            manager.save_version(version2, df2)
            
            versions = manager.list_versions()
            
            assert len(versions) == 2
            version_ids = [v.version_id for v in versions]
            assert "v1.0.0" in version_ids
            assert "v2.0.0" in version_ids
    
    def test_track_lineage(self):
        """Test tracking data transformation lineage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DataVersionManager(storage_path=temp_dir)
            
            # Create input version
            df_input = pd.DataFrame({'a': [1, 2, 3], 'b': [0, 1, 0]})
            input_version = manager.create_version(data=df_input, source_path="/data/input.csv", version_id="v1.0.0")
            manager.save_version(input_version, df_input)
            
            # Create output version
            df_output = pd.DataFrame({'a': [1, 3], 'b': [0, 0]})  # Filtered data
            output_version = manager.create_version(data=df_output, source_path="/data/output.csv", version_id="v1.1.0")
            manager.save_version(output_version, df_output)
            
            # Track transformation
            lineage = manager.track_transformation(
                transformation_id="filter_1",
                input_versions=["v1.0.0"],
                output_version="v1.1.0",
                transformation_type="filter",
                parameters={"condition": "b == 0"}
            )
            
            assert lineage.transformation_id == "filter_1"
            assert lineage.input_versions == ["v1.0.0"]
            assert lineage.output_version == "v1.1.0"
            assert lineage.transformation_type == "filter"
    
    def test_get_lineage_history(self):
        """Test retrieving lineage history for a version."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DataVersionManager(storage_path=temp_dir)
            
            # Create chain of transformations
            df1 = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [0, 1, 0, 1]})
            v1 = manager.create_version(data=df1, source_path="/data/original.csv", version_id="v1.0.0")
            manager.save_version(v1, df1)
            
            df2 = pd.DataFrame({'a': [1, 3], 'b': [0, 0]})  # Filter b==0
            v2 = manager.create_version(data=df2, source_path="/data/filtered.csv", version_id="v1.1.0")
            manager.save_version(v2, df2)
            
            df3 = pd.DataFrame({'a': [2, 6], 'b': [0, 0]})  # Transform a = a*2
            v3 = manager.create_version(data=df3, source_path="/data/transformed.csv", version_id="v1.2.0")
            manager.save_version(v3, df3)
            
            # Track transformations
            manager.track_transformation("filter_1", ["v1.0.0"], "v1.1.0", "filter", {"condition": "b==0"})
            manager.track_transformation("transform_1", ["v1.1.0"], "v1.2.0", "transform", {"operation": "a*2"})
            
            # Get lineage
            history = manager.get_lineage_history("v1.2.0")
            
            assert len(history) >= 1  # Should have at least the direct transformation
            assert any(l.output_version == "v1.2.0" for l in history)


class TestDataVersioningFunctions:
    """Test cases for standalone data versioning functions."""
    
    @patch('src.data_versioning.DataVersionManager')
    def test_create_data_version_function(self, mock_manager_class):
        """Test create_data_version convenience function."""
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager
        
        df = pd.DataFrame({'a': [1, 2], 'b': [0, 1]})
        
        create_data_version(
            data=df,
            source_path="/data/test.csv",
            version_id="v1.0.0",
            storage_path="/tmp/versions"
        )
        
        # Verify manager was created and used correctly
        mock_manager_class.assert_called_once_with("/tmp/versions")
        mock_manager.create_version.assert_called_once()
        mock_manager.save_version.assert_called_once()
    
    @patch('src.data_versioning.DataVersionManager')
    def test_track_data_transformation_function(self, mock_manager_class):
        """Test track_data_transformation convenience function."""
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager
        
        track_data_transformation(
            transformation_id="test_transform",
            input_versions=["v1.0.0"],
            output_version="v1.1.0",
            transformation_type="preprocessing",
            parameters={"test_size": 0.3},
            storage_path="/tmp/versions"
        )
        
        # Verify tracking was called
        mock_manager_class.assert_called_once_with("/tmp/versions")
        mock_manager.track_transformation.assert_called_once()


class TestDataVersioningIntegration:
    """Integration tests for data versioning system."""
    
    def test_end_to_end_workflow(self):
        """Test complete data versioning workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DataVersionManager(storage_path=temp_dir)
            
            # Step 1: Create initial dataset version
            original_data = pd.DataFrame({
                'age': [25, 30, 35, 40],
                'income': [50000, 60000, 70000, 80000],
                'approved': [1, 1, 0, 1]
            })
            
            v1 = manager.create_version(
                data=original_data,
                source_path="/data/raw/credit_data.csv",
                version_id="v1.0.0"
            )
            manager.save_version(v1, original_data)
            
            # Step 2: Create preprocessed version
            preprocessed_data = original_data.copy()
            preprocessed_data['age_normalized'] = preprocessed_data['age'] / 100
            preprocessed_data['income_log'] = preprocessed_data['income'].apply(lambda x: x / 1000)
            
            v2 = manager.create_version(
                data=preprocessed_data,
                source_path="/data/processed/credit_data_v2.csv", 
                version_id="v2.0.0"
            )
            manager.save_version(v2, preprocessed_data)
            
            # Track preprocessing transformation
            manager.track_transformation(
                transformation_id="preprocessing_v1",
                input_versions=["v1.0.0"],
                output_version="v2.0.0",
                transformation_type="feature_engineering",
                parameters={
                    "age_normalization": "divide_by_100",
                    "income_transformation": "divide_by_1000"
                }
            )
            
            # Step 3: Create train/test split versions
            from sklearn.model_selection import train_test_split
            
            train_data, test_data = train_test_split(
                preprocessed_data, 
                test_size=0.3, 
                random_state=42
            )
            
            v3_train = manager.create_version(
                data=train_data,
                source_path="/data/splits/train_v3.csv",
                version_id="v3.0.0-train"
            )
            manager.save_version(v3_train, train_data)
            
            v3_test = manager.create_version(
                data=test_data,
                source_path="/data/splits/test_v3.csv",
                version_id="v3.0.0-test"
            )
            manager.save_version(v3_test, test_data)
            
            # Track splitting transformation
            manager.track_transformation(
                transformation_id="train_test_split_v1",
                input_versions=["v2.0.0"],
                output_version="v3.0.0-train",
                transformation_type="data_split",
                parameters={"test_size": 0.3, "random_state": 42, "split_type": "train"}
            )
            
            manager.track_transformation(
                transformation_id="train_test_split_v2",
                input_versions=["v2.0.0"],
                output_version="v3.0.0-test",
                transformation_type="data_split", 
                parameters={"test_size": 0.3, "random_state": 42, "split_type": "test"}
            )
            
            # Verify the complete workflow
            versions = manager.list_versions()
            assert len(versions) == 4
            
            # Check train version lineage
            train_lineage = manager.get_lineage_history("v3.0.0-train")
            assert len(train_lineage) >= 1
            
            # Verify data integrity
            loaded_train = manager.load_version("v3.0.0-train")
            assert loaded_train.metadata.rows == len(train_data)
            assert loaded_train.metadata.columns == len(train_data.columns)


# Fixtures for common test data
@pytest.fixture
def sample_dataframe():
    """Fixture providing sample DataFrame for testing."""
    return pd.DataFrame({
        'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'feature2': [10, 20, 30, 40, 50],
        'protected_attr': [0, 1, 0, 1, 0],
        'target': [0, 1, 0, 1, 1]
    })


@pytest.fixture
def temp_storage_dir():
    """Fixture providing temporary directory for version storage."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir