"""Data versioning and tracking system for reproducible ML workflows.

This module provides comprehensive data versioning capabilities including:
- Data version management with metadata tracking
- Data lineage tracking for transformation history
- Hash-based data integrity verification
- Serialization and storage of versioned datasets
- Quality assessment and monitoring

Key Features:
- Automatic version generation with content-based hashing
- Complete transformation lineage tracking
- Metadata extraction from DataFrames
- JSON-based storage with data integrity checks
- Query capabilities for version history and lineage

Classes:
    DataVersion: Represents a versioned dataset with metadata
    DataLineage: Tracks transformation relationships between versions
    DataMetadata: Stores dataset characteristics and quality metrics
    DataVersionManager: Main interface for version management operations

Functions:
    create_data_version: Convenience function for creating versions
    track_data_transformation: Convenience function for lineage tracking

Usage:
    >>> from data_versioning import DataVersionManager, create_data_version
    >>> 
    >>> # Create version manager
    >>> manager = DataVersionManager("./data_versions")
    >>> 
    >>> # Version a dataset
    >>> version = manager.create_version(df, source_path="data.csv")
    >>> manager.save_version(version, df)
    >>> 
    >>> # Track transformation
    >>> manager.track_transformation(
    ...     "preprocessing_1", ["v1.0.0"], "v1.1.0", "preprocessing",
    ...     {"test_size": 0.3}
    ... )
"""

import os
import json
import hashlib
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DataMetadata:
    """Metadata information for a versioned dataset.
    
    Attributes:
        rows: Number of rows in the dataset
        columns: Number of columns in the dataset  
        size_bytes: Size of the dataset in bytes
        schema: Column names and data types mapping
        quality_score: Data quality assessment score (0-1)
        missing_values: Count of missing values per column
        statistics: Basic statistical summaries
        created_at: Timestamp when metadata was created
    """
    rows: int
    columns: int
    size_bytes: int
    schema: Dict[str, str]
    quality_score: float = 1.0
    missing_values: Optional[Dict[str, int]] = None
    statistics: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize computed fields after object creation."""
        if self.created_at is None:
            self.created_at = datetime.now()
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, include_quality_assessment: bool = True,
                      include_statistics: bool = True) -> "DataMetadata":
        """Create DataMetadata from a pandas DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to analyze
        include_quality_assessment : bool, optional
            Whether to compute quality score, by default True
        include_statistics : bool, optional
            Whether to include statistical summaries, by default True
            
        Returns
        -------
        DataMetadata
            Metadata object with computed information
        """
        # Basic information
        rows, columns = df.shape
        size_bytes = df.memory_usage(deep=True).sum()
        
        # Schema information
        schema = {col: str(df[col].dtype) for col in df.columns}
        
        # Missing values analysis
        missing_values = df.isnull().sum().to_dict()
        
        # Quality assessment
        quality_score = 1.0
        if include_quality_assessment:
            total_cells = rows * columns
            missing_cells = sum(missing_values.values())
            if total_cells > 0:
                quality_score = max(0.0, 1.0 - (missing_cells / total_cells))
        
        # Statistical summaries
        statistics = None
        if include_statistics:
            try:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    stats = df[numeric_cols].describe()
                    statistics = {
                        col: {
                            'mean': float(stats.loc['mean', col]) if not pd.isna(stats.loc['mean', col]) else None,
                            'std': float(stats.loc['std', col]) if not pd.isna(stats.loc['std', col]) else None,
                            'min': float(stats.loc['min', col]) if not pd.isna(stats.loc['min', col]) else None,
                            'max': float(stats.loc['max', col]) if not pd.isna(stats.loc['max', col]) else None,
                        }
                        for col in numeric_cols
                    }
            except Exception as e:
                logger.warning(f"Failed to compute statistics: {e}")
                statistics = None
        
        return cls(
            rows=rows,
            columns=columns,
            size_bytes=size_bytes,
            schema=schema,
            quality_score=quality_score,
            missing_values=missing_values,
            statistics=statistics
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        if self.created_at:
            result['created_at'] = self.created_at.isoformat()
        
        # Convert numpy types to native Python types for JSON serialization
        if self.missing_values:
            result['missing_values'] = {k: int(v) for k, v in self.missing_values.items()}
        if self.statistics:
            result['statistics'] = {
                col: {k: float(v) if v is not None else None for k, v in stats.items()}
                for col, stats in self.statistics.items()
            }
        
        # Convert all numeric fields to native Python types
        result['rows'] = int(result['rows'])
        result['columns'] = int(result['columns'])
        result['size_bytes'] = int(result['size_bytes'])
        result['quality_score'] = float(result['quality_score'])
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataMetadata":
        """Create DataMetadata from dictionary."""
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


@dataclass
class DataVersion:
    """Represents a versioned dataset with complete metadata.
    
    Attributes:
        version_id: Unique identifier for this version
        data_hash: Content-based hash for integrity verification
        timestamp: When this version was created
        source_path: Original data source path or identifier
        metadata: Dataset metadata and characteristics
        tags: Optional tags for categorization
        description: Human-readable description of this version
    """
    version_id: str
    data_hash: str
    timestamp: datetime
    source_path: str
    metadata: DataMetadata
    tags: Optional[List[str]] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        """Initialize computed fields after object creation."""
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            'version_id': self.version_id,
            'data_hash': self.data_hash,
            'timestamp': self.timestamp.isoformat(),
            'source_path': self.source_path,
            'metadata': self.metadata.to_dict(),
            'tags': self.tags,
            'description': self.description
        }
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataVersion":
        """Create DataVersion from dictionary."""
        metadata_dict = data.pop('metadata')
        metadata = DataMetadata.from_dict(metadata_dict)
        
        timestamp_str = data.pop('timestamp')
        timestamp = datetime.fromisoformat(timestamp_str)
        
        return cls(
            metadata=metadata,
            timestamp=timestamp,
            **data
        )
    
    def __eq__(self, other) -> bool:
        """Check equality based on version_id and data_hash."""
        if not isinstance(other, DataVersion):
            return False
        return (self.version_id == other.version_id and 
                self.data_hash == other.data_hash)
    
    def __hash__(self) -> int:
        """Hash based on version_id for use in sets/dicts."""
        return hash(self.version_id)


@dataclass
class DataLineage:
    """Tracks transformation relationships between data versions.
    
    Attributes:
        transformation_id: Unique identifier for this transformation
        input_versions: List of input version IDs
        output_version: Output version ID
        transformation_type: Type of transformation (e.g., 'preprocessing', 'split')
        parameters: Transformation parameters and configuration
        timestamp: When transformation was performed
        code_hash: Optional hash of transformation code for reproducibility
        environment: Optional environment information
    """
    transformation_id: str
    input_versions: List[str]
    output_version: str
    transformation_type: str
    parameters: Dict[str, Any]
    timestamp: Optional[datetime] = None
    code_hash: Optional[str] = None
    environment: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        """Initialize computed fields after object creation."""
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        if self.timestamp:
            result['timestamp'] = self.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataLineage":
        """Create DataLineage from dictionary."""
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class DataVersionManager:
    """Main interface for data version management operations.
    
    This class provides comprehensive data versioning capabilities including
    version creation, storage, retrieval, and lineage tracking.
    """
    
    def __init__(self, storage_path: str):
        """Initialize DataVersionManager.
        
        Parameters
        ----------
        storage_path : str
            Directory path for storing version data and metadata
        """
        self.storage_path = os.path.abspath(storage_path)
        self.versions_dir = os.path.join(self.storage_path, "versions")
        self.lineage_dir = os.path.join(self.storage_path, "lineage")
        self.data_dir = os.path.join(self.storage_path, "data")
        
        # Create storage directories
        os.makedirs(self.versions_dir, exist_ok=True)
        os.makedirs(self.lineage_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def _compute_data_hash(self, data: pd.DataFrame) -> str:
        """Compute content-based hash for data integrity.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame to hash
            
        Returns
        -------
        str
            SHA256 hash of the data content
        """
        # Convert DataFrame to string representation for hashing
        # Use sort to ensure consistent ordering
        data_sorted = data.sort_index(axis=1).sort_index(axis=0)
        data_string = data_sorted.to_string()
        
        # Compute SHA256 hash
        hash_object = hashlib.sha256(data_string.encode('utf-8'))
        return hash_object.hexdigest()
    
    def _generate_version_id(self, base_name: Optional[str] = None) -> str:
        """Generate unique version ID.
        
        Parameters
        ----------
        base_name : str, optional
            Base name for version ID generation
            
        Returns
        -------
        str
            Unique version identifier
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:17]  # Include microseconds, truncate to 3 digits
        if base_name:
            return f"{base_name}_{timestamp}"
        else:
            return f"v{timestamp}"
    
    def create_version(self, data: pd.DataFrame, source_path: str, 
                      version_id: Optional[str] = None,
                      tags: Optional[List[str]] = None,
                      description: Optional[str] = None) -> DataVersion:
        """Create a new data version.
        
        Parameters
        ----------
        data : pd.DataFrame
            Dataset to version
        source_path : str
            Original source path or identifier
        version_id : str, optional
            Custom version ID. If None, auto-generated
        tags : List[str], optional
            Tags for categorization
        description : str, optional
            Human-readable description
            
        Returns
        -------
        DataVersion
            Created data version object
        """
        if version_id is None:
            version_id = self._generate_version_id()
        
        # Compute data hash and metadata
        data_hash = self._compute_data_hash(data)
        metadata = DataMetadata.from_dataframe(data)
        
        # Create version object
        version = DataVersion(
            version_id=version_id,
            data_hash=data_hash,
            timestamp=datetime.now(),
            source_path=source_path,
            metadata=metadata,
            tags=tags or [],
            description=description
        )
        
        self.logger.info(f"Created data version {version_id} with hash {data_hash[:8]}...")
        return version
    
    def save_version(self, version: DataVersion, data: pd.DataFrame) -> None:
        """Save version metadata and data to storage.
        
        Parameters
        ----------
        version : DataVersion
            Version metadata to save
        data : pd.DataFrame
            Actual dataset to save
        """
        # Save metadata
        metadata_path = os.path.join(self.versions_dir, f"{version.version_id}.json")
        with open(metadata_path, 'w') as f:
            json.dump(version.to_dict(), f, indent=2)
        
        # Save data
        data_path = os.path.join(self.data_dir, f"{version.version_id}.parquet")
        data.to_parquet(data_path, index=False)
        
        self.logger.info(f"Saved version {version.version_id} to storage")
    
    def load_version(self, version_id: str) -> DataVersion:
        """Load version metadata from storage.
        
        Parameters
        ----------
        version_id : str
            Version ID to load
            
        Returns
        -------
        DataVersion
            Loaded version object
            
        Raises
        ------
        FileNotFoundError
            If version does not exist
        """
        metadata_path = os.path.join(self.versions_dir, f"{version_id}.json")
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Version {version_id} not found")
        
        with open(metadata_path, 'r') as f:
            version_dict = json.load(f)
        
        return DataVersion.from_dict(version_dict)
    
    def load_data(self, version_id: str) -> pd.DataFrame:
        """Load actual dataset for a version.
        
        Parameters
        ----------
        version_id : str
            Version ID to load data for
            
        Returns
        -------
        pd.DataFrame
            Dataset for the specified version
            
        Raises
        ------
        FileNotFoundError
            If version data does not exist
        """
        data_path = os.path.join(self.data_dir, f"{version_id}.parquet")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data for version {version_id} not found")
        
        return pd.read_parquet(data_path)
    
    def list_versions(self) -> List[DataVersion]:
        """List all available data versions.
        
        Returns
        -------
        List[DataVersion]
            List of all versions sorted by timestamp
        """
        versions = []
        
        for filename in os.listdir(self.versions_dir):
            if filename.endswith('.json'):
                version_id = filename[:-5]  # Remove .json extension
                try:
                    version = self.load_version(version_id)
                    versions.append(version)
                except Exception as e:
                    self.logger.warning(f"Failed to load version {version_id}: {e}")
        
        # Sort by timestamp
        versions.sort(key=lambda v: v.timestamp)
        return versions
    
    def track_transformation(self, transformation_id: str,
                           input_versions: List[str],
                           output_version: str,
                           transformation_type: str,
                           parameters: Dict[str, Any],
                           code_hash: Optional[str] = None,
                           environment: Optional[Dict[str, str]] = None) -> DataLineage:
        """Track a data transformation in the lineage.
        
        Parameters
        ----------
        transformation_id : str
            Unique identifier for transformation
        input_versions : List[str]
            List of input version IDs
        output_version : str
            Output version ID
        transformation_type : str
            Type of transformation
        parameters : Dict[str, Any]
            Transformation parameters
        code_hash : str, optional
            Hash of transformation code
        environment : Dict[str, str], optional
            Environment information
            
        Returns
        -------
        DataLineage
            Created lineage object
        """
        lineage = DataLineage(
            transformation_id=transformation_id,
            input_versions=input_versions,
            output_version=output_version,
            transformation_type=transformation_type,
            parameters=parameters,
            code_hash=code_hash,
            environment=environment
        )
        
        # Save lineage
        lineage_path = os.path.join(self.lineage_dir, f"{transformation_id}.json")
        with open(lineage_path, 'w') as f:
            json.dump(lineage.to_dict(), f, indent=2)
        
        self.logger.info(f"Tracked transformation {transformation_id}: {input_versions} -> {output_version}")
        return lineage
    
    def get_lineage_history(self, version_id: str) -> List[DataLineage]:
        """Get transformation lineage history for a version.
        
        Parameters
        ----------
        version_id : str
            Version ID to get lineage for
            
        Returns
        -------
        List[DataLineage]
            List of transformations that produced this version
        """
        lineage_history = []
        
        for filename in os.listdir(self.lineage_dir):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(self.lineage_dir, filename), 'r') as f:
                        lineage_dict = json.load(f)
                    
                    lineage = DataLineage.from_dict(lineage_dict)
                    
                    # Check if this transformation involves our version
                    if (version_id == lineage.output_version or 
                        version_id in lineage.input_versions):
                        lineage_history.append(lineage)
                        
                except Exception as e:
                    self.logger.warning(f"Failed to load lineage {filename}: {e}")
        
        # Sort by timestamp
        lineage_history.sort(key=lambda item: item.timestamp or datetime.min)
        return lineage_history
    
    def verify_data_integrity(self, version_id: str) -> bool:
        """Verify data integrity using stored hash.
        
        Parameters
        ----------
        version_id : str
            Version ID to verify
            
        Returns
        -------
        bool
            True if data integrity is verified, False otherwise
        """
        try:
            version = self.load_version(version_id)
            data = self.load_data(version_id)
            
            computed_hash = self._compute_data_hash(data)
            return computed_hash == version.data_hash
            
        except Exception as e:
            self.logger.error(f"Failed to verify integrity for {version_id}: {e}")
            return False
    
    def cleanup_old_versions(self, keep_latest: int = 10) -> int:
        """Clean up old versions keeping only the latest N versions.
        
        Parameters
        ----------
        keep_latest : int, optional
            Number of latest versions to keep, by default 10
            
        Returns
        -------
        int
            Number of versions cleaned up
        """
        versions = self.list_versions()
        
        if len(versions) <= keep_latest:
            return 0
        
        # Keep the latest versions
        versions_to_remove = versions[:-keep_latest]
        removed_count = 0
        
        for version in versions_to_remove:
            try:
                # Remove metadata and data files
                metadata_path = os.path.join(self.versions_dir, f"{version.version_id}.json")
                data_path = os.path.join(self.data_dir, f"{version.version_id}.parquet")
                
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
                if os.path.exists(data_path):
                    os.remove(data_path)
                
                removed_count += 1
                self.logger.info(f"Cleaned up version {version.version_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to cleanup version {version.version_id}: {e}")
        
        return removed_count


def create_data_version(data: pd.DataFrame, source_path: str,
                       version_id: Optional[str] = None,
                       storage_path: str = "./data_versions",
                       tags: Optional[List[str]] = None,
                       description: Optional[str] = None) -> DataVersion:
    """Convenience function to create and save a data version.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataset to version
    source_path : str
        Original source path
    version_id : str, optional
        Custom version ID
    storage_path : str, optional
        Storage directory path
    tags : List[str], optional
        Version tags
    description : str, optional
        Version description
        
    Returns
    -------
    DataVersion
        Created and saved version
    """
    manager = DataVersionManager(storage_path)
    version = manager.create_version(
        data=data,
        source_path=source_path,
        version_id=version_id,
        tags=tags,
        description=description
    )
    manager.save_version(version, data)
    return version


def track_data_transformation(transformation_id: str,
                            input_versions: List[str],
                            output_version: str,
                            transformation_type: str,
                            parameters: Dict[str, Any],
                            storage_path: str = "./data_versions",
                            code_hash: Optional[str] = None,
                            environment: Optional[Dict[str, str]] = None) -> DataLineage:
    """Convenience function to track a data transformation.
    
    Parameters
    ----------
    transformation_id : str
        Unique transformation identifier
    input_versions : List[str]
        Input version IDs
    output_version : str
        Output version ID
    transformation_type : str
        Type of transformation
    parameters : Dict[str, Any]
        Transformation parameters
    storage_path : str, optional
        Storage directory path
    code_hash : str, optional
        Code hash for reproducibility
    environment : Dict[str, str], optional
        Environment information
        
    Returns
    -------
    DataLineage
        Created lineage record
    """
    manager = DataVersionManager(storage_path)
    return manager.track_transformation(
        transformation_id=transformation_id,
        input_versions=input_versions,
        output_version=output_version,
        transformation_type=transformation_type,
        parameters=parameters,
        code_hash=code_hash,
        environment=environment
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Data versioning CLI")
    parser.add_argument("--storage", default="./data_versions", 
                       help="Storage directory path")
    parser.add_argument("--list", action="store_true",
                       help="List all versions")
    parser.add_argument("--verify", type=str,
                       help="Verify integrity of specific version")
    parser.add_argument("--cleanup", type=int,
                       help="Clean up old versions, keep N latest")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    manager = DataVersionManager(args.storage)
    
    if args.list:
        versions = manager.list_versions()
        print(f"\nFound {len(versions)} versions:")
        for version in versions:
            print(f"  {version.version_id}: {version.metadata.rows} rows, "
                  f"{version.metadata.columns} cols, {version.source_path}")
    
    if args.verify:
        is_valid = manager.verify_data_integrity(args.verify)
        print(f"Version {args.verify} integrity: {'VALID' if is_valid else 'INVALID'}")
    
    if args.cleanup:
        removed = manager.cleanup_old_versions(keep_latest=args.cleanup)
        print(f"Cleaned up {removed} old versions")