"""
Data versioning and lineage tracking system.

This module provides comprehensive data version control capabilities
with Git-like versioning, lineage tracking, and rollback functionality.
"""

import hashlib
import json
import logging
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import pandas as pd
import numpy as np

from ..logging_config import get_logger

logger = get_logger(__name__)


class VersionType(Enum):
    """Types of data versions."""
    MAJOR = "major"      # Breaking changes
    MINOR = "minor"      # Feature additions
    PATCH = "patch"      # Bug fixes, small changes
    SNAPSHOT = "snapshot"  # Temporary versions


@dataclass
class DataVersion:
    """Data version metadata."""
    version_id: str
    version_number: str
    version_type: VersionType
    parent_version: Optional[str]
    timestamp: datetime
    author: str
    message: str
    data_hash: str
    schema_hash: str
    stats: Dict[str, Any]
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['version_type'] = self.version_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataVersion":
        """Create from dictionary."""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['version_type'] = VersionType(data['version_type'])
        return cls(**data)


@dataclass
class DataLineage:
    """Data lineage information."""
    version_id: str
    source_files: List[str]
    transformations: List[Dict[str, Any]]
    dependencies: List[str]
    outputs: List[str]
    execution_time: float
    environment: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataLineage":
        """Create from dictionary."""
        return cls(**data)


class DataVersionManager:
    """
    Data version control system with Git-like functionality.
    
    Features:
    - Semantic versioning for datasets
    - Content-based hashing for integrity
    - Lineage tracking and dependency management
    - Branch and tag support
    - Rollback and diff capabilities
    - Metadata and schema versioning
    """
    
    def __init__(self, repository_path: Union[str, Path], author: str = "system"):
        """
        Initialize data version manager.
        
        Args:
            repository_path: Path to version repository
            author: Default author for versions
        """
        self.repository_path = Path(repository_path)
        self.author = author
        self.current_branch = "main"
        
        # Create repository structure
        self._init_repository()
        
        logger.info(f"DataVersionManager initialized at {self.repository_path}")
    
    def _init_repository(self):
        """Initialize repository structure."""
        self.repository_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.repository_path / "versions").mkdir(exist_ok=True)
        (self.repository_path / "data").mkdir(exist_ok=True)
        (self.repository_path / "metadata").mkdir(exist_ok=True)
        (self.repository_path / "lineage").mkdir(exist_ok=True)
        (self.repository_path / "branches").mkdir(exist_ok=True)
        (self.repository_path / "tags").mkdir(exist_ok=True)
        
        # Create config file if it doesn't exist
        config_file = self.repository_path / "config.json"
        if not config_file.exists():
            config = {
                "repository_version": "1.0",
                "created_at": datetime.utcnow().isoformat(),
                "default_branch": "main",
                "current_branch": "main"
            }
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
    
    def commit(
        self,
        data: pd.DataFrame,
        message: str,
        version_type: VersionType = VersionType.MINOR,
        tags: Optional[List[str]] = None,
        lineage: Optional[DataLineage] = None
    ) -> str:
        """
        Commit a new data version.
        
        Args:
            data: DataFrame to version
            message: Commit message
            version_type: Type of version (major, minor, patch)
            tags: Optional tags for this version
            lineage: Optional lineage information
            
        Returns:
            Version ID of the committed data
        """
        try:
            logger.info(f"Committing new data version: {message}")
            
            # Generate content hashes
            data_hash = self._compute_data_hash(data)
            schema_hash = self._compute_schema_hash(data)
            
            # Get parent version
            parent_version = self._get_latest_version()
            
            # Generate version number
            version_number = self._generate_version_number(version_type, parent_version)
            
            # Create version ID
            version_id = f"v{version_number}_{data_hash[:8]}"
            
            # Compute data statistics
            stats = self._compute_data_stats(data)
            
            # Create version metadata
            version = DataVersion(
                version_id=version_id,
                version_number=version_number,
                version_type=version_type,
                parent_version=parent_version.version_id if parent_version else None,
                timestamp=datetime.utcnow(),
                author=self.author,
                message=message,
                data_hash=data_hash,
                schema_hash=schema_hash,
                stats=stats,
                tags=tags or []
            )
            
            # Save data and metadata
            self._save_data(data, version_id)
            self._save_version_metadata(version)
            
            # Save lineage if provided
            if lineage:
                lineage.version_id = version_id
                self._save_lineage(lineage)
            
            # Update branch reference
            self._update_branch_reference(self.current_branch, version_id)
            
            logger.info(f"Successfully committed version {version_id}")
            return version_id
            
        except Exception as e:
            logger.error(f"Failed to commit data version: {e}")
            raise
    
    def checkout(self, version_ref: str) -> pd.DataFrame:
        """
        Checkout a specific version of data.
        
        Args:
            version_ref: Version ID, tag, or branch name
            
        Returns:
            DataFrame for the specified version
        """
        try:
            version_id = self._resolve_version_reference(version_ref)
            
            if not version_id:
                raise ValueError(f"Version reference not found: {version_ref}")
            
            logger.info(f"Checking out version {version_id}")
            
            # Load data
            data = self._load_data(version_id)
            
            # Verify integrity
            version = self._load_version_metadata(version_id)
            computed_hash = self._compute_data_hash(data)
            
            if computed_hash != version.data_hash:
                logger.warning(f"Data integrity check failed for version {version_id}")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to checkout version {version_ref}: {e}")
            raise
    
    def list_versions(
        self,
        branch: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[DataVersion]:
        """
        List available data versions.
        
        Args:
            branch: Branch to list versions from (current branch if None)
            limit: Maximum number of versions to return
            
        Returns:
            List of DataVersion objects
        """
        try:
            versions = []
            metadata_dir = self.repository_path / "metadata"
            
            for metadata_file in metadata_dir.glob("*.json"):
                version_id = metadata_file.stem
                version = self._load_version_metadata(version_id)
                versions.append(version)
            
            # Sort by timestamp (newest first)
            versions.sort(key=lambda v: v.timestamp, reverse=True)
            
            # Apply limit
            if limit:
                versions = versions[:limit]
            
            return versions
            
        except Exception as e:
            logger.error(f"Failed to list versions: {e}")
            return []
    
    def diff(self, version1: str, version2: str) -> Dict[str, Any]:
        """
        Compare two data versions.
        
        Args:
            version1: First version reference
            version2: Second version reference
            
        Returns:
            Diff information
        """
        try:
            # Resolve version references
            v1_id = self._resolve_version_reference(version1)
            v2_id = self._resolve_version_reference(version2)
            
            if not v1_id or not v2_id:
                raise ValueError("Invalid version references")
            
            # Load versions
            v1_meta = self._load_version_metadata(v1_id)
            v2_meta = self._load_version_metadata(v2_id)
            
            # Load data for comparison
            data1 = self._load_data(v1_id)
            data2 = self._load_data(v2_id)
            
            # Compute differences
            diff = {
                "version1": v1_meta.to_dict(),
                "version2": v2_meta.to_dict(),
                "schema_changes": self._diff_schemas(data1, data2),
                "data_changes": self._diff_data(data1, data2),
                "stats_changes": self._diff_stats(v1_meta.stats, v2_meta.stats)
            }
            
            return diff
            
        except Exception as e:
            logger.error(f"Failed to compute diff between {version1} and {version2}: {e}")
            raise
    
    def create_branch(self, branch_name: str, from_version: Optional[str] = None) -> bool:
        """
        Create a new branch.
        
        Args:
            branch_name: Name of the new branch
            from_version: Version to branch from (current if None)
            
        Returns:
            Success status
        """
        try:
            if from_version:
                version_id = self._resolve_version_reference(from_version)
            else:
                latest = self._get_latest_version()
                version_id = latest.version_id if latest else None
            
            if not version_id:
                raise ValueError("No version to branch from")
            
            # Create branch reference
            branch_file = self.repository_path / "branches" / f"{branch_name}.json"
            if branch_file.exists():
                raise ValueError(f"Branch {branch_name} already exists")
            
            branch_info = {
                "branch_name": branch_name,
                "head": version_id,
                "created_at": datetime.utcnow().isoformat(),
                "created_from": from_version or "current"
            }
            
            with open(branch_file, 'w') as f:
                json.dump(branch_info, f, indent=2)
            
            logger.info(f"Created branch {branch_name} from {version_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create branch {branch_name}: {e}")
            return False
    
    def switch_branch(self, branch_name: str) -> bool:
        """
        Switch to a different branch.
        
        Args:
            branch_name: Name of the branch to switch to
            
        Returns:
            Success status
        """
        try:
            branch_file = self.repository_path / "branches" / f"{branch_name}.json"
            if not branch_file.exists():
                raise ValueError(f"Branch {branch_name} does not exist")
            
            self.current_branch = branch_name
            
            # Update config
            config_file = self.repository_path / "config.json"
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            config["current_branch"] = branch_name
            
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Switched to branch {branch_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to switch to branch {branch_name}: {e}")
            return False
    
    def create_tag(self, tag_name: str, version_ref: str, message: str = "") -> bool:
        """
        Create a tag for a specific version.
        
        Args:
            tag_name: Name of the tag
            version_ref: Version reference to tag
            message: Optional tag message
            
        Returns:
            Success status
        """
        try:
            version_id = self._resolve_version_reference(version_ref)
            if not version_id:
                raise ValueError(f"Version not found: {version_ref}")
            
            tag_file = self.repository_path / "tags" / f"{tag_name}.json"
            if tag_file.exists():
                raise ValueError(f"Tag {tag_name} already exists")
            
            tag_info = {
                "tag_name": tag_name,
                "version_id": version_id,
                "message": message,
                "created_at": datetime.utcnow().isoformat(),
                "author": self.author
            }
            
            with open(tag_file, 'w') as f:
                json.dump(tag_info, f, indent=2)
            
            logger.info(f"Created tag {tag_name} for version {version_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create tag {tag_name}: {e}")
            return False
    
    def get_lineage(self, version_ref: str) -> Optional[DataLineage]:
        """
        Get lineage information for a version.
        
        Args:
            version_ref: Version reference
            
        Returns:
            DataLineage object or None if not found
        """
        try:
            version_id = self._resolve_version_reference(version_ref)
            if not version_id:
                return None
            
            lineage_file = self.repository_path / "lineage" / f"{version_id}.json"
            if not lineage_file.exists():
                return None
            
            with open(lineage_file, 'r') as f:
                lineage_data = json.load(f)
            
            return DataLineage.from_dict(lineage_data)
            
        except Exception as e:
            logger.error(f"Failed to get lineage for {version_ref}: {e}")
            return None
    
    def _compute_data_hash(self, data: pd.DataFrame) -> str:
        """Compute hash of data content."""
        # Sort data for consistent hashing
        sorted_data = data.sort_index(axis=1).sort_values(by=list(data.columns))
        data_bytes = pickle.dumps(sorted_data.values.tobytes())
        return hashlib.sha256(data_bytes).hexdigest()
    
    def _compute_schema_hash(self, data: pd.DataFrame) -> str:
        """Compute hash of data schema."""
        schema_info = {
            'columns': list(data.columns),
            'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()},
            'shape': data.shape
        }
        schema_bytes = json.dumps(schema_info, sort_keys=True).encode('utf-8')
        return hashlib.sha256(schema_bytes).hexdigest()
    
    def _compute_data_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Compute statistics for data."""
        stats = {
            'shape': data.shape,
            'memory_usage': data.memory_usage(deep=True).sum(),
            'missing_values': data.isnull().sum().sum(),
            'duplicate_rows': data.duplicated().sum(),
            'numeric_columns': len(data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(data.select_dtypes(include=['object', 'category']).columns)
        }
        
        # Add numeric statistics for numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) > 0:
            stats['numeric_stats'] = {
                'mean': numeric_data.mean().to_dict(),
                'std': numeric_data.std().to_dict(),
                'min': numeric_data.min().to_dict(),
                'max': numeric_data.max().to_dict()
            }
        
        return stats
    
    def _save_data(self, data: pd.DataFrame, version_id: str):
        """Save data to disk."""
        data_file = self.repository_path / "data" / f"{version_id}.parquet"
        data.to_parquet(data_file, index=False)
    
    def _load_data(self, version_id: str) -> pd.DataFrame:
        """Load data from disk."""
        data_file = self.repository_path / "data" / f"{version_id}.parquet"
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found for version {version_id}")
        return pd.read_parquet(data_file)
    
    def _save_version_metadata(self, version: DataVersion):
        """Save version metadata."""
        metadata_file = self.repository_path / "metadata" / f"{version.version_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump(version.to_dict(), f, indent=2)
    
    def _load_version_metadata(self, version_id: str) -> DataVersion:
        """Load version metadata."""
        metadata_file = self.repository_path / "metadata" / f"{version_id}.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata not found for version {version_id}")
        
        with open(metadata_file, 'r') as f:
            data = json.load(f)
        
        return DataVersion.from_dict(data)
    
    def _save_lineage(self, lineage: DataLineage):
        """Save lineage information."""
        lineage_file = self.repository_path / "lineage" / f"{lineage.version_id}.json"
        with open(lineage_file, 'w') as f:
            json.dump(lineage.to_dict(), f, indent=2)
    
    def _get_latest_version(self) -> Optional[DataVersion]:
        """Get the latest version in current branch."""
        branch_file = self.repository_path / "branches" / f"{self.current_branch}.json"
        if not branch_file.exists():
            return None
        
        with open(branch_file, 'r') as f:
            branch_info = json.load(f)
        
        head_version = branch_info.get("head")
        if not head_version:
            return None
        
        try:
            return self._load_version_metadata(head_version)
        except FileNotFoundError:
            return None
    
    def _generate_version_number(self, version_type: VersionType, parent: Optional[DataVersion]) -> str:
        """Generate next version number."""
        if not parent:
            return "1.0.0"
        
        parts = parent.version_number.split('.')
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        
        if version_type == VersionType.MAJOR:
            major += 1
            minor = 0
            patch = 0
        elif version_type == VersionType.MINOR:
            minor += 1
            patch = 0
        elif version_type == VersionType.PATCH:
            patch += 1
        
        return f"{major}.{minor}.{patch}"
    
    def _update_branch_reference(self, branch_name: str, version_id: str):
        """Update branch to point to new version."""
        branch_file = self.repository_path / "branches" / f"{branch_name}.json"
        
        if branch_file.exists():
            with open(branch_file, 'r') as f:
                branch_info = json.load(f)
        else:
            branch_info = {
                "branch_name": branch_name,
                "created_at": datetime.utcnow().isoformat()
            }
        
        branch_info["head"] = version_id
        branch_info["updated_at"] = datetime.utcnow().isoformat()
        
        with open(branch_file, 'w') as f:
            json.dump(branch_info, f, indent=2)
    
    def _resolve_version_reference(self, ref: str) -> Optional[str]:
        """Resolve version reference to version ID."""
        # Check if it's already a version ID
        if ref.startswith('v') and '_' in ref:
            metadata_file = self.repository_path / "metadata" / f"{ref}.json"
            if metadata_file.exists():
                return ref
        
        # Check if it's a tag
        tag_file = self.repository_path / "tags" / f"{ref}.json"
        if tag_file.exists():
            with open(tag_file, 'r') as f:
                tag_info = json.load(f)
            return tag_info.get("version_id")
        
        # Check if it's a branch
        branch_file = self.repository_path / "branches" / f"{ref}.json"
        if branch_file.exists():
            with open(branch_file, 'r') as f:
                branch_info = json.load(f)
            return branch_info.get("head")
        
        return None
    
    def _diff_schemas(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Dict[str, Any]:
        """Compare schemas of two datasets."""
        cols1, cols2 = set(data1.columns), set(data2.columns)
        
        return {
            "added_columns": list(cols2 - cols1),
            "removed_columns": list(cols1 - cols2),
            "common_columns": list(cols1 & cols2),
            "dtype_changes": {
                col: {
                    "old": str(data1[col].dtype),
                    "new": str(data2[col].dtype)
                }
                for col in (cols1 & cols2)
                if str(data1[col].dtype) != str(data2[col].dtype)
            }
        }
    
    def _diff_data(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Dict[str, Any]:
        """Compare data content of two datasets."""
        return {
            "row_count_change": len(data2) - len(data1),
            "shape_change": {
                "old": data1.shape,
                "new": data2.shape
            }
        }
    
    def _diff_stats(self, stats1: Dict[str, Any], stats2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare statistics of two datasets."""
        changes = {}
        
        for key in set(stats1.keys()) | set(stats2.keys()):
            if key in stats1 and key in stats2:
                if stats1[key] != stats2[key]:
                    changes[key] = {
                        "old": stats1[key],
                        "new": stats2[key]
                    }
            elif key in stats1:
                changes[key] = {"old": stats1[key], "new": None}
            else:
                changes[key] = {"old": None, "new": stats2[key]}
        
        return changes


# CLI interface
def main():
    """CLI interface for data versioning operations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Version Manager CLI")
    parser.add_argument("command", choices=["init", "commit", "checkout", "list", "diff", "branch", "tag"])
    parser.add_argument("--repo", default="./data_repo", help="Repository path")
    parser.add_argument("--data", help="Data file path")
    parser.add_argument("--message", "-m", help="Commit message")
    parser.add_argument("--version", help="Version reference")
    parser.add_argument("--version2", help="Second version for diff")
    parser.add_argument("--branch", help="Branch name")
    parser.add_argument("--tag", help="Tag name")
    parser.add_argument("--type", choices=["major", "minor", "patch"], default="minor", help="Version type")
    
    args = parser.parse_args()
    
    # Initialize version manager
    if args.command == "init":
        manager = DataVersionManager(args.repo)
        print(f"Initialized data repository at {args.repo}")
        return
    
    manager = DataVersionManager(args.repo)
    
    if args.command == "commit":
        if not args.data or not args.message:
            print("Error: --data and --message required for commit")
            return
        
        data = pd.read_csv(args.data)
        version_type = VersionType(args.type)
        version_id = manager.commit(data, args.message, version_type)
        print(f"Committed version: {version_id}")
    
    elif args.command == "checkout":
        if not args.version:
            print("Error: --version required for checkout")
            return
        
        data = manager.checkout(args.version)
        print(f"Checked out version {args.version}: {data.shape}")
    
    elif args.command == "list":
        versions = manager.list_versions(limit=10)
        print(f"Found {len(versions)} versions:")
        for version in versions:
            print(f"  {version.version_id} - {version.message} ({version.timestamp})")
    
    elif args.command == "diff":
        if not args.version or not args.version2:
            print("Error: --version and --version2 required for diff")
            return
        
        diff = manager.diff(args.version, args.version2)
        print(f"Diff between {args.version} and {args.version2}:")
        print(json.dumps(diff, indent=2, default=str))
    
    elif args.command == "branch":
        if args.branch:
            success = manager.create_branch(args.branch, args.version)
            if success:
                print(f"Created branch: {args.branch}")
            else:
                print("Failed to create branch")
        else:
            print("Error: --branch required")
    
    elif args.command == "tag":
        if args.tag and args.version:
            success = manager.create_tag(args.tag, args.version, args.message or "")
            if success:
                print(f"Created tag: {args.tag}")
            else:
                print("Failed to create tag")
        else:
            print("Error: --tag and --version required")


if __name__ == "__main__":
    main()