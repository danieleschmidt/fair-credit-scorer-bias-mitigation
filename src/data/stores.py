"""
Feature stores and caching mechanisms for efficient data access.

This module provides high-performance feature stores with caching,
real-time feature serving, and offline batch processing capabilities.
"""

import json
import logging
import pickle
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np

from ..logging_config import get_logger

logger = get_logger(__name__)


class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"      # Least Recently Used
    LFU = "lfu"      # Least Frequently Used
    TTL = "ttl"      # Time To Live
    FIFO = "fifo"    # First In First Out


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    data: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    size_bytes: int
    ttl_seconds: Optional[int] = None
    
    def is_expired(self) -> bool:
        """Check if entry is expired based on TTL."""
        if self.ttl_seconds is None:
            return False
        return (datetime.utcnow() - self.created_at).total_seconds() > self.ttl_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding data)."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['last_accessed'] = self.last_accessed.isoformat()
        del data['data']  # Don't include actual data in metadata
        return data


class CacheManager:
    """
    High-performance cache manager with multiple eviction strategies.
    
    Features:
    - Multiple eviction strategies (LRU, LFU, TTL, FIFO)
    - Thread-safe operations
    - Configurable size limits
    - Cache statistics and monitoring
    - Persistence support
    """
    
    def __init__(
        self,
        max_size_mb: float = 1024,
        strategy: CacheStrategy = CacheStrategy.LRU,
        default_ttl: Optional[int] = None,
        persistence_path: Optional[str] = None
    ):
        """
        Initialize cache manager.
        
        Args:
            max_size_mb: Maximum cache size in MB
            strategy: Cache eviction strategy
            default_ttl: Default TTL in seconds
            persistence_path: Path for cache persistence
        """
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.strategy = strategy
        self.default_ttl = default_ttl
        self.persistence_path = Path(persistence_path) if persistence_path else None
        
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._current_size = 0
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'writes': 0
        }
        
        # Load from persistence if available
        if self.persistence_path and self.persistence_path.exists():
            self._load_from_disk()
        
        logger.info(f"CacheManager initialized with {max_size_mb}MB limit, strategy: {strategy.value}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        with self._lock:
            if key not in self._cache:
                self._stats['misses'] += 1
                return None
            
            entry = self._cache[key]
            
            # Check if expired
            if entry.is_expired():
                del self._cache[key]
                self._current_size -= entry.size_bytes
                self._stats['misses'] += 1
                return None
            
            # Update access metadata
            entry.last_accessed = datetime.utcnow()
            entry.access_count += 1
            
            self._stats['hits'] += 1
            logger.debug(f"Cache hit for key: {key}")
            return entry.data
    
    def put(self, key: str, data: Any, ttl: Optional[int] = None) -> bool:
        """
        Put value in cache.
        
        Args:
            key: Cache key
            data: Data to cache
            ttl: Time to live in seconds
            
        Returns:
            Success status
        """
        try:
            # Estimate data size
            data_size = self._estimate_size(data)
            
            with self._lock:
                # Check if we need to evict entries
                while self._current_size + data_size > self.max_size_bytes and self._cache:
                    self._evict_one()
                
                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    data=data,
                    created_at=datetime.utcnow(),
                    last_accessed=datetime.utcnow(),
                    access_count=1,
                    size_bytes=data_size,
                    ttl_seconds=ttl or self.default_ttl
                )
                
                # Remove existing entry if present
                if key in self._cache:
                    self._current_size -= self._cache[key].size_bytes
                
                # Add new entry
                self._cache[key] = entry
                self._current_size += data_size
                self._stats['writes'] += 1
                
                logger.debug(f"Cached data for key: {key} ({data_size} bytes)")
                return True
                
        except Exception as e:
            logger.error(f"Failed to cache data for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete entry from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Success status
        """
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                del self._cache[key]
                self._current_size -= entry.size_bytes
                logger.debug(f"Deleted cache entry: {key}")
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._current_size = 0
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                'entries': len(self._cache),
                'current_size_mb': self._current_size / 1024 / 1024,
                'max_size_mb': self.max_size_bytes / 1024 / 1024,
                'utilization': self._current_size / self.max_size_bytes,
                'hit_rate': hit_rate,
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'evictions': self._stats['evictions'],
                'writes': self._stats['writes']
            }
    
    def _evict_one(self):
        """Evict one entry based on strategy."""
        if not self._cache:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # Evict least recently used
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].last_accessed)
        elif self.strategy == CacheStrategy.LFU:
            # Evict least frequently used
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].access_count)
        elif self.strategy == CacheStrategy.TTL:
            # Evict expired entries first, then oldest
            expired_keys = [k for k, v in self._cache.items() if v.is_expired()]
            if expired_keys:
                oldest_key = expired_keys[0]
            else:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
        else:  # FIFO
            # Evict oldest by creation time
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
        
        entry = self._cache[oldest_key]
        del self._cache[oldest_key]
        self._current_size -= entry.size_bytes
        self._stats['evictions'] += 1
        
        logger.debug(f"Evicted cache entry: {oldest_key}")
    
    def _estimate_size(self, data: Any) -> int:
        """Estimate size of data object."""
        try:
            if isinstance(data, pd.DataFrame):
                return data.memory_usage(deep=True).sum()
            elif isinstance(data, (str, bytes)):
                return len(data)
            elif isinstance(data, (list, tuple)):
                return sum(self._estimate_size(item) for item in data)
            elif isinstance(data, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in data.items())
            else:
                # Use pickle for general objects
                return len(pickle.dumps(data))
        except Exception:
            # Fallback estimation
            return 1024  # 1KB default
    
    def _save_to_disk(self):
        """Save cache to disk for persistence."""
        if not self.persistence_path:
            return
        
        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save cache metadata and data separately
            metadata = {}
            data = {}
            
            for key, entry in self._cache.items():
                metadata[key] = entry.to_dict()
                data[key] = entry.data
            
            # Save metadata
            with open(self.persistence_path.with_suffix('.meta'), 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Save data
            with open(self.persistence_path.with_suffix('.data'), 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Cache saved to {self.persistence_path}")
            
        except Exception as e:
            logger.error(f"Failed to save cache to disk: {e}")
    
    def _load_from_disk(self):
        """Load cache from disk."""
        try:
            meta_file = self.persistence_path.with_suffix('.meta')
            data_file = self.persistence_path.with_suffix('.data')
            
            if not meta_file.exists() or not data_file.exists():
                return
            
            # Load metadata
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            
            # Load data
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
            
            # Reconstruct cache entries
            for key, meta in metadata.items():
                if key in data:
                    entry = CacheEntry(
                        key=key,
                        data=data[key],
                        created_at=datetime.fromisoformat(meta['created_at']),
                        last_accessed=datetime.fromisoformat(meta['last_accessed']),
                        access_count=meta['access_count'],
                        size_bytes=meta['size_bytes'],
                        ttl_seconds=meta.get('ttl_seconds')
                    )
                    
                    # Skip expired entries
                    if not entry.is_expired():
                        self._cache[key] = entry
                        self._current_size += entry.size_bytes
            
            logger.info(f"Cache loaded from {self.persistence_path}: {len(self._cache)} entries")
            
        except Exception as e:
            logger.error(f"Failed to load cache from disk: {e}")


class FeatureStore(ABC):
    """Abstract base class for feature stores."""
    
    @abstractmethod
    def get_features(self, feature_names: List[str], entity_ids: List[str]) -> pd.DataFrame:
        """Get features for entities."""
        pass
    
    @abstractmethod
    def put_features(self, features: pd.DataFrame, entity_column: str) -> bool:
        """Store features."""
        pass
    
    @abstractmethod
    def list_features(self) -> List[str]:
        """List available features."""
        pass


class InMemoryFeatureStore(FeatureStore):
    """
    In-memory feature store with caching capabilities.
    
    Designed for high-performance feature serving with automatic
    caching and expiration policies.
    """
    
    def __init__(
        self,
        cache_manager: Optional[CacheManager] = None,
        default_ttl: int = 3600  # 1 hour
    ):
        """
        Initialize in-memory feature store.
        
        Args:
            cache_manager: Cache manager instance
            default_ttl: Default TTL for features in seconds
        """
        self.cache_manager = cache_manager or CacheManager()
        self.default_ttl = default_ttl
        self._feature_metadata: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        
        logger.info("InMemoryFeatureStore initialized")
    
    def get_features(self, feature_names: List[str], entity_ids: List[str]) -> pd.DataFrame:
        """
        Get features for specified entities.
        
        Args:
            feature_names: List of feature names to retrieve
            entity_ids: List of entity IDs
            
        Returns:
            DataFrame with features for entities
        """
        logger.debug(f"Getting features {feature_names} for {len(entity_ids)} entities")
        
        result_data = []
        
        for entity_id in entity_ids:
            entity_features = {'entity_id': entity_id}
            
            for feature_name in feature_names:
                cache_key = f"feature:{feature_name}:entity:{entity_id}"
                feature_value = self.cache_manager.get(cache_key)
                
                if feature_value is not None:
                    entity_features[feature_name] = feature_value
                else:
                    # Feature not found - could implement fallback to persistent store
                    entity_features[feature_name] = None
            
            result_data.append(entity_features)
        
        return pd.DataFrame(result_data)
    
    def put_features(self, features: pd.DataFrame, entity_column: str = 'entity_id') -> bool:
        """
        Store features in the feature store.
        
        Args:
            features: DataFrame containing features
            entity_column: Name of the entity ID column
            
        Returns:
            Success status
        """
        try:
            logger.debug(f"Storing features for {len(features)} entities")
            
            if entity_column not in features.columns:
                raise ValueError(f"Entity column '{entity_column}' not found in features")
            
            feature_columns = [col for col in features.columns if col != entity_column]
            
            # Store each feature value
            for _, row in features.iterrows():
                entity_id = row[entity_column]
                
                for feature_name in feature_columns:
                    cache_key = f"feature:{feature_name}:entity:{entity_id}"
                    feature_value = row[feature_name]
                    
                    # Store in cache
                    self.cache_manager.put(cache_key, feature_value, ttl=self.default_ttl)
                    
                    # Update feature metadata
                    self._update_feature_metadata(feature_name, feature_value)
            
            logger.info(f"Successfully stored {len(feature_columns)} features for {len(features)} entities")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store features: {e}")
            return False
    
    def list_features(self) -> List[str]:
        """List all available features."""
        with self._lock:
            return list(self._feature_metadata.keys())
    
    def get_feature_metadata(self, feature_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific feature."""
        with self._lock:
            return self._feature_metadata.get(feature_name)
    
    def batch_get_features(
        self,
        requests: List[Tuple[List[str], List[str]]]
    ) -> List[pd.DataFrame]:
        """
        Batch get features for multiple requests.
        
        Args:
            requests: List of (feature_names, entity_ids) tuples
            
        Returns:
            List of DataFrames, one per request
        """
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(self.get_features, feature_names, entity_ids)
                for feature_names, entity_ids in requests
            ]
            
            return [future.result() for future in futures]
    
    def _update_feature_metadata(self, feature_name: str, feature_value: Any):
        """Update metadata for a feature."""
        with self._lock:
            if feature_name not in self._feature_metadata:
                self._feature_metadata[feature_name] = {
                    'first_seen': datetime.utcnow().isoformat(),
                    'data_type': type(feature_value).__name__,
                    'sample_values': [],
                    'update_count': 0
                }
            
            metadata = self._feature_metadata[feature_name]
            metadata['last_updated'] = datetime.utcnow().isoformat()
            metadata['update_count'] += 1
            
            # Keep sample values (up to 10)
            if len(metadata['sample_values']) < 10:
                if feature_value not in metadata['sample_values']:
                    metadata['sample_values'].append(feature_value)


class FileBasedFeatureStore(FeatureStore):
    """
    File-based feature store with partitioning and compression.
    
    Suitable for batch processing and offline feature serving.
    """
    
    def __init__(
        self,
        storage_path: Union[str, Path],
        partition_by: Optional[str] = None,
        compression: str = 'snappy'
    ):
        """
        Initialize file-based feature store.
        
        Args:
            storage_path: Path to feature storage directory
            partition_by: Column to partition data by
            compression: Compression algorithm for parquet files
        """
        self.storage_path = Path(storage_path)
        self.partition_by = partition_by
        self.compression = compression
        
        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata
        self.metadata_file = self.storage_path / "_metadata.json"
        self._load_metadata()
        
        logger.info(f"FileBasedFeatureStore initialized at {self.storage_path}")
    
    def get_features(self, feature_names: List[str], entity_ids: List[str]) -> pd.DataFrame:
        """Get features from file storage."""
        try:
            # Find relevant files
            feature_files = []
            for feature_name in feature_names:
                feature_dir = self.storage_path / feature_name
                if feature_dir.exists():
                    for file_path in feature_dir.glob("*.parquet"):
                        feature_files.append((feature_name, file_path))
            
            if not feature_files:
                return pd.DataFrame(columns=['entity_id'] + feature_names)
            
            # Load and combine features
            all_features = []
            
            for feature_name, file_path in feature_files:
                try:
                    df = pd.read_parquet(file_path)
                    
                    # Filter by entity IDs if needed
                    if 'entity_id' in df.columns:
                        df = df[df['entity_id'].isin(entity_ids)]
                    
                    # Select relevant columns
                    relevant_cols = ['entity_id'] + [col for col in df.columns if col.startswith(feature_name)]
                    df = df[relevant_cols]
                    
                    all_features.append(df)
                    
                except Exception as e:
                    logger.warning(f"Failed to load feature file {file_path}: {e}")
            
            # Merge features
            if all_features:
                result = all_features[0]
                for df in all_features[1:]:
                    result = pd.merge(result, df, on='entity_id', how='outer')
                
                return result
            else:
                return pd.DataFrame(columns=['entity_id'] + feature_names)
                
        except Exception as e:
            logger.error(f"Failed to get features: {e}")
            return pd.DataFrame(columns=['entity_id'] + feature_names)
    
    def put_features(self, features: pd.DataFrame, entity_column: str = 'entity_id') -> bool:
        """Store features to file storage."""
        try:
            if entity_column not in features.columns:
                raise ValueError(f"Entity column '{entity_column}' not found")
            
            # Normalize entity column name
            if entity_column != 'entity_id':
                features = features.rename(columns={entity_column: 'entity_id'})
            
            feature_columns = [col for col in features.columns if col != 'entity_id']
            
            # Store each feature separately
            for feature_name in feature_columns:
                feature_data = features[['entity_id', feature_name]].copy()
                
                # Create feature directory
                feature_dir = self.storage_path / feature_name
                feature_dir.mkdir(exist_ok=True)
                
                # Generate filename with timestamp
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                filename = f"{feature_name}_{timestamp}.parquet"
                file_path = feature_dir / filename
                
                # Save to parquet
                feature_data.to_parquet(
                    file_path,
                    compression=self.compression,
                    index=False
                )
                
                # Update metadata
                self._update_feature_metadata(feature_name, len(feature_data), file_path)
            
            logger.info(f"Stored {len(feature_columns)} features to file storage")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store features: {e}")
            return False
    
    def list_features(self) -> List[str]:
        """List available features."""
        features = []
        for item in self.storage_path.iterdir():
            if item.is_dir() and not item.name.startswith('_'):
                features.append(item.name)
        return features
    
    def _load_metadata(self):
        """Load feature store metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
                self.metadata = {}
        else:
            self.metadata = {}
    
    def _save_metadata(self):
        """Save feature store metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def _update_feature_metadata(self, feature_name: str, record_count: int, file_path: Path):
        """Update metadata for a feature."""
        if feature_name not in self.metadata:
            self.metadata[feature_name] = {
                'created_at': datetime.utcnow().isoformat(),
                'files': [],
                'total_records': 0
            }
        
        self.metadata[feature_name]['last_updated'] = datetime.utcnow().isoformat()
        self.metadata[feature_name]['total_records'] += record_count
        self.metadata[feature_name]['files'].append({
            'path': str(file_path),
            'records': record_count,
            'created_at': datetime.utcnow().isoformat()
        })
        
        self._save_metadata()


# CLI interface
def main():
    """CLI interface for feature store operations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Feature Store CLI")
    parser.add_argument("command", choices=["put", "get", "list", "cache-stats"])
    parser.add_argument("--data", help="Data file path")
    parser.add_argument("--features", nargs="+", help="Feature names")
    parser.add_argument("--entities", nargs="+", help="Entity IDs")
    parser.add_argument("--store-type", choices=["memory", "file"], default="memory")
    parser.add_argument("--store-path", help="Storage path for file-based store")
    
    args = parser.parse_args()
    
    # Create feature store
    if args.store_type == "file":
        if not args.store_path:
            print("Error: --store-path required for file-based store")
            return
        store = FileBasedFeatureStore(args.store_path)
    else:
        store = InMemoryFeatureStore()
    
    if args.command == "put":
        if not args.data:
            print("Error: --data required for put command")
            return
        
        data = pd.read_csv(args.data)
        success = store.put_features(data)
        print(f"Put features: {'success' if success else 'failed'}")
    
    elif args.command == "get":
        if not args.features or not args.entities:
            print("Error: --features and --entities required for get command")
            return
        
        features = store.get_features(args.features, args.entities)
        print(f"Retrieved features:\n{features}")
    
    elif args.command == "list":
        features = store.list_features()
        print(f"Available features: {features}")
    
    elif args.command == "cache-stats":
        if hasattr(store, 'cache_manager'):
            stats = store.cache_manager.get_stats()
            print("Cache Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        else:
            print("Cache statistics not available for this store type")


if __name__ == "__main__":
    main()