"""
Data layer package for fair credit scoring system.

This package provides comprehensive data management capabilities including:
- Data loading and preprocessing
- Data validation and quality checking
- Data versioning and lineage tracking
- Feature stores and caching
- Database connections and query optimization

Modules:
    - loaders: Data loading from various sources
    - validators: Data quality and integrity validation
    - versioning: Data version control and lineage
    - stores: Feature stores and caching mechanisms
    - connections: Database connections and query builders
"""

__version__ = "0.2.0"

from .loaders import DataLoader, CreditDataLoader
from .validators import DataValidator, FairnessValidator
from .versioning import DataVersionManager
from .stores import FeatureStore, CacheManager

__all__ = [
    "DataLoader",
    "CreditDataLoader", 
    "DataValidator",
    "FairnessValidator",
    "DataVersionManager",
    "FeatureStore",
    "CacheManager"
]