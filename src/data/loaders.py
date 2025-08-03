"""
Advanced data loading with multiple source support.

This module provides flexible data loading capabilities with support for
various data sources, formats, and preprocessing options.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import json
import asyncio
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import requests

from ..logging_config import get_logger

logger = get_logger(__name__)


class DataLoader(ABC):
    """
    Abstract base class for data loaders.
    
    Provides a consistent interface for loading data from various sources
    with validation, preprocessing, and error handling.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data loader.
        
        Args:
            config: Configuration dictionary for the loader
        """
        self.config = config or {}
        self.metadata = {}
        self.last_load_time = None
        
    @abstractmethod
    def load(self, source: str, **kwargs) -> pd.DataFrame:
        """
        Load data from source.
        
        Args:
            source: Data source identifier
            **kwargs: Additional parameters
            
        Returns:
            Loaded DataFrame
        """
        pass
    
    @abstractmethod
    def validate_source(self, source: str) -> bool:
        """
        Validate that the data source is accessible.
        
        Args:
            source: Data source identifier
            
        Returns:
            True if source is valid and accessible
        """
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the last loaded dataset."""
        return self.metadata.copy()
    
    def _update_metadata(self, data: pd.DataFrame, source: str):
        """Update metadata after loading data."""
        self.metadata = {
            "source": source,
            "shape": data.shape,
            "columns": list(data.columns),
            "dtypes": data.dtypes.to_dict(),
            "memory_usage": data.memory_usage(deep=True).sum(),
            "missing_values": data.isnull().sum().to_dict(),
            "load_time": datetime.utcnow().isoformat(),
            "duplicate_rows": data.duplicated().sum()
        }
        self.last_load_time = datetime.utcnow()


class FileDataLoader(DataLoader):
    """
    Data loader for file-based sources (CSV, JSON, Parquet, etc.).
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize file data loader."""
        super().__init__(config)
        self.supported_formats = {'.csv', '.json', '.parquet', '.xlsx', '.tsv'}
    
    def load(self, source: str, **kwargs) -> pd.DataFrame:
        """
        Load data from a file.
        
        Args:
            source: File path
            **kwargs: Additional pandas parameters
            
        Returns:
            Loaded DataFrame
        """
        if not self.validate_source(source):
            raise ValueError(f"Invalid or inaccessible file: {source}")
        
        try:
            file_path = Path(source)
            extension = file_path.suffix.lower()
            
            logger.info(f"Loading data from {source}")
            
            # Load based on file extension
            if extension == '.csv':
                data = self._load_csv(source, **kwargs)
            elif extension == '.json':
                data = self._load_json(source, **kwargs)
            elif extension == '.parquet':
                data = self._load_parquet(source, **kwargs)
            elif extension == '.xlsx':
                data = self._load_excel(source, **kwargs)
            elif extension == '.tsv':
                kwargs.setdefault('sep', '\t')
                data = self._load_csv(source, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {extension}")
            
            self._update_metadata(data, source)
            logger.info(f"Successfully loaded {data.shape[0]} rows and {data.shape[1]} columns")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load data from {source}: {e}")
            raise
    
    def validate_source(self, source: str) -> bool:
        """Validate file source."""
        try:
            file_path = Path(source)
            
            # Check if file exists and is readable
            if not file_path.exists():
                logger.warning(f"File does not exist: {source}")
                return False
            
            if not file_path.is_file():
                logger.warning(f"Path is not a file: {source}")
                return False
            
            # Check file extension
            if file_path.suffix.lower() not in self.supported_formats:
                logger.warning(f"Unsupported file format: {file_path.suffix}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Source validation failed for {source}: {e}")
            return False
    
    def _load_csv(self, source: str, **kwargs) -> pd.DataFrame:
        """Load CSV file with enhanced error handling."""
        try:
            # Set default parameters
            default_params = {
                'encoding': 'utf-8',
                'na_values': ['', 'NULL', 'null', 'N/A', 'n/a', 'NA'],
                'keep_default_na': True
            }
            default_params.update(kwargs)
            
            return pd.read_csv(source, **default_params)
            
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'iso-8859-1', 'cp1252']:
                try:
                    kwargs['encoding'] = encoding
                    logger.warning(f"Retrying with encoding: {encoding}")
                    return pd.read_csv(source, **kwargs)
                except UnicodeDecodeError:
                    continue
            raise
    
    def _load_json(self, source: str, **kwargs) -> pd.DataFrame:
        """Load JSON file."""
        with open(source, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            return pd.DataFrame([data])
        else:
            raise ValueError("JSON data must be a list or dictionary")
    
    def _load_parquet(self, source: str, **kwargs) -> pd.DataFrame:
        """Load Parquet file."""
        try:
            import pyarrow.parquet as pq
            return pd.read_parquet(source, **kwargs)
        except ImportError:
            raise ImportError("pyarrow is required for Parquet support")
    
    def _load_excel(self, source: str, **kwargs) -> pd.DataFrame:
        """Load Excel file."""
        try:
            return pd.read_excel(source, **kwargs)
        except ImportError:
            raise ImportError("openpyxl is required for Excel support")


class DatabaseDataLoader(DataLoader):
    """
    Data loader for database sources.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize database data loader."""
        super().__init__(config)
        self.connection = None
        self.connection_params = self.config.get('connection', {})
    
    def load(self, source: str, **kwargs) -> pd.DataFrame:
        """
        Load data from database query.
        
        Args:
            source: SQL query or table name
            **kwargs: Additional parameters
            
        Returns:
            Loaded DataFrame
        """
        if not self.validate_source(source):
            raise ValueError(f"Invalid database source: {source}")
        
        try:
            logger.info(f"Executing database query: {source[:100]}...")
            
            # Use provided connection or create new one
            conn = kwargs.get('connection', self.connection)
            if conn is None:
                conn = self._create_connection()
            
            # Determine if source is a query or table name
            if source.strip().upper().startswith('SELECT'):
                data = pd.read_sql_query(source, conn, **kwargs)
            else:
                data = pd.read_sql_table(source, conn, **kwargs)
            
            self._update_metadata(data, f"Database: {source}")
            logger.info(f"Successfully loaded {data.shape[0]} rows from database")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load data from database: {e}")
            raise
        finally:
            # Close connection if we created it
            if conn != self.connection and conn is not None:
                conn.close()
    
    def validate_source(self, source: str) -> bool:
        """Validate database source."""
        try:
            if not source or not isinstance(source, str):
                return False
            
            # Basic SQL injection protection
            dangerous_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE']
            upper_source = source.upper()
            
            for keyword in dangerous_keywords:
                if keyword in upper_source and not upper_source.strip().startswith('SELECT'):
                    logger.error(f"Potentially dangerous SQL detected: {keyword}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Database source validation failed: {e}")
            return False
    
    def _create_connection(self):
        """Create database connection based on config."""
        try:
            # This is a simplified implementation
            # In production, you would use proper database drivers
            import sqlite3
            
            db_type = self.connection_params.get('type', 'sqlite')
            
            if db_type == 'sqlite':
                db_path = self.connection_params.get('path', ':memory:')
                return sqlite3.connect(db_path)
            else:
                raise ValueError(f"Unsupported database type: {db_type}")
                
        except Exception as e:
            logger.error(f"Failed to create database connection: {e}")
            raise


class APIDataLoader(DataLoader):
    """
    Data loader for REST API sources.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize API data loader."""
        super().__init__(config)
        self.session = requests.Session()
        self.headers = self.config.get('headers', {})
        self.timeout = self.config.get('timeout', 30)
        
        # Set default headers
        if self.headers:
            self.session.headers.update(self.headers)
    
    def load(self, source: str, **kwargs) -> pd.DataFrame:
        """
        Load data from REST API.
        
        Args:
            source: API endpoint URL
            **kwargs: Additional request parameters
            
        Returns:
            Loaded DataFrame
        """
        if not self.validate_source(source):
            raise ValueError(f"Invalid API source: {source}")
        
        try:
            logger.info(f"Fetching data from API: {source}")
            
            method = kwargs.pop('method', 'GET')
            params = kwargs.pop('params', {})
            data = kwargs.pop('data', None)
            
            response = self.session.request(
                method=method,
                url=source,
                params=params,
                json=data,
                timeout=self.timeout,
                **kwargs
            )
            
            response.raise_for_status()
            
            # Parse response based on content type
            content_type = response.headers.get('content-type', '').lower()
            
            if 'application/json' in content_type:
                json_data = response.json()
                if isinstance(json_data, list):
                    df = pd.DataFrame(json_data)
                elif isinstance(json_data, dict):
                    # Handle nested JSON structures
                    if 'data' in json_data:
                        df = pd.DataFrame(json_data['data'])
                    elif 'results' in json_data:
                        df = pd.DataFrame(json_data['results'])
                    else:
                        df = pd.DataFrame([json_data])
                else:
                    raise ValueError("Unexpected JSON structure")
            
            elif 'text/csv' in content_type:
                from io import StringIO
                df = pd.read_csv(StringIO(response.text))
            
            else:
                raise ValueError(f"Unsupported content type: {content_type}")
            
            self._update_metadata(df, source)
            logger.info(f"Successfully loaded {df.shape[0]} rows from API")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data from API {source}: {e}")
            raise
    
    def validate_source(self, source: str) -> bool:
        """Validate API source."""
        try:
            if not source or not isinstance(source, str):
                return False
            
            # Basic URL validation
            if not source.startswith(('http://', 'https://')):
                logger.warning(f"Invalid URL scheme: {source}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"API source validation failed: {e}")
            return False


class CreditDataLoader(FileDataLoader):
    """
    Specialized data loader for credit scoring datasets.
    
    Extends FileDataLoader with credit-specific validation and preprocessing.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize credit data loader."""
        super().__init__(config)
        self.required_columns = self.config.get('required_columns', ['age', 'income', 'credit_score'])
        self.protected_attributes = self.config.get('protected_attributes', ['age'])
        self.target_column = self.config.get('target_column', 'approved')
    
    def load(self, source: str, **kwargs) -> pd.DataFrame:
        """
        Load and validate credit dataset.
        
        Args:
            source: Data source
            **kwargs: Additional parameters
            
        Returns:
            Validated credit dataset
        """
        # Load data using parent method
        data = super().load(source, **kwargs)
        
        # Perform credit-specific validation and preprocessing
        data = self._validate_credit_data(data)
        data = self._preprocess_credit_data(data)
        
        return data
    
    def load_with_split(
        self,
        source: str,
        test_size: float = 0.3,
        random_state: int = 42,
        stratify: bool = True,
        **kwargs
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Load data and split into train/test sets.
        
        Args:
            source: Data source
            test_size: Proportion of test set
            random_state: Random seed for reproducibility
            stratify: Whether to stratify split by target
            **kwargs: Additional load parameters
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        data = self.load(source, **kwargs)
        
        if self.target_column not in data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in data")
        
        X = data.drop(columns=[self.target_column])
        y = data[self.target_column]
        
        stratify_by = y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_by
        )
        
        logger.info(f"Split data: train={len(X_train)}, test={len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def _validate_credit_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate credit-specific data requirements."""
        logger.info("Validating credit dataset")
        
        # Check for required columns
        missing_columns = [col for col in self.required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for target column
        if self.target_column not in data.columns:
            logger.warning(f"Target column '{self.target_column}' not found - creating synthetic target")
            # Create synthetic target for demo purposes
            data[self.target_column] = np.random.binomial(1, 0.7, len(data))
        
        # Validate data types and ranges
        self._validate_feature_ranges(data)
        
        return data
    
    def _validate_feature_ranges(self, data: pd.DataFrame):
        """Validate that features are within expected ranges."""
        validations = {
            'age': (18, 100),
            'income': (0, 1000000),
            'credit_score': (300, 850),
            'debt_to_income': (0, 1.0)
        }
        
        for feature, (min_val, max_val) in validations.items():
            if feature in data.columns:
                out_of_range = data[
                    (data[feature] < min_val) | (data[feature] > max_val)
                ]
                
                if len(out_of_range) > 0:
                    logger.warning(f"Found {len(out_of_range)} {feature} values outside expected range [{min_val}, {max_val}]")
    
    def _preprocess_credit_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply credit-specific preprocessing."""
        logger.info("Applying credit data preprocessing")
        
        data = data.copy()
        
        # Create protected attribute if not present
        if 'protected' not in data.columns and self.protected_attributes:
            primary_protected = self.protected_attributes[0]
            if primary_protected in data.columns:
                if primary_protected == 'age':
                    # Create binary protected attribute based on age
                    data['protected'] = (data['age'] >= 40).astype(int)
                else:
                    # For other attributes, create binary encoding
                    median_val = data[primary_protected].median()
                    data['protected'] = (data[primary_protected] >= median_val).astype(int)
        
        # Handle missing values
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if data[col].isnull().any():
                data[col].fillna(data[col].median(), inplace=True)
        
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_columns:
            if data[col].isnull().any():
                data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else 'Unknown', inplace=True)
        
        return data
    
    async def load_async(self, source: str, **kwargs) -> pd.DataFrame:
        """
        Asynchronously load credit data.
        
        Args:
            source: Data source
            **kwargs: Additional parameters
            
        Returns:
            Loaded DataFrame
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.load, source, **kwargs)


class DataLoaderFactory:
    """
    Factory for creating appropriate data loaders.
    """
    
    @staticmethod
    def create_loader(source_type: str, config: Optional[Dict[str, Any]] = None) -> DataLoader:
        """
        Create appropriate data loader based on source type.
        
        Args:
            source_type: Type of data source ('file', 'database', 'api', 'credit')
            config: Configuration for the loader
            
        Returns:
            Configured data loader
        """
        loaders = {
            'file': FileDataLoader,
            'database': DatabaseDataLoader,
            'api': APIDataLoader,
            'credit': CreditDataLoader
        }
        
        if source_type not in loaders:
            raise ValueError(f"Unknown source type: {source_type}. Available: {list(loaders.keys())}")
        
        return loaders[source_type](config)


# CLI interface
def main():
    """CLI interface for data loading operations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Loader CLI")
    parser.add_argument("source", help="Data source (file path, URL, query)")
    parser.add_argument("--type", choices=['file', 'database', 'api', 'credit'], default='file', help="Source type")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--sample", type=int, help="Sample N rows")
    parser.add_argument("--info", action="store_true", help="Show dataset info")
    
    args = parser.parse_args()
    
    # Create loader
    loader = DataLoaderFactory.create_loader(args.type)
    
    try:
        # Load data
        print(f"Loading data from {args.source}...")
        data = loader.load(args.source)
        
        # Sample if requested
        if args.sample and args.sample < len(data):
            data = data.sample(n=args.sample, random_state=42)
            print(f"Sampled {args.sample} rows")
        
        # Show info
        if args.info:
            print("\nDataset Info:")
            print(f"Shape: {data.shape}")
            print(f"Columns: {list(data.columns)}")
            print(f"Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            print(f"Missing values: {data.isnull().sum().sum()}")
            
            metadata = loader.get_metadata()
            print(f"Load time: {metadata.get('load_time')}")
        
        # Save if requested
        if args.output:
            data.to_csv(args.output, index=False)
            print(f"Data saved to {args.output}")
        
        print(f"Successfully loaded {len(data)} rows")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()