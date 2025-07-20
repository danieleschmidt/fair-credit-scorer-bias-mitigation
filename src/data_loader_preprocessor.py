"""Data loading and preprocessing utilities for credit scoring datasets.

This module handles all data loading operations, including reading from CSV files
and generating synthetic datasets for testing and development. It provides consistent
data splitting and preprocessing for model training and evaluation.

Functions:
    load_credit_dataset: Load entire dataset without splitting into train/test
    load_credit_data: Load and split data into train/test sets with validation
    train_test_split_validated: Validated train/test splitting utility with comprehensive error handling

Features:
- Automatic synthetic data generation if CSV file doesn't exist
- Configurable column names and data generation parameters via configuration system
- Stratified train/test splitting with validation
- Comprehensive input validation and specific error handling
- Support for custom file paths and random states

Configuration:
    Data parameters are managed through the configuration system:
    - data.default_dataset_path: Default CSV file location
    - data.label_column_name/protected_column_name: Column name mappings
    - data.synthetic.*: Parameters for synthetic data generation

Example:
    >>> from data_loader_preprocessor import load_credit_data, load_credit_dataset
    >>> # Load and split data
    >>> X_train, X_test, y_train, y_test = load_credit_data(test_size=0.3)
    >>> 
    >>> # Load entire dataset without splitting
    >>> X, y = load_credit_dataset("path/to/data.csv")
    >>> 
    >>> # Validated splitting with error handling
    >>> X_train, X_test, y_train, y_test = train_test_split_validated(X, y, test_size=0.2)

The module automatically generates realistic synthetic credit scoring data when
real datasets are not available, ensuring consistent development and testing workflows.
"""

import os
from typing import Tuple
import logging

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

try:
    from .config import get_config
except ImportError:
    from config import get_config

logger = logging.getLogger(__name__)


def load_credit_dataset(
    path: str = "data/credit_data.csv", random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """Return the entire credit dataset as features and labels.
    
    Parameters
    ----------
    path : str
        Path to the CSV file. Defaults to configuration value.
    random_state : int
        Random seed for reproducibility. Defaults to configuration value.
        
    Returns
    -------
    tuple(pd.DataFrame, pd.Series)
        Features and labels
        
    Raises
    ------
    ValueError
        If path is empty or random_state is negative
    TypeError
        If path is not a string or random_state is not an integer
    FileNotFoundError
        If path directory cannot be created
    PermissionError
        If lacking permissions to read/write files
    """
    config = get_config()
    
    # Override defaults with configuration values if using defaults
    if path == "data/credit_data.csv":
        path = config.data.default_dataset_path
    if random_state == 42:
        random_state = config.general.default_random_state
    
    # Input validation
    if not isinstance(path, str):
        raise TypeError(f"path must be a string, got {type(path).__name__}")
    if not path.strip():
        raise ValueError("path cannot be empty or whitespace")
    if not isinstance(random_state, int):
        raise TypeError(f"random_state must be an integer, got {type(random_state).__name__}")
    if random_state < 0:
        raise ValueError(f"random_state must be non-negative, got {random_state}")
    
    # Normalize path
    path = os.path.normpath(path.strip())
    
    try:
        if os.path.exists(path):
            logger.info(f"Loading existing dataset from {path}")
            try:
                df = pd.read_csv(path)
                if df.empty:
                    raise ValueError(f"Dataset file {path} is empty")
                if config.data.label_column_name not in df.columns:
                    raise ValueError(f"Dataset file {path} missing required '{config.data.label_column_name}' column")
                logger.info(f"Successfully loaded {len(df)} rows from {path}")
            except pd.errors.EmptyDataError:
                raise ValueError(f"Dataset file {path} contains no data")
            except pd.errors.ParserError as e:
                raise ValueError(f"Failed to parse CSV file {path}: {e}")
        else:
            logger.info(f"Generating synthetic dataset and saving to {path}")
            X, y = make_classification(
                n_samples=config.data.synthetic.n_samples,
                n_features=config.data.synthetic.n_features,
                n_informative=config.data.synthetic.n_informative,
                n_redundant=config.data.synthetic.n_redundant,
                random_state=random_state,
            )
            protected = (X[:, 0] > X[:, 0].mean()).astype(int)
            df = pd.DataFrame(X, columns=[f"{config.data.feature_column_prefix}{i}" for i in range(X.shape[1])])
            df[config.data.protected_column_name] = protected
            df[config.data.label_column_name] = y
            
            # Create directory with proper error handling
            try:
                dir_path = os.path.dirname(path)
                if dir_path:  # Only create if there's actually a directory path
                    os.makedirs(dir_path, exist_ok=True)
                df.to_csv(path, index=config.data.csv_include_index)
                logger.info(f"Successfully saved {len(df)} rows to {path}")
            except PermissionError:
                raise PermissionError(f"Permission denied when creating directory or writing to {path}")
            except OSError as e:
                raise FileNotFoundError(f"Could not create directory or write file {path}: {e}")
    
    except pd.errors.EmptyDataError:
        raise ValueError(f"Invalid CSV format: File {path} is empty or contains no data")
    except pd.errors.ParserError as e:
        raise ValueError(f"Invalid CSV format: Unable to parse {path} - {e}")
    except (PermissionError, FileNotFoundError, OSError):
        # Re-raise file system errors as-is
        raise

    # Extract features and labels with validation
    try:
        if config.data.label_column_name not in df.columns:
            raise ValueError(f"Dataset missing required '{config.data.label_column_name}' column. Available columns: {list(df.columns)}")
        
        X = df.drop(config.data.label_column_name, axis=1)
        y = df[config.data.label_column_name]
        
        if X.empty:
            raise ValueError("No features found after removing label column")
        if len(X.columns) == 0:
            raise ValueError("Dataset has no feature columns")
        if y.empty:
            raise ValueError("No labels found")
            
        return X, y
    except KeyError as e:
        raise ValueError(f"Error extracting features and labels: {e}")
    except (TypeError, AttributeError) as e:
        raise ValueError(f"Invalid dataset format: {e}")


def load_credit_data(
    path: str = "data/credit_data.csv",
    test_size: float = 0.3,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load credit data from CSV or generate a synthetic dataset.

    Parameters
    ----------
    path : str
        CSV file path to read or write the dataset.
    test_size : float
        Proportion of the dataset to include in the test split.
        Must be between 0.0 and 1.0.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple(pd.DataFrame, pd.DataFrame, pd.Series, pd.Series)
        X_train, X_test, y_train, y_test
        
    Raises
    ------
    ValueError
        If test_size is not between 0.0 and 1.0
    TypeError
        If test_size is not a number
    """
    # Input validation for test_size
    if not isinstance(test_size, (int, float)):
        raise TypeError(f"test_size must be a number, got {type(test_size).__name__}")
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0.0 and 1.0, got {test_size}")
    
    try:
        # Reuse ``load_credit_dataset`` so dataset generation logic lives in one place
        X, y = load_credit_dataset(path=path, random_state=random_state)
        
        # Ensure we have enough samples for splitting
        if len(X) < 2:
            raise ValueError(f"Dataset has only {len(X)} samples, need at least 2 for train/test split")
        
        # Calculate minimum test size based on data size
        min_test_samples = 1
        min_test_size = min_test_samples / len(X)
        if test_size < min_test_size:
            raise ValueError(f"test_size {test_size} too small, minimum is {min_test_size:.3f} for {len(X)} samples")
        
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError:
        # Re-raise validation errors as-is
        raise
    except (TypeError, AttributeError) as e:
        raise ValueError(f"Invalid data format for train/test split: {e}")



def train_test_split_validated(X, y, test_size=0.3, random_state=None):
    """Validate inputs and perform train/test split with specific error handling.
    
    Args:
        X: Features array/list
        y: Labels array/list
        test_size: Fraction for test set (0.0 < test_size < 1.0)
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test
        
    Raises:
        ValueError: For validation errors (empty data, invalid sizes, etc.)
        TypeError: For invalid data types
    """
    # Validate inputs
    if len(X) == 0 or len(y) == 0:
        raise ValueError("Dataset cannot be empty")
    
    if len(X) != len(y):
        raise ValueError(f"Features and labels must have the same length, got {len(X)} and {len(y)}")
    
    if not isinstance(test_size, (int, float)):
        raise TypeError(f"test_size must be a number, got {type(test_size).__name__}")
    
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    
    try:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    except (TypeError, AttributeError) as e:
        raise ValueError(f"Invalid data format for train/test split: {e}")
