"""Data loading and preprocessing utilities for credit scoring datasets.

This module handles all data loading operations, including reading from CSV files
and generating synthetic datasets for testing and development. It provides consistent
data splitting and preprocessing for model training and evaluation with integrated
data versioning capabilities.

Functions:
    load_credit_dataset: Load entire dataset without splitting into train/test
    load_credit_data: Load and split data into train/test sets with validation
    train_test_split_validated: Validated train/test splitting utility with comprehensive error handling
    generate_synthetic_credit_data: Generate synthetic credit data for testing
    load_versioned_credit_data: Load data with automatic versioning support

Features:
- Automatic synthetic data generation if CSV file doesn't exist
- Configurable column names and data generation parameters via configuration system
- Stratified train/test splitting with validation
- Comprehensive input validation and specific error handling
- Support for custom file paths and random states
- Integrated data versioning and lineage tracking
- Automatic version creation for data transformations

Configuration:
    Data parameters are managed through the configuration system:
    - data.default_dataset_path: Default CSV file location
    - data.label_column_name/protected_column_name: Column name mappings
    - data.synthetic.*: Parameters for synthetic data generation
    - data.versioning.*: Data versioning configuration

Example:
    >>> from data_loader_preprocessor import load_credit_data, load_versioned_credit_data
    >>> # Load and split data with versioning
    >>> X_train, X_test, y_train, y_test = load_versioned_credit_data(test_size=0.3, enable_versioning=True)
    >>> 
    >>> # Load entire dataset without splitting
    >>> X, y = load_credit_dataset("path/to/data.csv")
    >>> 
    >>> # Validated splitting with error handling
    >>> X_train, X_test, y_train, y_test = train_test_split_validated(X, y, test_size=0.2)

The module automatically generates realistic synthetic credit scoring data when
real datasets are not available, ensuring consistent development and testing workflows.
Data versioning integration provides complete audit trails for reproducible ML pipelines.
"""

import logging
import os
from typing import Tuple

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

try:
    from .config import get_config
except ImportError:
    from config import get_config

logger = logging.getLogger(__name__)

# Optional import for data versioning (graceful fallback if not available)
try:
    from data_versioning import DataVersionManager
    VERSIONING_AVAILABLE = True
except ImportError:
    logger.debug("Data versioning module not available")
    VERSIONING_AVAILABLE = False


def generate_synthetic_credit_data(n_samples=10000, n_features=10, n_informative=5,
                                  n_redundant=2, random_state=None):
    """Generate synthetic credit scoring data for benchmarking and testing.
    
    Parameters
    ----------
    n_samples : int, optional
        Number of samples to generate, by default 10000
    n_features : int, optional
        Total number of features, by default 10
    n_informative : int, optional
        Number of informative features, by default 5
    n_redundant : int, optional
        Number of redundant features, by default 2
    random_state : int, optional
        Random seed for reproducibility, by default 42
        
    Returns
    -------
    tuple
        (X, y, sensitive_features) where:
        - X is the feature matrix (without protected attribute)
        - y is the target vector
        - sensitive_features is the protected attribute vector
        
    Raises
    ------
    ValueError
        If n_informative > n_features or other invalid parameters
    """
    if n_informative > n_features:
        raise ValueError(f"n_informative ({n_informative}) cannot exceed n_features ({n_features})")
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if n_features <= 0:
        raise ValueError(f"n_features must be positive, got {n_features}")

    # Use config default if random_state not provided
    if random_state is None:
        config = get_config()
        random_state = config.data.random_state

    logger.debug(f"Generating {n_samples} synthetic credit samples with {n_features} features, random_state={random_state}")

    # Generate base features and target
    X_full, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_clusters_per_class=1,
        random_state=random_state
    )

    # Create protected attribute based on first feature
    # This creates a realistic correlation pattern
    sensitive_features = (X_full[:, 0] > X_full[:, 0].mean()).astype(int)

    # Return features without the protected attribute embedded
    X = X_full[:, 1:]  # Remove first feature used for protected attribute

    return X, y, sensitive_features


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


def load_versioned_credit_data(
    path: str = "data/credit_data.csv",
    test_size: float = 0.3,
    random_state: int = 42,
    enable_versioning: bool = True,
    version_storage_path: str = "./data_versions",
    version_description: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load credit data with automatic versioning and lineage tracking.
    
    This function extends load_credit_data with integrated data versioning
    capabilities, automatically tracking data lineage and creating versions
    for the loaded dataset and train/test splits.
    
    Parameters
    ----------
    path : str, optional
        Path to the CSV file, by default "data/credit_data.csv"
    test_size : float, optional
        Proportion of data for testing, by default 0.3
    random_state : int, optional
        Random seed for reproducibility, by default 42
    enable_versioning : bool, optional
        Whether to enable data versioning, by default True
    version_storage_path : str, optional
        Directory for version storage, by default "./data_versions"
    version_description : str, optional
        Description for the created versions
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test splits
        
    Raises
    ------
    ImportError
        If versioning is enabled but data_versioning module is not available
    ValueError
        If path is empty, test_size is invalid, or random_state is negative
    TypeError
        If parameters have wrong types
    FileNotFoundError
        If path directory cannot be created
    PermissionError
        If lacking permissions to read/write files
    """
    if enable_versioning and not VERSIONING_AVAILABLE:
        raise ImportError(
            "Data versioning is enabled but data_versioning module is not available. "
            "Either disable versioning (enable_versioning=False) or ensure the module is installed."
        )

    logger.info(f"Loading credit data with versioning={'enabled' if enable_versioning else 'disabled'}")

    # Load data using standard function
    X_train, X_test, y_train, y_test = load_credit_data(
        path=path,
        test_size=test_size,
        random_state=random_state
    )

    if enable_versioning:
        try:
            # Initialize version manager
            manager = DataVersionManager(version_storage_path)

            # Create combined dataset for versioning
            X_combined = pd.concat([X_train, X_test], ignore_index=True)
            y_combined = pd.concat([y_train, y_test], ignore_index=True)

            # Create version for original dataset
            original_version = manager.create_version(
                data=pd.concat([X_combined, y_combined], axis=1),
                source_path=path,
                version_id=f"original_{manager._generate_version_id()}",
                description=version_description or f"Original dataset loaded from {path}",
                tags=["original", "credit_data"]
            )
            manager.save_version(original_version, pd.concat([X_combined, y_combined], axis=1))

            # Create versions for train and test splits
            train_data = pd.concat([X_train, y_train], axis=1)
            test_data = pd.concat([X_test, y_test], axis=1)

            train_version = manager.create_version(
                data=train_data,
                source_path=f"{path}_train_split",
                version_id=f"train_{manager._generate_version_id()}",
                description=f"Training split (test_size={test_size}, random_state={random_state})",
                tags=["train", "split"]
            )
            manager.save_version(train_version, train_data)

            test_version = manager.create_version(
                data=test_data,
                source_path=f"{path}_test_split",
                version_id=f"test_{manager._generate_version_id()}",
                description=f"Test split (test_size={test_size}, random_state={random_state})",
                tags=["test", "split"]
            )
            manager.save_version(test_version, test_data)

            # Track lineage for train/test split transformation
            manager.track_transformation(
                transformation_id=f"train_test_split_{random_state}_{int(test_size*100)}",
                input_versions=[original_version.version_id],
                output_version=train_version.version_id,
                transformation_type="train_test_split",
                parameters={
                    "test_size": test_size,
                    "random_state": random_state,
                    "stratify": True,
                    "split_type": "train"
                }
            )

            manager.track_transformation(
                transformation_id=f"train_test_split_{random_state}_{int(test_size*100)}_test",
                input_versions=[original_version.version_id],
                output_version=test_version.version_id,
                transformation_type="train_test_split",
                parameters={
                    "test_size": test_size,
                    "random_state": random_state,
                    "stratify": True,
                    "split_type": "test"
                }
            )

            logger.info(f"Created data versions: {original_version.version_id}, "
                       f"{train_version.version_id}, {test_version.version_id}")

        except Exception as e:
            logger.warning(f"Data versioning failed: {e}")
            if enable_versioning:
                logger.warning("Continuing without versioning")

    return X_train, X_test, y_train, y_test
