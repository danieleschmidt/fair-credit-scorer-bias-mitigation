import numpy as np
import pandas as pd
import pytest
from typing import Tuple


@pytest.fixture
def sample_credit_data() -> pd.DataFrame:
    """Generate sample credit scoring dataset for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.exponential(50000, n_samples),
        'credit_history_length': np.random.randint(0, 30, n_samples),
        'num_credit_accounts': np.random.randint(0, 20, n_samples),
        'debt_to_income_ratio': np.random.uniform(0, 2, n_samples),
        'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'employment_status': np.random.choice(['Employed', 'Self-employed', 'Unemployed'], n_samples),
        'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'race': np.random.choice(['White', 'Black', 'Hispanic', 'Asian', 'Other'], n_samples),
        'default': np.random.choice([0, 1], n_samples)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_processed_data(sample_credit_data) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate processed X, y data for testing."""
    df = sample_credit_data.copy()
    
    # Create target variable
    y = df['default']
    
    # Simple feature engineering
    X = pd.DataFrame({
        'age': df['age'],
        'income': df['income'],
        'credit_history_length': df['credit_history_length'],
        'num_credit_accounts': df['num_credit_accounts'],
        'debt_to_income_ratio': df['debt_to_income_ratio'],
        'education_high_school': (df['education_level'] == 'High School').astype(int),
        'education_bachelor': (df['education_level'] == 'Bachelor').astype(int),
        'education_master': (df['education_level'] == 'Master').astype(int),
        'education_phd': (df['education_level'] == 'PhD').astype(int),
        'employed': (df['employment_status'] == 'Employed').astype(int),
        'self_employed': (df['employment_status'] == 'Self-employed').astype(int),
        'unemployed': (df['employment_status'] == 'Unemployed').astype(int),
        'single': (df['marital_status'] == 'Single').astype(int),
        'married': (df['marital_status'] == 'Married').astype(int),
        'divorced': (df['marital_status'] == 'Divorced').astype(int),
        'male': (df['gender'] == 'Male').astype(int),
        'female': (df['gender'] == 'Female').astype(int),
        'race_white': (df['race'] == 'White').astype(int),
        'race_black': (df['race'] == 'Black').astype(int),
        'race_hispanic': (df['race'] == 'Hispanic').astype(int),
        'race_asian': (df['race'] == 'Asian').astype(int),
        'race_other': (df['race'] == 'Other').astype(int),
    })
    
    return X, y


@pytest.fixture
def sample_model_predictions() -> Tuple[np.ndarray, np.ndarray]:
    """Generate sample model predictions for testing fairness metrics."""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate predictions with some bias
    y_true = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    y_pred = y_true.copy()
    
    # Add some prediction errors
    error_indices = np.random.choice(n_samples, size=int(0.2 * n_samples), replace=False)
    y_pred[error_indices] = 1 - y_pred[error_indices]
    
    return y_true, y_pred


@pytest.fixture
def sample_sensitive_attributes() -> pd.Series:
    """Generate sample sensitive attributes for fairness testing."""
    np.random.seed(42)
    n_samples = 1000
    
    return pd.Series(np.random.choice(['group_a', 'group_b'], n_samples), name='sensitive_attr')


@pytest.fixture
def sample_config_dict() -> dict:
    """Generate sample configuration dictionary for testing."""
    return {
        'data': {
            'path': 'data/test_dataset.csv',
            'test_size': 0.3,
            'random_state': 42
        },
        'model': {
            'type': 'logistic_regression',
            'random_state': 42,
            'max_iter': 1000
        },
        'fairness': {
            'sensitive_attributes': ['gender', 'race'],
            'metrics': ['demographic_parity', 'equalized_odds'],
            'threshold': 0.5
        },
        'output': {
            'save_model': True,
            'save_metrics': True,
            'output_dir': 'models/'
        }
    }


@pytest.fixture
def sample_metrics_dict() -> dict:
    """Generate sample metrics dictionary for testing."""
    return {
        'accuracy': 0.85,
        'precision': 0.82,
        'recall': 0.78,
        'f1_score': 0.80,
        'roc_auc': 0.87,
        'demographic_parity_difference': 0.12,
        'equalized_odds_difference': 0.08,
        'true_positive_rate_difference': 0.06,
        'false_positive_rate_difference': 0.10,
        'accuracy_ratio': 0.95,
        'precision_ratio': 0.93,
        'recall_ratio': 0.91
    }


@pytest.fixture
def sample_empty_dataframe() -> pd.DataFrame:
    """Generate empty DataFrame for edge case testing."""
    return pd.DataFrame()


@pytest.fixture
def sample_single_row_dataframe() -> pd.DataFrame:
    """Generate single-row DataFrame for edge case testing."""
    return pd.DataFrame({
        'age': [25],
        'income': [50000],
        'default': [0]
    })


@pytest.fixture
def sample_data_with_missing_values() -> pd.DataFrame:
    """Generate sample data with missing values for robust testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.exponential(50000, n_samples),
        'default': np.random.choice([0, 1], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce missing values
    missing_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    df.loc[missing_indices, 'income'] = np.nan
    
    return df


@pytest.fixture
def temporary_csv_file(tmp_path, sample_credit_data):
    """Create a temporary CSV file for testing file operations."""
    csv_path = tmp_path / "test_data.csv"
    sample_credit_data.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def temporary_json_file(tmp_path, sample_metrics_dict):
    """Create a temporary JSON file for testing."""
    import json
    json_path = tmp_path / "test_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(sample_metrics_dict, f)
    return str(json_path)