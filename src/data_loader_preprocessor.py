import os
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def load_credit_dataset(path="data/credit_data.csv", random_state=42):
    """Return the entire credit dataset as features and labels."""
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=random_state,
        )
        protected = (X[:, 0] > X[:, 0].mean()).astype(int)
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        df["protected"] = protected
        df["label"] = y
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)

    X = df.drop("label", axis=1)
    y = df["label"]
    return X, y


def load_credit_data(path="data/credit_data.csv", test_size=0.3, random_state=42):
    """Load credit data from CSV or generate a synthetic dataset.

    Parameters
    ----------
    path : str
        CSV file path to read or write the dataset.
    test_size : float
        Proportion of the dataset to include in the test split.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple(pd.DataFrame, pd.DataFrame, pd.Series, pd.Series)
        X_train, X_test, y_train, y_test
    """
    # Reuse ``load_credit_dataset`` so dataset generation logic lives in one place
    X, y = load_credit_dataset(path=path, random_state=random_state)
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
