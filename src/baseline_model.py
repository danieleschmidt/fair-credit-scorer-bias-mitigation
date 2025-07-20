"""Baseline logistic regression model for credit scoring.

This module provides functionality to train and evaluate a baseline
logistic regression model for credit scoring. It serves as the foundation
for comparing bias mitigation techniques and establishes performance benchmarks.

Functions:
    train_baseline_model: Train a logistic regression classifier with configurable parameters
    evaluate_model: Evaluate model performance with optional threshold adjustment

Configuration:
    Model parameters are loaded from the centralized configuration system, supporting
    environment variable overrides for solver and max_iter parameters. Default values
    can be customized through config/default.yaml or FAIRNESS_* environment variables.

Example:
    >>> from baseline_model import train_baseline_model, evaluate_model
    >>> # Train with default configuration
    >>> model = train_baseline_model(X_train, y_train)
    >>> # Evaluate with custom threshold
    >>> accuracy, predictions = evaluate_model(model, X_test, y_test, threshold=0.6)
    >>> 
    >>> # Train with custom parameters
    >>> model = train_baseline_model(X_train, y_train, solver='lbfgs', max_iter=200)

The module supports both probability predictions and hard classifications,
making it suitable for fairness-aware evaluation pipelines that require
threshold optimization across different protected groups.
"""

from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

try:
    from .config import get_config
except ImportError:
    from config import get_config


def train_baseline_model(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    sample_weight: Iterable[float] | None = None,
    solver: str | None = None,
    max_iter: int | None = None,
) -> LogisticRegression:
    """Train a simple logistic regression model.

    Parameters
    ----------
    X_train : array-like
        Training features.
    y_train : array-like
        Labels for the training data.
    sample_weight : array-like or None, optional
        Sample weights passed to ``fit``.
    solver : str, optional
        Solver to use in ``LogisticRegression``. If None, uses configuration default.
    max_iter : int, optional
        Maximum iterations for solver. If None, uses configuration default.
    """
    config = get_config()
    
    # Use provided values or fall back to configuration defaults
    if solver is None:
        solver = config.model.logistic_regression.solver
    if max_iter is None:
        max_iter = config.model.logistic_regression.max_iter
    
    model = LogisticRegression(max_iter=max_iter, solver=solver)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def evaluate_model(
    model: LogisticRegression,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    sensitive_features: pd.Series | np.ndarray | None = None,
    return_probs: bool = False,
    threshold: float | None = None,
) -> Tuple[float, np.ndarray] | Tuple[float, np.ndarray, np.ndarray | None]:
    """Return accuracy score, predictions, and optionally probabilities.

    Parameters
    ----------
    model : estimator
        Fitted classifier implementing ``predict`` and optionally ``predict_proba``.
    X_test : array-like
        Test features.
    y_test : array-like
        True labels for the test set.
    sensitive_features : array-like or None, optional
        Protected attribute values corresponding to ``X_test``.
    return_probs : bool, optional
        If True, also return probability scores from ``predict_proba``.
    threshold : float or None, optional
        Decision threshold for converting probabilities to labels. When ``None``,
        the model's ``predict`` method is used directly.
    """

    use_proba = threshold is not None or return_probs
    if use_proba and hasattr(model, "predict_proba"):
        if sensitive_features is not None:
            proba = model.predict_proba(X_test, sensitive_features=sensitive_features)[
                :, 1
            ]
        else:
            proba = model.predict_proba(X_test)[:, 1]
        if threshold is None:
            predictions = (
                model.predict(X_test, sensitive_features=sensitive_features)
                if sensitive_features is not None
                else model.predict(X_test)
            )
        else:
            predictions = (proba >= threshold).astype(int)
    else:
        # Fall back to the model's predict method if predict_proba is unavailable
        predictions = (
            model.predict(X_test, sensitive_features=sensitive_features)
            if sensitive_features is not None
            else model.predict(X_test)
        )
        proba = None

    acc = accuracy_score(y_test, predictions)
    if return_probs:
        return acc, predictions, proba
    return acc, predictions
