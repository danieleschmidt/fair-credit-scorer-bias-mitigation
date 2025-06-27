from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train_baseline_model(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    sample_weight: Iterable[float] | None = None,
    solver: str = "liblinear",
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
        Solver to use in ``LogisticRegression``. Defaults to ``"liblinear"`` so
        tests can assert the configured solver.
    """
    model = LogisticRegression(max_iter=1000, solver=solver)
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
