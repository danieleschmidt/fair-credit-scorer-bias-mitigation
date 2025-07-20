"""Bias mitigation strategies for fair machine learning.

This module implements various bias mitigation techniques to reduce discrimination
in machine learning models. It provides pre-processing, in-processing, and
post-processing methods to improve fairness across different protected groups.

Mitigation methods available:
- expgrad_demographic_parity: In-processing using Exponentiated Gradient optimization
- reweight_samples: Pre-processing through sample reweighting for demographic parity
- postprocess_equalized_odds: Post-processing with threshold optimization

Each method targets different fairness criteria:
- Demographic parity: Equal selection rates across groups
- Equalized odds: Equal true positive and false positive rates across groups

Example:
    >>> from bias_mitigator import expgrad_demographic_parity, reweight_samples
    >>> # In-processing approach
    >>> fair_model = expgrad_demographic_parity(X_train, y_train, protected_attr)
    >>> 
    >>> # Pre-processing approach
    >>> sample_weights = reweight_samples(y_train, protected_attr)
    >>> model.fit(X_train, y_train, sample_weight=sample_weights)

The module integrates with fairlearn and scikit-learn to provide standardized
bias mitigation capabilities suitable for production ML systems.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression


def expgrad_demographic_parity(X, y, protected):
    """Train a classifier using the Exponentiated Gradient algorithm.

    The returned model is optimized for demographic parity using
    ``fairlearn.reductions.ExponentiatedGradient``.

    Parameters
    ----------
    X : pandas.DataFrame
        Training features.
    y : array-like
        Training labels.
    protected : array-like
        Protected attribute values for each training sample.

    Returns
    -------
    ExponentiatedGradient
        Fitted model enforcing demographic parity.
    """
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity

    base_est = LogisticRegression(max_iter=1000, solver="liblinear")
    mitigator = ExponentiatedGradient(base_est, DemographicParity())
    mitigator.fit(X, y, sensitive_features=protected)
    return mitigator


def reweight_samples(y, protected):
    """Return sample weights that balance label distribution across protected groups."""
    if not isinstance(protected, pd.Series):
        protected = pd.Series(protected, name="protected")

    df = pd.DataFrame({"y": y, "protected": protected})
    # compute weights per label/protected combination
    counts = df.value_counts().rename("count").reset_index()
    total = len(df)
    weights = {}
    for _, row in counts.iterrows():
        group = (row["y"], row["protected"])
        weights[group] = total / (len(counts) * row["count"])

    return [weights[(yi, pi)] for yi, pi in zip(df["y"], df["protected"])]


def postprocess_equalized_odds(model, X, y, protected):
    """Post-process predictions using the Equalized Odds constraint.

    Parameters
    ----------
    model : estimator
        A fitted scikit-learn compatible classifier with ``predict_proba``.
    X : pandas.DataFrame
        Training features.
    y : array-like
        Training labels.
    protected : array-like
        Protected attribute values for each training sample.

    Returns
    -------
    ThresholdOptimizer
        A post-processed classifier enforcing the equalized odds constraint.
    """
    from fairlearn.postprocessing import ThresholdOptimizer

    optimizer = ThresholdOptimizer(
        estimator=model,
        constraints="equalized_odds",
        predict_method="predict_proba",
        prefit=True,
    )
    optimizer.fit(X, y, sensitive_features=protected)
    return optimizer
