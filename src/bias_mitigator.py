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

try:
    from .config import get_config
    from .logging_config import get_logger
except ImportError:
    from config import get_config
    from logging_config import get_logger

logger = get_logger(__name__)


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
    from fairlearn.reductions import DemographicParity, ExponentiatedGradient

    logger.info("Starting Exponentiated Gradient training with demographic parity constraint")
    logger.debug(f"Training data shape: {X.shape}, protected groups: {len(set(protected))}")

    try:
        # Get configuration values
        config = get_config()
        max_iter = config.model.bias_mitigation.max_iter
        solver = config.model.bias_mitigation.solver

        base_est = LogisticRegression(max_iter=max_iter, solver=solver)
        logger.debug(f"Created base LogisticRegression estimator with max_iter={max_iter}, solver={solver}")

        mitigator = ExponentiatedGradient(base_est, DemographicParity())
        logger.debug("Initialized ExponentiatedGradient with DemographicParity constraint")

        mitigator.fit(X, y, sensitive_features=protected)
        logger.info("Successfully completed Exponentiated Gradient training")

        return mitigator

    except Exception as e:
        logger.error(f"Failed to train Exponentiated Gradient model: {e}")
        raise


def reweight_samples(y, protected):
    """Return sample weights that balance label distribution across protected groups."""
    logger.info("Computing sample weights for demographic parity reweighting")

    try:
        if not isinstance(protected, pd.Series):
            protected = pd.Series(protected, name="protected")

        df = pd.DataFrame({"y": y, "protected": protected})
        logger.debug(f"Created DataFrame with {len(df)} samples for weight computation")

        # compute weights per label/protected combination
        counts = df.value_counts().rename("count").reset_index()
        total = len(df)
        weights = {}

        logger.debug(f"Found {len(counts)} unique label/protected combinations")

        for _, row in counts.iterrows():
            group = (row["y"], row["protected"])
            weight = total / (len(counts) * row["count"])
            weights[group] = weight
            logger.debug(f"Group {group}: count={row['count']}, weight={weight:.4f}")

        sample_weights = [weights[(yi, pi)] for yi, pi in zip(df["y"], df["protected"])]

        logger.info(f"Successfully computed {len(sample_weights)} sample weights")
        logger.debug(f"Weight range: [{min(sample_weights):.4f}, {max(sample_weights):.4f}]")

        return sample_weights

    except Exception as e:
        logger.error(f"Failed to compute sample weights: {e}")
        raise


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

    logger.info("Starting post-processing with equalized odds constraint")
    logger.debug(f"Input data shape: {X.shape}, protected groups: {len(set(protected))}")

    try:
        # Verify model has required methods
        if not hasattr(model, 'predict_proba'):
            logger.warning("Model does not have predict_proba method, post-processing may fail")

        optimizer = ThresholdOptimizer(
            estimator=model,
            constraints="equalized_odds",
            predict_method="predict_proba",
            prefit=True,
        )
        logger.debug("Created ThresholdOptimizer with equalized_odds constraint")

        optimizer.fit(X, y, sensitive_features=protected)
        logger.info("Successfully completed threshold optimization for equalized odds")

        return optimizer

    except Exception as e:
        logger.error(f"Failed to optimize thresholds for equalized odds: {e}")
        raise
