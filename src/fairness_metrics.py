"""Fairness metrics computation for machine learning models.

This module provides comprehensive fairness metrics using the fairlearn library,
including demographic parity, equalized odds, and various performance metrics
across different protected groups.

Key metrics computed:
- Demographic parity difference/ratio: Measures selection rate equality
- Equalized odds difference/ratio: Measures equal TPR and FPR across groups
- False positive/negative rate differences: Detailed error rate analysis
- Performance metrics by protected group (accuracy, precision, recall, F1)
- ROC AUC and log loss with fairness considerations

Functions:
    compute_fairness_metrics: Main function to compute all fairness metrics

Example:
    >>> from fairness_metrics import compute_fairness_metrics
    >>> overall, by_group = compute_fairness_metrics(y_true, y_pred, protected)
    >>> print(f"Demographic parity difference: {overall['demographic_parity_difference']}")
    >>> print(f"Accuracy by group: {by_group['accuracy']}")

The module integrates with the fairlearn library to provide standardized fairness
assessments and supports both binary and probability predictions for comprehensive
bias evaluation in machine learning systems.
"""

from typing import Tuple

import pandas as pd
import numpy as np
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    equalized_odds_difference,
    demographic_parity_difference,
    false_positive_rate_difference,
    false_negative_rate_difference,
    false_positive_rate,
    false_negative_rate,
    true_positive_rate,
    true_negative_rate,
    demographic_parity_ratio,
    equalized_odds_ratio,
    false_positive_rate_ratio,
    false_negative_rate_ratio,
    true_positive_rate_ratio,
    true_negative_rate_ratio,
    accuracy_score_ratio,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
)

try:
    from .logging_config import get_logger
except ImportError:
    from logging_config import get_logger

logger = get_logger(__name__)


def compute_fairness_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    protected: pd.Series | np.ndarray,
    y_scores: pd.Series | np.ndarray | None = None,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Compute basic fairness metrics using fairlearn.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    protected : array-like or pandas.Series
        Protected attribute for each sample.
    y_scores : array-like or None, optional
        Scores or probabilities used for metrics like ROC AUC. If ``None``,
        ``y_pred`` will be used instead.

    Notes
    -----
    Includes metrics such as selection rate, precision/recall, log loss,
    ROC AUC, false discovery rate, and their differences across
    protected groups.
    """
    logger.info("Computing comprehensive fairness metrics")
    
    try:
        logger.debug(f"Input data - labels: {len(y_true)}, predictions: {len(y_pred)}, protected groups: {len(set(protected))}")
        
        if not isinstance(protected, pd.Series):
            protected = pd.Series(protected, name="protected")
        
        metrics = {
            "selection_rate": selection_rate,
            "accuracy": accuracy_score,
            "balanced_accuracy": balanced_accuracy_score,
            "precision": precision_score,
            "recall": recall_score,
            "f1": f1_score,
            "true_positive_rate": true_positive_rate,
            "true_negative_rate": true_negative_rate,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
        }
        
        # If scores are provided, compute ROC AUC and log loss using them
        roc_input = y_scores if y_scores is not None else y_pred
        roc_auc_frame = MetricFrame(
            metrics={"roc_auc": roc_auc_score},
            y_true=y_true,
            y_pred=roc_input,
            sensitive_features=protected,
        )
        log_input = y_scores if y_scores is not None else y_pred
        logloss_frame = MetricFrame(
            metrics={"log_loss": lambda y_true, y_pred: log_loss(y_true, y_pred)},
            y_true=y_true,
            y_pred=log_input,
            sensitive_features=protected,
        )
        frame = MetricFrame(
            metrics=metrics,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=protected,
        )
        
        eod = equalized_odds_difference(y_true, y_pred, sensitive_features=protected)
        dpd = demographic_parity_difference(y_true, y_pred, sensitive_features=protected)
        fpr_diff = false_positive_rate_difference(
            y_true, y_pred, sensitive_features=protected
        )
        fnr_diff = false_negative_rate_difference(
            y_true, y_pred, sensitive_features=protected
        )
        dp_ratio = demographic_parity_ratio(y_true, y_pred, sensitive_features=protected)
        eod_ratio = equalized_odds_ratio(y_true, y_pred, sensitive_features=protected)
        fpr_ratio = np.nan_to_num(
            false_positive_rate_ratio(y_true, y_pred, sensitive_features=protected), nan=1.0
        )
        fnr_ratio = np.nan_to_num(
            false_negative_rate_ratio(y_true, y_pred, sensitive_features=protected), nan=1.0
        )
        tpr_ratio = np.nan_to_num(
            true_positive_rate_ratio(y_true, y_pred, sensitive_features=protected), nan=1.0
        )
        tnr_ratio = np.nan_to_num(
            true_negative_rate_ratio(y_true, y_pred, sensitive_features=protected), nan=1.0
        )
        acc_ratio = np.nan_to_num(
            accuracy_score_ratio(y_true, y_pred, sensitive_features=protected), nan=1.0
        )
        
        overall = frame.overall
        overall["equalized_odds_difference"] = eod
        overall["demographic_parity_difference"] = dpd
        overall["false_positive_rate_difference"] = fpr_diff
        overall["false_negative_rate_difference"] = fnr_diff
        overall["demographic_parity_ratio"] = dp_ratio
        overall["equalized_odds_ratio"] = eod_ratio
        overall["false_positive_rate_ratio"] = fpr_ratio
        overall["false_negative_rate_ratio"] = fnr_ratio
        overall["true_positive_rate_ratio"] = tpr_ratio
        overall["true_negative_rate_ratio"] = tnr_ratio
        overall["accuracy_ratio"] = acc_ratio
        diffs = frame.difference()
        overall["accuracy_difference"] = diffs["accuracy"]
        overall["balanced_accuracy_difference"] = diffs["balanced_accuracy"]
        overall["precision_difference"] = diffs["precision"]
        overall["recall_difference"] = diffs["recall"]
        overall["f1_difference"] = diffs["f1"]
        overall["log_loss"] = logloss_frame.overall["log_loss"]
        overall["log_loss_difference"] = logloss_frame.difference()["log_loss"]
        overall["false_discovery_rate"] = 1 - overall["precision"]
        overall["false_discovery_rate_difference"] = -diffs["precision"]
        overall["roc_auc"] = roc_auc_frame.overall["roc_auc"]
        overall["roc_auc_difference"] = roc_auc_frame.difference()["roc_auc"]
        overall["true_positive_rate_difference"] = diffs["true_positive_rate"]
        overall["true_negative_rate_difference"] = diffs["true_negative_rate"]
        
        by_group = frame.by_group
        by_group["log_loss"] = logloss_frame.by_group["log_loss"]
        by_group["roc_auc"] = roc_auc_frame.by_group["roc_auc"]
        by_group["false_discovery_rate"] = 1 - by_group["precision"]
        
        logger.info("Successfully computed fairness metrics")
        logger.debug(f"Overall demographic parity difference: {overall['demographic_parity_difference']:.4f}")
        logger.debug(f"Overall equalized odds difference: {overall['equalized_odds_difference']:.4f}")
        
        return overall, by_group
        
    except Exception as e:
        logger.error(f"Failed to compute fairness metrics: {e}")
        raise
