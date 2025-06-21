import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    equalized_odds_difference,
    demographic_parity_difference,
    false_positive_rate_difference,
    false_negative_rate_difference,
)
from sklearn.metrics import accuracy_score


def compute_fairness_metrics(y_true, y_pred, protected):
    """Compute basic fairness metrics using fairlearn."""
    if not isinstance(protected, pd.Series):
        protected = pd.Series(protected, name="protected")

    metrics = {"selection_rate": selection_rate, "accuracy": accuracy_score}
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

    overall = frame.overall
    overall["equalized_odds_difference"] = eod
    overall["demographic_parity_difference"] = dpd
    overall["false_positive_rate_difference"] = fpr_diff
    overall["false_negative_rate_difference"] = fnr_diff
    overall["accuracy_difference"] = frame.difference()["accuracy"]

    return overall, frame.by_group
