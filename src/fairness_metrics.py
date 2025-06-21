import pandas as pd
from fairlearn.metrics import MetricFrame, selection_rate, equalized_odds_difference


def compute_fairness_metrics(y_true, y_pred, protected):
    """Compute basic fairness metrics using fairlearn."""
    if not isinstance(protected, pd.Series):
        protected = pd.Series(protected, name="protected")

    metrics = {"selection_rate": selection_rate}
    frame = MetricFrame(metrics=metrics, y_true=y_true, y_pred=y_pred, sensitive_features=protected)

    eod = equalized_odds_difference(y_true, y_pred, sensitive_features=protected)
    overall = frame.overall
    overall["equalized_odds_difference"] = eod

    return overall, frame.by_group
