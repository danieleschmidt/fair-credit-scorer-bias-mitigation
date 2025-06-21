import numpy as np
from src.fairness_metrics import compute_fairness_metrics


def test_fairness_metrics_balanced():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    protected = np.array([0, 1, 0, 1])

    overall, by_group = compute_fairness_metrics(y_true, y_pred, protected)

    # When predictions equal labels, equalized_odds_difference should be 0
    assert abs(overall['equalized_odds_difference']) < 1e-6
    # Selection rates should be equal across groups
    assert by_group['selection_rate'][0] == by_group['selection_rate'][1]
