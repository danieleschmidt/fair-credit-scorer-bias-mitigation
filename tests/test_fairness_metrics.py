import numpy as np
from src.fairness_metrics import compute_fairness_metrics


def test_fairness_metrics_balanced():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    protected = np.array([0, 1, 0, 1])

    overall, by_group = compute_fairness_metrics(y_true, y_pred, protected, y_scores=y_pred)

    # When predictions equal labels, equalized_odds_difference should be 0
    assert abs(overall['equalized_odds_difference']) < 1e-6
    # Demographic parity difference should also be 0
    assert abs(overall['demographic_parity_difference']) < 1e-6
    # False positive/negative rate differences should be 0
    assert abs(overall['false_positive_rate_difference']) < 1e-6
    assert abs(overall['false_negative_rate_difference']) < 1e-6
    assert abs(overall['false_positive_rate']) < 1e-6
    assert abs(overall['false_negative_rate']) < 1e-6
    assert abs(overall['accuracy_difference']) < 1e-6
    assert abs(overall['balanced_accuracy_difference']) < 1e-6
    assert abs(overall['precision_difference']) < 1e-6
    assert abs(overall['recall_difference']) < 1e-6
    assert abs(overall['f1_difference']) < 1e-6
    assert abs(overall['log_loss_difference']) < 1e-6
    assert abs(overall['false_discovery_rate_difference']) < 1e-6
    assert abs(overall['roc_auc_difference']) < 1e-6
    assert abs(overall['true_positive_rate_difference']) < 1e-6
    assert abs(overall['true_negative_rate_difference']) < 1e-6
    assert abs(overall['demographic_parity_ratio'] - 1.0) < 1e-6
    assert abs(overall['equalized_odds_ratio'] - 1.0) < 1e-6
    assert abs(overall['false_positive_rate_ratio'] - 1.0) < 1e-6
    assert abs(overall['false_negative_rate_ratio'] - 1.0) < 1e-6
    assert abs(overall['true_positive_rate_ratio'] - 1.0) < 1e-6
    assert abs(overall['true_negative_rate_ratio'] - 1.0) < 1e-6
    assert abs(overall['accuracy_ratio'] - 1.0) < 1e-6
    # Selection rates should be equal across groups
    assert by_group['selection_rate'][0] == by_group['selection_rate'][1]
    assert abs(by_group['false_positive_rate'][0] - by_group['false_positive_rate'][1]) < 1e-6
    assert abs(by_group['false_negative_rate'][0] - by_group['false_negative_rate'][1]) < 1e-6
    assert abs(by_group['false_discovery_rate'][0] - by_group['false_discovery_rate'][1]) < 1e-6
