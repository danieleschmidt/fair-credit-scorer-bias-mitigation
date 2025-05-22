import numpy as np
import pandas as pd # Useful for structuring sensitive features in tests
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference
)

def calculate_demographic_parity_difference(y_true, y_pred, sensitive_features):
    """
    Calculates the Demographic Parity Difference.

    Demographic Parity Difference is the difference between the largest and smallest group-level
    selection rates (e.g., proportion of positive predictions) across sensitive feature groups.
    A value of 0 indicates perfect demographic parity.

    Args:
        y_true (array-like): Ground truth (correct) labels.
        y_pred (array-like): Predicted labels, as returned by a classifier.
        sensitive_features (array-like): Sensitive features for grouping (e.g., gender, race).
                                         This should be a 1D array or Series.

    Returns:
        float: The Demographic Parity Difference.
    """
    dpd = demographic_parity_difference(
        y_true,
        y_pred,
        sensitive_features=sensitive_features
    )
    return dpd

def calculate_equalized_odds_difference(y_true, y_pred, sensitive_features):
    """
    Calculates the Equalized Odds Difference.

    Equalized Odds Difference is the maximum difference between the largest and smallest
    group-level true positive rates (TPR) and the largest and smallest group-level
    false positive rates (FPR) across sensitive feature groups.
    A value of 0 indicates perfect equalized odds.

    Args:
        y_true (array-like): Ground truth (correct) labels.
        y_pred (array-like): Predicted labels, as returned by a classifier.
        sensitive_features (array-like): Sensitive features for grouping (e.g., gender, race).
                                         This should be a 1D array or Series.

    Returns:
        float: The Equalized Odds Difference.
    """
    eod = equalized_odds_difference(
        y_true,
        y_pred,
        sensitive_features=sensitive_features
    )
    return eod

if __name__ == '__main__':
    print("Running fairness metrics script example...")

    # Dummy data for demonstration
    # y_true: True labels
    y_true_dummy = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    # y_pred: Predicted labels from a hypothetical model
    y_pred_dummy = np.array([0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1])
    # sensitive_features: Two groups, 'A' and 'B'
    sensitive_features_dummy = np.array(['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 
                                         'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'])
    
    # Using pandas Series for sensitive features can sometimes be more robust with fairlearn
    # sensitive_features_dummy_pd = pd.Series(sensitive_features_dummy)


    print(f"\nDummy Data:")
    print(f"y_true:             {y_true_dummy}")
    print(f"y_pred:             {y_pred_dummy}")
    print(f"sensitive_features: {sensitive_features_dummy}")

    # Calculate Demographic Parity Difference
    dpd_value = calculate_demographic_parity_difference(
        y_true_dummy, y_pred_dummy, sensitive_features_dummy
    )
    print(f"\nDemographic Parity Difference: {dpd_value:.4f}")

    # Calculate Equalized Odds Difference
    eod_value = calculate_equalized_odds_difference(
        y_true_dummy, y_pred_dummy, sensitive_features_dummy
    )
    print(f"Equalized Odds Difference: {eod_value:.4f}")

    # Another example with more skewed predictions/groups to see non-zero values
    print("\n--- Another Example (Potentially More Imbalanced) ---")
    y_true_ex2 = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
    y_pred_ex2 = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1]) # Model predicts 1 more for group B
    sensitive_features_ex2 = np.array(['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 
                                       'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B'])
    
    print(f"y_true_ex2:             {y_true_ex2}")
    print(f"y_pred_ex2:             {y_pred_ex2}")
    print(f"sensitive_features_ex2: {sensitive_features_ex2}")

    dpd_ex2 = calculate_demographic_parity_difference(
        y_true_ex2, y_pred_ex2, sensitive_features_ex2
    )
    print(f"Demographic Parity Difference (Ex2): {dpd_ex2:.4f}")

    eod_ex2 = calculate_equalized_odds_difference(
        y_true_ex2, y_pred_ex2, sensitive_features_ex2
    )
    print(f"Equalized Odds Difference (Ex2): {eod_ex2:.4f}")

    # Example where one group has no positive predictions
    print("\n--- Example: One group with no positive predictions ---")
    y_true_ex3 = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    y_pred_ex3 = np.array([1, 1, 0, 0, 0, 0, 0, 0]) # Group B gets no positive predictions
    sensitive_features_ex3 = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])

    print(f"y_true_ex3:             {y_true_ex3}")
    print(f"y_pred_ex3:             {y_pred_ex3}")
    print(f"sensitive_features_ex3: {sensitive_features_ex3}")
    
    dpd_ex3 = calculate_demographic_parity_difference(
        y_true_ex3, y_pred_ex3, sensitive_features_ex3
    )
    print(f"Demographic Parity Difference (Ex3): {dpd_ex3:.4f}")

    eod_ex3 = calculate_equalized_odds_difference(
        y_true_ex3, y_pred_ex3, sensitive_features_ex3
    )
    print(f"Equalized Odds Difference (Ex3): {eod_ex3:.4f}")
    
    # Example with perfect fairness for DPD
    print("\n--- Example: Perfect Demographic Parity ---")
    # Selection rate for group A: 2/4 = 0.5
    # Selection rate for group B: 2/4 = 0.5
    # DPD = 0
    y_true_ex4 = np.array([1, 1, 0, 0, 1, 1, 0, 0]) # y_true doesn't directly affect DPD, but good for EOD
    y_pred_ex4 = np.array([1, 1, 0, 0, 1, 1, 0, 0]) 
    sensitive_features_ex4 = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])
    
    dpd_ex4 = calculate_demographic_parity_difference(y_true_ex4, y_pred_ex4, sensitive_features_ex4)
    print(f"Demographic Parity Difference (Ex4 - Perfect DPD): {dpd_ex4:.4f}")
    eod_ex4 = calculate_equalized_odds_difference(y_true_ex4, y_pred_ex4, sensitive_features_ex4)
    print(f"Equalized Odds Difference (Ex4 - Perfect DPD): {eod_ex4:.4f}")
