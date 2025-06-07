import sys
import os

# Ensure the src directory is in the Python path
# This allows for imports like 'from src.module import ...' when running from root
# Or for imports like 'from module import ...' if this script is run from src/
# For consistency with the problem statement's import style, we ensure src's parent is on path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader_preprocessor import load_and_preprocess_credit_data
from src.baseline_model import train_baseline_classifier
from src.fairness_metrics import (
    calculate_demographic_parity_difference,
    calculate_equalized_odds_difference
)
# Import bias mitigation techniques
# apply_reweighing and apply_exponentiated_gradient are placeholders due to prior import issues.
# apply_threshold_optimizer is expected to work.
from src.bias_mitigator import apply_reweighing, apply_exponentiated_gradient, apply_threshold_optimizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score # For post-processed accuracy
import pandas as pd

# Define constants for data loading and processing
FILE_PATH = 'data/credit_data.csv'
PROTECTED_ATTRIBUTE_COL = 'gender' # Example: 'gender' or 'race'
TARGET_COL = 'credit_risk' # Name of the target variable column

def main():
    """
    Main function to run the fairness evaluation pipeline.
    """
    print("Starting fairness evaluation pipeline...")

    # 1. Load and preprocess data
    print(f"\nLoading and preprocessing data from: {FILE_PATH}")
    X, y, sensitive_features = load_and_preprocess_credit_data(
        FILE_PATH, PROTECTED_ATTRIBUTE_COL, TARGET_COL
    )

    if X is None or y is None or sensitive_features is None:
        print("Data loading and preprocessing failed. Exiting.")
        return

    print(f"Data loaded. X shape: {X.shape}, y shape: {y.shape}, Sensitive features unique values: {sensitive_features.unique()}")

    # 2. Split data into training and testing sets
    # Stratify by y to ensure similar class distribution in train and test sets
    # Ensure sensitive_features are split along with X and y
    print("\nSplitting data into training and testing sets...")
    X_train, X_test, y_train, y_test, sensitive_features_train, sensitive_features_test = train_test_split(
        X, y, sensitive_features, test_size=0.3, random_state=42, stratify=y
    )
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
    print(f"Sensitive features train shape: {sensitive_features_train.shape}, Sensitive features test shape: {sensitive_features_test.shape}")

    # 3. Train the baseline model
    print("\nTraining the baseline Logistic Regression model...")
    model, accuracy = train_baseline_classifier(X_train, y_train, X_test, y_test)
    print(f"Baseline model trained. Accuracy: {accuracy:.4f}")

    # 4. Get predictions from the trained model on the test set
    print("\nMaking predictions on the test set...")
    y_pred_test = model.predict(X_test)

    # 5. Calculate fairness metrics
    print("\nCalculating fairness metrics...")

    # Ensure sensitive_features_test is a 1D array or Series as expected by fairlearn
    if isinstance(sensitive_features_test, pd.DataFrame):
        sensitive_features_test = sensitive_features_test.squeeze() # Convert to Series if it's a single-column DataFrame

    demographic_parity_diff = calculate_demographic_parity_difference(
        y_test, y_pred_test, sensitive_features=sensitive_features_test
    )
    equalized_odds_diff = calculate_equalized_odds_difference(
        y_test, y_pred_test, sensitive_features=sensitive_features_test
    )

    # 6. Print Baseline Results
    print("\n--- Baseline Model ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Demographic Parity Difference: {demographic_parity_diff:.4f}")
    print(f"Equalized Odds Difference: {equalized_odds_diff:.4f}")

    # 7. Apply ThresholdOptimizer
    print("\nApplying ThresholdOptimizer (constraint: demographic_parity)...")
    # baseline_model is the trained Logistic Regression estimator
    y_pred_postprocessed = apply_threshold_optimizer(
        baseline_model,  # Corrected: pass the model object itself
        X_train,
        y_train,
        sensitive_features_train,
        X_test,
        sensitive_features_test,
        constraint='demographic_parity'
    )

    print("\n--- ThresholdOptimizer (constraint: demographic_parity) ---")
    if y_pred_postprocessed is not None:
        accuracy_postprocessed = accuracy_score(y_test, y_pred_postprocessed)
        dpd_postprocessed = calculate_demographic_parity_difference(
            y_test, y_pred_postprocessed, sensitive_features=sensitive_features_test
        )
        eod_postprocessed = calculate_equalized_odds_difference(
            y_test, y_pred_postprocessed, sensitive_features=sensitive_features_test
        )
        print(f"Accuracy: {accuracy_postprocessed:.4f}")
        print(f"Demographic Parity Difference: {dpd_postprocessed:.4f}")
        print(f"Equalized Odds Difference: {eod_postprocessed:.4f}")
    else:
        print("ThresholdOptimizer application (demographic_parity) failed or was skipped (check bias_mitigator.py for import/execution errors).")

    # 8. Apply ThresholdOptimizer with Equalized Odds constraint
    print("\nApplying ThresholdOptimizer (constraint: equalized_odds)...")
    y_pred_eq_odds = apply_threshold_optimizer(
        model, # Pass the trained baseline model
        X_train,
        y_train,
        sensitive_features_train,
        X_test,
        sensitive_features_test,
        constraint='equalized_odds'
    )

    print("\n--- ThresholdOptimizer (constraint: equalized_odds) ---")
    if y_pred_eq_odds is not None:
        accuracy_eq_odds = accuracy_score(y_test, y_pred_eq_odds)
        dpd_eq_odds = calculate_demographic_parity_difference(
            y_test, y_pred_eq_odds, sensitive_features=sensitive_features_test
        )
        eod_eq_odds = calculate_equalized_odds_difference(
            y_test, y_pred_eq_odds, sensitive_features=sensitive_features_test
        )
        print(f"Accuracy: {accuracy_eq_odds:.4f}")
        print(f"Demographic Parity Difference: {dpd_eq_odds:.4f}")
        print(f"Equalized Odds Difference: {eod_eq_odds:.4f}")
    else:
        print("ThresholdOptimizer application (equalized_odds) failed or was skipped (check bias_mitigator.py for import/execution errors).")

    print("\n\n--------------------------------------------------------------------")
    print("NOTE: `fairlearn.postprocessing.ThresholdOptimizer` was attempted with different constraints.")
    print("      Please check its output above. Other fairness mitigation techniques")
    print("      (Reweighing from `fairlearn.preprocessing` and ExponentiatedGradient")
    print("      from `fairlearn.reductions`) were planned but could not be fully")
    print("      implemented or evaluated due to persistent library import issues")
    print("      encountered with those specific modules in this environment.")
    print("--------------------------------------------------------------------")

    print("\nPipeline finished.")

if __name__ == '__main__':
    main()
