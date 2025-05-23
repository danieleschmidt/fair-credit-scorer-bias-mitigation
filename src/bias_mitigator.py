# src/bias_mitigator.py
import pandas as pd
import numpy as np

# FAIRNESS MITIGATION TECHNIQUES (Reweighing, ExponentiatedGradient) WERE NOT IMPLEMENTED
# DUE TO PERSISTENT LIBRARY IMPORT ISSUES WITH `fairlearn`.
# Specifically, `fairlearn.preprocessing.Reweighing` and
# `fairlearn.reductions.ExponentiatedGradient` could not be imported
# despite multiple attempts and troubleshooting.

# The following are placeholder functions for those.

def apply_reweighing(X_train, y_train, sensitive_features_train):
    print("INFO: `apply_reweighing` was not implemented due to fairlearn import issues.")
    return None

def apply_exponentiated_gradient(X_train, y_train, sensitive_features_train):
    print("INFO: `apply_exponentiated_gradient` was not implemented due to fairlearn import issues.")
    return None

# --- New Imports for ThresholdOptimizer ---
ThresholdOptimizer = None
ThresholdOptimizerImportError = None
# Imports for the test block also need to be handled carefully
LogisticRegression = None
train_test_split = None
SklearnImportError = None

try:
    from fairlearn.postprocessing import ThresholdOptimizer
    # print("Successfully imported ThresholdOptimizer from fairlearn.postprocessing.")
except ImportError as e:
    ThresholdOptimizerImportError = e
except Exception as e_gen:
    ThresholdOptimizerImportError = e_gen

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    # print("Successfully imported sklearn components for ThresholdOptimizer testing.")
except ImportError as e: 
    SklearnImportError = e
except Exception as e_gen:
    SklearnImportError = e_gen


def apply_threshold_optimizer(estimator, X_train, y_train, sensitive_features_train, X_test, sensitive_features_test, constraint='demographic_parity'):
    """
    Applies the ThresholdOptimizer post-processing technique to adjust predictions for fairness.

    Args:
        estimator: The pre-trained estimator/model. Must have predict_proba method.
        X_train (pd.DataFrame or np.ndarray): Training features.
        y_train (pd.Series or np.ndarray): Training target variable.
        sensitive_features_train (pd.Series or np.ndarray): Sensitive features for training data (1D array-like).
        X_test (pd.DataFrame or np.ndarray): Test features.
        sensitive_features_test (pd.Series or np.ndarray): Sensitive features for test data (1D array-like).
        constraint (str, optional): The fairness constraint to apply. 
                                    Defaults to 'demographic_parity'. 
                                    Other options include 'equalized_odds', etc.

    Returns:
        np.ndarray: Post-processed predictions for X_test.
                    Returns None if ThresholdOptimizer could not be imported or an error occurs.
    """
    if ThresholdOptimizer is None:
        print(f"ThresholdOptimizer class not imported successfully. Cannot apply. "
              f"Import error: {ThresholdOptimizerImportError}")
        return None

    # Ensure sensitive features are 1D array-like
    def process_sensitive_features(sf_features, sf_name_for_error):
        if isinstance(sf_features, pd.DataFrame):
            if sf_features.shape[1] == 1:
                return sf_features.iloc[:, 0]
            else:
                print(f"Error: {sf_name_for_error} for ThresholdOptimizer is a multi-column DataFrame. Please provide a 1D array or Series.")
                return None
        elif isinstance(sf_features, np.ndarray) and sf_features.ndim == 2 and sf_features.shape[1] == 1:
            return sf_features.squeeze()
        elif isinstance(sf_features, np.ndarray) and sf_features.ndim > 1: # Catches 2D with more than 1 col, or >2D
            print(f"Error: {sf_name_for_error} for ThresholdOptimizer is a multi-dimensional/multi-column NumPy array. Please provide a 1D array.")
            return None
        return sf_features # Assumed to be pd.Series or 1D np.ndarray

    sf_train_processed = process_sensitive_features(sensitive_features_train, "sensitive_features_train")
    if sf_train_processed is None: return None
    sf_test_processed = process_sensitive_features(sensitive_features_test, "sensitive_features_test")
    if sf_test_processed is None: return None
    
    try:
        threshold_optimizer = ThresholdOptimizer(
            estimator=estimator,
            constraints=constraint,
            objective='accuracy_score', 
            prefit=True,
            predict_method='predict_proba'
        )

        threshold_optimizer.fit(X_train, y_train, sensitive_features=sf_train_processed)
        y_pred_postprocessed = threshold_optimizer.predict(X_test, sensitive_features=sf_test_processed)
        
        return y_pred_postprocessed
    except Exception as e:
        print(f"Error during ThresholdOptimizer application: {e}")
        # For more detailed debugging:
        # import traceback
        # print(traceback.format_exc())
        return None


if __name__ == '__main__':
    print("This script (`src/bias_mitigator.py`) contains placeholders for some fairness mitigation functions")
    print("due to `fairlearn` import issues for Reweighing/ExponentiatedGradient.")
    print("It now also includes `apply_threshold_optimizer`.\n")

    # Test apply_threshold_optimizer
    print("\n--- Testing apply_threshold_optimizer ---")
    if ThresholdOptimizer is None:
        print(f"Cannot run ThresholdOptimizer example due to ThresholdOptimizer import failure: {ThresholdOptimizerImportError}")
    elif LogisticRegression is None or train_test_split is None:
        print(f"Cannot run ThresholdOptimizer example due to scikit-learn import failure(s): {SklearnImportError}")
    else:
        print(f"Using ThresholdOptimizer class: {ThresholdOptimizer}")
        num_samples_to = 200
        X_to = pd.DataFrame({
            'featureA': np.random.rand(num_samples_to) * 10, # Ensure X has at least two features
            'featureB': np.random.rand(num_samples_to) * 5,
            'featureC': np.random.choice([0,1,2,3], num_samples_to) 
        })
        y_to = pd.Series((X_to['featureA'] + X_to['featureB'] * 0.75 + X_to['featureC'] > 8.0).astype(int), name="target")
        sensitive_features_to = pd.Series(np.random.choice(['GroupX', 'GroupY', 'GroupZ'], num_samples_to), name="SensitiveGroup")

        print(f"Sample data shapes: X={X_to.shape}, y={y_to.shape}, SF={sensitive_features_to.shape}")

        X_train_to, X_test_to, y_train_to, y_test_to, sf_train_to, sf_test_to = train_test_split(
            X_to, y_to, sensitive_features_to, test_size=0.4, random_state=42, stratify=y_to
        )
        
        print("Training baseline LogisticRegression model for ThresholdOptimizer...")
        baseline_model_to = LogisticRegression(solver='liblinear', random_state=42) # Using liblinear for small datasets
        baseline_model_to.fit(X_train_to, y_train_to)
        
        y_pred_baseline_to = baseline_model_to.predict(X_test_to)
        accuracy_baseline = np.mean(y_pred_baseline_to == y_test_to) # Manual accuracy calculation
        print(f"Baseline model accuracy on test set: {accuracy_baseline:.4f}")

        print("Applying ThresholdOptimizer...")
        y_pred_postprocessed_to = apply_threshold_optimizer(
            estimator=baseline_model_to, # Pass the trained model
            X_train=X_train_to,
            y_train=y_train_to,
            sensitive_features_train=sf_train_to,
            X_test=X_test_to,
            sensitive_features_test=sf_test_to,
            constraint='demographic_parity' # Example constraint
        )

        if y_pred_postprocessed_to is not None:
            print(f"Post-processed predictions (first 10 of {len(y_pred_postprocessed_to)}): {y_pred_postprocessed_to[:10]}")
            accuracy_postprocessed = np.mean(y_pred_postprocessed_to == y_test_to) # Manual accuracy
            print(f"Post-processed model accuracy on test set: {accuracy_postprocessed:.4f}")
            # Compare distribution of predictions
            print(f"Baseline predictions distribution on test set: \n{pd.Series(y_pred_baseline_to).value_counts(normalize=True)}")
            print(f"Post-processed predictions distribution on test set: \n{pd.Series(y_pred_postprocessed_to).value_counts(normalize=True)}")
        else:
            print("ThresholdOptimizer application failed (see error messages above).")

    print("\n--- End of bias_mitigator.py tests ---")
```
