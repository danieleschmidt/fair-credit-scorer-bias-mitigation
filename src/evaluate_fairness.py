"""Main evaluation pipeline for fairness-aware credit scoring models.

This module provides the primary entry point for training and evaluating
credit scoring models with various bias mitigation techniques. It supports
both single train/test splits and cross-validation evaluation workflows.

Evaluation methods supported:
- baseline: Standard logistic regression without bias mitigation
- reweight: Sample reweighting for demographic parity
- postprocess: Post-processing with equalized odds optimization  
- expgrad: Exponentiated gradient optimization for demographic parity

The module can be used both programmatically and via command-line interface,
providing comprehensive fairness metrics and JSON output for automated pipelines.

Functions:
    run_pipeline: Execute single evaluation with train/test split
    run_cross_validation: Execute k-fold cross-validation evaluation

Command-line interface:
    python -m evaluate_fairness --method baseline --cv 5 --output-json results.json

Programmatic usage:
    >>> from evaluate_fairness import run_pipeline, run_cross_validation
    >>> # Single evaluation
    >>> results = run_pipeline(method="reweight", test_size=0.3)
    >>> 
    >>> # Cross-validation evaluation
    >>> cv_results = run_cross_validation(method="expgrad", cv=5)

Configuration:
    Evaluation parameters can be configured through:
    - Command-line arguments (highest priority)
    - Environment variables (FAIRNESS_* prefix)
    - Configuration file (config/default.yaml)

The module automatically handles data loading, model training, bias mitigation
application, and comprehensive fairness evaluation with detailed reporting.
"""

import argparse
import logging

# Import version directly to avoid circular import
__version__ = "0.2.0"

from data_loader_preprocessor import load_credit_data, load_credit_dataset
from baseline_model import train_baseline_model, evaluate_model
from bias_mitigator import (
    reweight_samples,
    postprocess_equalized_odds,
    expgrad_demographic_parity,
)
from fairness_metrics import compute_fairness_metrics
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import json

logger = logging.getLogger(__name__)

def _serialize_metrics(results):
    """Convert metrics dictionary with pandas objects to JSON-friendly types."""
    data = {
        "accuracy": float(results["accuracy"]),
        "overall": {k: float(v) for k, v in results["overall"].to_dict().items()},
        "by_group": {
            str(k): {m: float(v) for m, v in row.items()}
            for k, row in results["by_group"].to_dict(orient="index").items()
        },
    }
    if "overall_std" in results:
        data["overall_std"] = {
            k: float(v) for k, v in results["overall_std"].to_dict().items()
        }
    if "by_group_std" in results:
        data["by_group_std"] = {
            str(k): {m: float(v) for m, v in row.items()}
            for k, row in results["by_group_std"].to_dict(orient="index").items()
        }
    if "folds" in results:
        data["folds"] = [
            {
                "accuracy": float(fold["accuracy"]),
                "overall": {k: float(v) for k, v in fold["overall"].to_dict().items()},
                "by_group": {
                    str(k): {m: float(v) for m, v in row.items()}
                    for k, row in fold["by_group"].to_dict(orient="index").items()
                },
            }
            for fold in results["folds"]
        ]
    return data


def _save_metrics_json(results, path):
    """Write metrics dictionary to ``path`` as JSON.
    
    Parameters
    ----------
    results : dict
        Metrics dictionary to serialize
    path : str
        Output file path
        
    Raises
    ------
    ValueError
        If path is empty or results is invalid
    PermissionError
        If lacking write permissions
    OSError
        If file cannot be created
    """
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path must be a non-empty string")
    if not isinstance(results, dict):
        raise ValueError("results must be a dictionary")
        
    try:
        # Create directory if needed
        import os
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
            
        with open(path, "w") as f:
            json.dump(_serialize_metrics(results), f, indent=2)
        logger.info(f"Successfully saved metrics to {path}")
    except PermissionError:
        raise PermissionError(f"Permission denied writing to {path}")
    except OSError as e:
        raise OSError(f"Could not write to {path}: {e}")
    except (TypeError, ValueError) as e:
        raise TypeError(f"Cannot serialize metrics to JSON: {e}")


def _validate_common_parameters(method, threshold=None, output_path=None):
    """Validate common parameters used by both run_pipeline and run_cross_validation.
    
    Parameters
    ----------
    method : str
        Training approach name to validate
    threshold : float or None, optional  
        Custom decision threshold to validate
    output_path : str or None, optional
        Output file path to validate
        
    Raises
    ------
    ValueError
        If method is not supported or threshold is invalid
    TypeError
        If parameters have wrong types
    """
    # Method validation
    valid_methods = {"baseline", "reweight", "postprocess", "expgrad"}
    if not isinstance(method, str):
        raise TypeError(f"method must be a string, got {type(method).__name__}")
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got '{method}'")
    
    # Threshold validation
    if threshold is not None:
        if not isinstance(threshold, (int, float)):
            raise TypeError(f"threshold must be a number or None, got {type(threshold).__name__}")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be between 0.0 and 1.0, got {threshold}")
    
    # Output path validation
    if output_path is not None:
        if not isinstance(output_path, str):
            raise TypeError(f"output_path must be a string or None, got {type(output_path).__name__}")
        if not output_path.strip():
            raise ValueError("output_path cannot be empty or whitespace")


def run_pipeline(
    method="baseline",
    test_size=0.3,
    output_path=None,
    random_state=42,
    data_path="data/credit_data.csv",
    threshold=None,
    X=None,
    y=None,
    sensitive_features=None,
    verbose=True,
):
    """Train the model and return accuracy and fairness metrics.

    Parameters
    ----------
    method : str, optional
        Training approach: ``"baseline"`` (default), ``"reweight"``, or ``"postprocess"``.
    test_size : float, optional
        Portion of the data to reserve for testing, by default 0.3.
    output_path : str or None, optional
        If provided, the returned metrics dictionary will also be written to the
        given JSON file.
    threshold : float or None, optional
        Custom decision threshold applied to probability scores. The default
        ``None`` uses the model's built-in ``predict`` method.
    random_state : int, optional
        Random seed used when splitting the data, by default 42.
    data_path : str, optional
        Path to load data from, by default "data/credit_data.csv".
    X : array-like, optional
        Pre-loaded feature matrix. If provided, data_path is ignored.
    y : array-like, optional  
        Pre-loaded target vector. Required if X is provided.
    sensitive_features : array-like, optional
        Pre-loaded sensitive feature vector. Required if X is provided.
    verbose : bool, optional
        Whether to log detailed information, by default True.

    Returns
    -------
    dict
        Dictionary containing ``accuracy``, ``overall``, and ``by_group`` metrics.
        
    Raises
    ------
    ValueError
        If method is not supported or threshold is invalid
    TypeError
        If parameters have wrong types
    """
    # Input validation
    _validate_common_parameters(method, threshold, output_path)
    
    # Validate in-memory data parameters
    if X is not None:
        if y is None or sensitive_features is None:
            raise ValueError("When X is provided, y and sensitive_features must also be provided")
        if len(X) != len(y) or len(X) != len(sensitive_features):
            raise ValueError("X, y, and sensitive_features must have the same length")
            
    if verbose:
        logger.info(f"Running pipeline with method={method}, test_size={test_size}, threshold={threshold}")

    # Load or use provided data
    if X is not None:
        # Use in-memory data for benchmarking
        from sklearn.model_selection import train_test_split
        import pandas as pd
        
        # Convert to DataFrame format expected by the pipeline
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        if 'protected' not in X_df.columns:
            X_df['protected'] = sensitive_features
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        # Load from file
        X_train, X_test, y_train, y_test = load_credit_data(
            path=data_path, test_size=test_size, random_state=random_state
        )

    features_train = X_train.drop("protected", axis=1)
    features_test = X_test.drop("protected", axis=1)

    if method == "reweight":
        sample_weights = reweight_samples(y_train, X_train["protected"])
        model = train_baseline_model(
            features_train, y_train, sample_weight=sample_weights
        )
    else:
        model = train_baseline_model(features_train, y_train)
        if method == "postprocess":
            model = postprocess_equalized_odds(
                model,
                features_train,
                y_train,
                X_train["protected"],
            )
        elif method == "expgrad":
            model = expgrad_demographic_parity(
                features_train,
                y_train,
                X_train["protected"],
            )

    if method == "postprocess":
        accuracy, preds, probs = evaluate_model(
            model,
            features_test,
            y_test,
            sensitive_features=X_test["protected"],
            return_probs=True,
            threshold=threshold,
        )
    else:
        accuracy, preds, probs = evaluate_model(
            model,
            features_test,
            y_test,
            return_probs=True,
            threshold=threshold,
        )

    overall, by_group = compute_fairness_metrics(
        y_true=y_test,
        y_pred=preds,
        protected=X_test["protected"],
        y_scores=probs,
    )
    accuracy = overall["accuracy"]
    if verbose:
        logger.info("Accuracy: %.3f", accuracy)
        logger.info("Overall fairness metrics:\n%s", overall)
        logger.info("Metrics by group:\n%s", by_group)

    results = {"accuracy": accuracy, "overall": overall, "by_group": by_group}
    if output_path is not None:
        _save_metrics_json(results, output_path)

    return results


def run_cross_validation(
    method="baseline",
    cv=5,
    random_state=42,
    output_path=None,
    data_path="data/credit_data.csv",
    threshold=None,
):
    """Run cross-validated evaluation and return averaged metrics.

    Parameters
    ----------
    method : str, optional
        Training approach used in each fold.
    cv : int, optional
        Number of cross-validation splits. Must be at least 2.
    random_state : int, optional
        Seed controlling the cross-validation shuffle.
    output_path : str or None, optional
        If provided, write the averaged metrics to this JSON file.
    threshold : float or None, optional
        Custom probability threshold applied in every fold. ``None`` uses the
        estimator's default ``predict`` behaviour.
        
    Raises
    ------
    ValueError
        If cv is less than 2 or method is invalid
    TypeError
        If parameters have wrong types
    """
    # Input validation
    _validate_common_parameters(method, threshold, output_path)
    
    # CV-specific validation
    if not isinstance(cv, int):
        raise TypeError(f"cv must be an integer, got {type(cv).__name__}")
    if cv < 2:
        raise ValueError(f"cv must be at least 2, got {cv}")
            
    logger.info(f"Running {cv}-fold cross-validation with method={method}, threshold={threshold}")
    X, y = load_credit_dataset(path=data_path, random_state=random_state)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    overall_metrics = []
    by_group_metrics = []
    fold_results = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        features_train = X_train.drop("protected", axis=1)
        features_test = X_test.drop("protected", axis=1)

        if method == "reweight":
            weights = reweight_samples(y_train, X_train["protected"])
            model = train_baseline_model(features_train, y_train, sample_weight=weights)
        else:
            model = train_baseline_model(features_train, y_train)
            if method == "postprocess":
                model = postprocess_equalized_odds(
                    model,
                    features_train,
                    y_train,
                    X_train["protected"],
                )
            elif method == "expgrad":
                model = expgrad_demographic_parity(
                    features_train,
                    y_train,
                    X_train["protected"],
                )

        if method == "postprocess":
            _, preds, probs = evaluate_model(
                model,
                features_test,
                y_test,
                sensitive_features=X_test["protected"],
                return_probs=True,
                threshold=threshold,
            )
        else:
            _, preds, probs = evaluate_model(
                model,
                features_test,
                y_test,
                return_probs=True,
                threshold=threshold,
            )

        overall, by_group = compute_fairness_metrics(
            y_true=y_test,
            y_pred=preds,
            protected=X_test["protected"],
            y_scores=probs,
        )
        overall_metrics.append(overall)
        by_group_metrics.append(by_group)
        fold_results.append(
            {"accuracy": overall["accuracy"], "overall": overall, "by_group": by_group}
        )

    metrics_concat = pd.concat(overall_metrics, axis=1)
    mean_overall = metrics_concat.mean(axis=1)
    std_overall = metrics_concat.std(axis=1)
    by_group_concat = pd.concat(by_group_metrics)
    mean_by_group = by_group_concat.groupby(level=0).mean()
    std_by_group = by_group_concat.groupby(level=0).std()
    accuracy = float(mean_overall["accuracy"])
    logger.info("Average overall metrics:\n%s", mean_overall)
    logger.info(
        "Standard deviation of metrics across folds:\n%s",
        std_overall,
    )
    logger.info("Average metrics by group:\n%s", mean_by_group)
    logger.info("Standard deviation by group:\n%s", std_by_group)
    results = {
        "accuracy": accuracy,
        "overall": mean_overall,
        "overall_std": std_overall,
        "by_group": mean_by_group,
        "by_group_std": std_by_group,
        "folds": fold_results,
    }
    if output_path is not None:
        _save_metrics_json(results, output_path)
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate model fairness")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--method",
        choices=["baseline", "reweight", "postprocess", "expgrad"],
        default="baseline",
        help="Training method to use",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Proportion of data to use for testing",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/credit_data.csv",
        help="Path to the dataset CSV file",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=1,
        help="Number of cross-validation folds (1 to disable)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional probability threshold for predictions",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write metrics as JSON",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the train/test split",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Increase logging verbosity",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if args.cv > 1:
        run_cross_validation(
            method=args.method,
            cv=args.cv,
            random_state=args.random_state,
            output_path=args.output_json,
            data_path=args.data_path,
            threshold=args.threshold,
        )
    else:
        run_pipeline(
            method=args.method,
            test_size=args.test_size,
            output_path=args.output_json,
            random_state=args.random_state,
            data_path=args.data_path,
            threshold=args.threshold,
        )


if __name__ == "__main__":
    main()
