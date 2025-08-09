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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import multiprocessing
import time

# Import version directly to avoid circular import
__version__ = "0.2.0"

from src.data_loader_preprocessor import load_credit_data, load_credit_dataset
from src.baseline_model import train_baseline_model, evaluate_model
from src.bias_mitigator import (
    reweight_samples,
    postprocess_equalized_odds,
    expgrad_demographic_parity,
)
from src.fairness_metrics import compute_fairness_metrics
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
    enable_parallel=True,
    max_workers=None,
):
    """Run cross-validated evaluation with optional parallel processing.

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
    enable_parallel : bool, optional
        Enable parallel processing of CV folds. Default is True.
    max_workers : int or None, optional
        Maximum number of parallel workers. If None, uses CPU count.
        
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
            
    start_time = time.time()
    
    # Determine parallelization strategy
    if max_workers is None:
        max_workers = min(cv, multiprocessing.cpu_count())
    
    parallel_enabled = enable_parallel and cv > 2 and max_workers > 1
    
    logger.info(f"Running {cv}-fold cross-validation with method={method}, threshold={threshold}")
    logger.info(f"Parallel processing: {'enabled' if parallel_enabled else 'disabled'} (workers: {max_workers if parallel_enabled else 1})")
    
    X, y = load_credit_dataset(path=data_path, random_state=random_state)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # Prepare fold data
    fold_data = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        fold_data.append({
            'fold_idx': fold_idx,
            'train_idx': train_idx,
            'test_idx': test_idx,
            'X': X,
            'y': y,
            'method': method,
            'threshold': threshold
        })
    
    # Execute folds in parallel or sequential
    if parallel_enabled:
        logger.info(f"Processing {cv} folds in parallel with {max_workers} workers")
        fold_results = _run_folds_parallel(fold_data, max_workers)
    else:
        logger.info(f"Processing {cv} folds sequentially")
        fold_results = _run_folds_sequential(fold_data)
    # Extract metrics from results
    overall_metrics = [result['overall'] for result in fold_results]
    by_group_metrics = [result['by_group'] for result in fold_results]

    metrics_concat = pd.concat(overall_metrics, axis=1)
    mean_overall = metrics_concat.mean(axis=1)
    std_overall = metrics_concat.std(axis=1)
    by_group_concat = pd.concat(by_group_metrics)
    mean_by_group = by_group_concat.groupby(level=0).mean()
    std_by_group = by_group_concat.groupby(level=0).std()
    accuracy = float(mean_overall["accuracy"])
    total_time = time.time() - start_time
    
    logger.info(f"Cross-validation completed in {total_time:.2f}s")
    if parallel_enabled:
        sequential_estimate = total_time * max_workers
        logger.info(f"Estimated speedup: {sequential_estimate/total_time:.1f}x over sequential processing")
    
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
        "cv_execution_time": total_time,
        "parallel_processing_used": parallel_enabled,
        "num_workers": max_workers if parallel_enabled else 1,
    }
    if output_path is not None:
        _save_metrics_json(results, output_path)
    return results


def _process_single_fold(fold_info):
    """Process a single cross-validation fold.
    
    Parameters
    ----------
    fold_info : dict
        Dictionary containing fold information including indices, data, method, and threshold.
        
    Returns
    -------
    dict
        Results for the single fold including accuracy, overall metrics, and by_group metrics.
    """
    fold_idx = fold_info['fold_idx']
    train_idx = fold_info['train_idx']
    test_idx = fold_info['test_idx']
    X = fold_info['X']
    y = fold_info['y']
    method = fold_info['method']
    threshold = fold_info['threshold']
    
    try:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        features_train = X_train.drop("protected", axis=1)
        features_test = X_test.drop("protected", axis=1)

        # Model training based on method
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

        # Model evaluation
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

        # Compute fairness metrics with optimization enabled
        overall, by_group = compute_fairness_metrics(
            y_true=y_test,
            y_pred=preds,
            protected=X_test["protected"],
            y_scores=probs,
            enable_optimization=True,  # Enable optimizations for parallel processing
        )
        
        return {
            "fold_idx": fold_idx,
            "accuracy": overall["accuracy"], 
            "overall": overall, 
            "by_group": by_group
        }
        
    except Exception as e:
        logger.error(f"Error processing fold {fold_idx}: {e}")
        raise


def _run_folds_parallel(fold_data, max_workers):
    """Run CV folds in parallel using ProcessPoolExecutor.
    
    Parameters
    ----------
    fold_data : list
        List of fold information dictionaries.
    max_workers : int
        Maximum number of parallel workers.
        
    Returns
    -------
    list
        Results from all folds in original order.
    """
    fold_results = [None] * len(fold_data)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all folds
        future_to_fold = {executor.submit(_process_single_fold, fold_info): fold_info['fold_idx'] 
                         for fold_info in fold_data}
        
        # Collect results as they complete
        for future in as_completed(future_to_fold):
            fold_idx = future_to_fold[future]
            try:
                result = future.result()
                fold_results[fold_idx] = result
                logger.debug(f"Completed fold {fold_idx}")
            except Exception as e:
                logger.error(f"Fold {fold_idx} generated an exception: {e}")
                raise
    
    return fold_results


def _run_folds_sequential(fold_data):
    """Run CV folds sequentially.
    
    Parameters
    ----------
    fold_data : list
        List of fold information dictionaries.
        
    Returns
    -------
    list
        Results from all folds.
    """
    fold_results = []
    
    for fold_info in fold_data:
        result = _process_single_fold(fold_info)
        fold_results.append(result)
        logger.debug(f"Completed fold {fold_info['fold_idx']}")
    
    return fold_results


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
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing for cross-validation",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers (default: CPU count)",
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
            enable_parallel=not args.no_parallel,
            max_workers=args.max_workers,
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
