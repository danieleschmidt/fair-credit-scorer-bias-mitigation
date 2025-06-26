import argparse

from fair_credit_scorer_bias_mitigation import __version__

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
    """Write metrics dictionary to ``path`` as JSON."""
    with open(path, "w") as f:
        json.dump(_serialize_metrics(results), f)


def run_pipeline(
    method="baseline",
    test_size=0.3,
    output_path=None,
    random_state=42,
    data_path="data/credit_data.csv",
    threshold=None,
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

    Returns
    -------
    dict
        Dictionary containing ``accuracy``, ``overall``, and ``by_group`` metrics.
    """

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
    print(f"Accuracy: {accuracy:.3f}")
    print("Overall fairness metrics:\n", overall)
    print("Metrics by group:\n", by_group)

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
        Number of cross-validation splits.
    random_state : int, optional
        Seed controlling the cross-validation shuffle.
    output_path : str or None, optional
        If provided, write the averaged metrics to this JSON file.
    threshold : float or None, optional
        Custom probability threshold applied in every fold. ``None`` uses the
        estimator's default ``predict`` behaviour.
    """
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
    print("Average overall metrics:\n", mean_overall)
    print("Standard deviation of metrics across folds:\n", std_overall)
    print("Average metrics by group:\n", mean_by_group)
    print("Standard deviation by group:\n", std_by_group)
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
    args = parser.parse_args()

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
