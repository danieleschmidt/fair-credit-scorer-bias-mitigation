import argparse

from .data_loader_preprocessor import load_credit_data
from .baseline_model import train_baseline_model, evaluate_model
from .bias_mitigator import reweight_samples, postprocess_equalized_odds
from .fairness_metrics import compute_fairness_metrics


def run_pipeline(method="baseline", test_size=0.3, output_path=None, random_state=42):
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
    random_state : int, optional
        Random seed used when splitting the data, by default 42.

    Returns
    -------
    dict
        Dictionary containing ``accuracy``, ``overall``, and ``by_group`` metrics.
    """

    X_train, X_test, y_train, y_test = load_credit_data(
        test_size=test_size, random_state=random_state
    )

    features_train = X_train.drop("protected", axis=1)
    features_test = X_test.drop("protected", axis=1)

    if method == "reweight":
        sample_weights = reweight_samples(y_train, X_train["protected"])
        model = train_baseline_model(features_train, y_train, sample_weight=sample_weights)
    else:
        model = train_baseline_model(features_train, y_train)
        if method == "postprocess":
            model = postprocess_equalized_odds(
                model,
                features_train,
                y_train,
                X_train["protected"],
            )

    if method == "postprocess":
        accuracy, preds = evaluate_model(
            model,
            features_test,
            y_test,
            sensitive_features=X_test["protected"],
        )
    else:
        accuracy, preds = evaluate_model(model, features_test, y_test)

    overall, by_group = compute_fairness_metrics(
        y_true=y_test,
        y_pred=preds,
        protected=X_test["protected"],
    )
    print(f"Accuracy: {accuracy:.3f}")
    print("Overall fairness metrics:\n", overall)
    print("Metrics by group:\n", by_group)

    results = {"accuracy": accuracy, "overall": overall, "by_group": by_group}
    if output_path is not None:
        import json

        with open(output_path, "w") as f:
            json.dump(results, f, default=str)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate model fairness")
    parser.add_argument(
        "--method",
        choices=["baseline", "reweight", "postprocess"],
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

    run_pipeline(
        method=args.method,
        test_size=args.test_size,
        output_path=args.output_json,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
