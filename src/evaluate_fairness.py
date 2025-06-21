from data_loader_preprocessor import load_credit_data
from baseline_model import train_baseline_model, evaluate_model
from bias_mitigator import reweight_samples
from fairness_metrics import compute_fairness_metrics


def run_pipeline(use_mitigation=False):
    X_train, X_test, y_train, y_test = load_credit_data()

    features_train = X_train.drop("protected", axis=1)
    features_test = X_test.drop("protected", axis=1)

    if use_mitigation:
        sample_weights = reweight_samples(y_train, X_train["protected"])
        model = train_baseline_model(features_train, y_train, sample_weight=sample_weights)
    else:
        model = train_baseline_model(features_train, y_train)

    accuracy = evaluate_model(model, features_test, y_test)
    overall, by_group = compute_fairness_metrics(
        y_true=y_test,
        y_pred=model.predict(features_test),
        protected=X_test["protected"],
    )
    print(f"Accuracy: {accuracy:.3f}")
    print("Overall fairness metrics:\n", overall)
    print("Metrics by group:\n", by_group)


if __name__ == "__main__":
    run_pipeline(use_mitigation=True)
