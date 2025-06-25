from fair_credit_scorer_bias_mitigation import load_credit_dataset, train_baseline_model
from sklearn.linear_model import LogisticRegression


def test_returns_logistic_regression(tmp_path):
    X, y = load_credit_dataset(path=str(tmp_path / "credit.csv"))
    model = train_baseline_model(X, y)
    assert isinstance(model, LogisticRegression)
    assert hasattr(model, "coef_")
    assert model.solver == "liblinear"
