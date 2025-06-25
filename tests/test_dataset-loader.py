
import os
from fair_credit_scorer_bias_mitigation import load_credit_data


def test_generates_file_if_missing(tmp_path):
    data_file = tmp_path / "credit_data.csv"
    assert not data_file.exists()
    load_credit_data(path=str(data_file))
    assert data_file.exists()


def test_returns_train_test_split(tmp_path):
    data_file = tmp_path / "credit.csv"
    X_train, X_test, y_train, y_test = load_credit_data(path=str(data_file), test_size=0.3)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_train) + len(X_test) == len(y_train) + len(y_test)
