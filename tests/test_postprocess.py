import numpy as np
from src.data_loader_preprocessor import load_credit_data
from src.baseline_model import train_baseline_model
from src.bias_mitigator import postprocess_equalized_odds


def test_postprocess_predict_length():
    X_train, X_test, y_train, y_test = load_credit_data(test_size=0.5, random_state=0)
    features_train = X_train.drop("protected", axis=1)
    features_test = X_test.drop("protected", axis=1)

    base_model = train_baseline_model(features_train, y_train)
    opt_model = postprocess_equalized_odds(base_model, features_train, y_train, X_train["protected"])
    preds = opt_model.predict(features_test, sensitive_features=X_test["protected"])

    assert len(preds) == len(y_test)
