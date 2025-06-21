from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train_baseline_model(X_train, y_train, sample_weight=None):
    """Train a simple logistic regression model."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def evaluate_model(model, X_test, y_test, sensitive_features=None):
    """Return accuracy score and predictions for the model."""
    if sensitive_features is not None:
        predictions = model.predict(X_test, sensitive_features=sensitive_features)
    else:
        predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions), predictions
