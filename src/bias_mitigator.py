import pandas as pd
from sklearn.linear_model import LogisticRegression


def expgrad_demographic_parity(X, y, protected):
    """Train a classifier using the Exponentiated Gradient algorithm.

    The returned model is optimized for demographic parity using
    ``fairlearn.reductions.ExponentiatedGradient``.

    Parameters
    ----------
    X : pandas.DataFrame
        Training features.
    y : array-like
        Training labels.
    protected : array-like
        Protected attribute values for each training sample.

    Returns
    -------
    ExponentiatedGradient
        Fitted model enforcing demographic parity.
    """
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity

    base_est = LogisticRegression(max_iter=1000)
    mitigator = ExponentiatedGradient(base_est, DemographicParity())
    mitigator.fit(X, y, sensitive_features=protected)
    return mitigator


def reweight_samples(y, protected):
    """Return sample weights that balance label distribution across protected groups."""
    if not isinstance(protected, pd.Series):
        protected = pd.Series(protected, name="protected")

    df = pd.DataFrame({"y": y, "protected": protected})
    # compute weights per label/protected combination
    counts = df.value_counts().rename("count").reset_index()
    total = len(df)
    weights = {}
    for _, row in counts.iterrows():
        group = (row["y"], row["protected"])
        weights[group] = total / (len(counts) * row["count"])

    return [weights[(yi, pi)] for yi, pi in zip(df["y"], df["protected"])]


def postprocess_equalized_odds(model, X, y, protected):
    """Post-process predictions using the Equalized Odds constraint.

    Parameters
    ----------
    model : estimator
        A fitted scikit-learn compatible classifier with ``predict_proba``.
    X : pandas.DataFrame
        Training features.
    y : array-like
        Training labels.
    protected : array-like
        Protected attribute values for each training sample.

    Returns
    -------
    ThresholdOptimizer
        A post-processed classifier enforcing the equalized odds constraint.
    """
    from fairlearn.postprocessing import ThresholdOptimizer

    optimizer = ThresholdOptimizer(
        estimator=model,
        constraints="equalized_odds",
        predict_method="predict_proba",
        prefit=True,
    )
    optimizer.fit(X, y, sensitive_features=protected)
    return optimizer
