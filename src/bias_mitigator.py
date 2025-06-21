import pandas as pd


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
