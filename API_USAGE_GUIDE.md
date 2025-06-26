# API Usage Guide

This guide shows how to run the fairness evaluation pipeline from Python or the command line.

## Running from Python

```python
from fair_credit_scorer_bias_mitigation import run_pipeline, run_cross_validation

# run a single train/test split with sample reweighting
results = run_pipeline(method="reweight", test_size=0.3)
print(results["overall"])

# run cross validation using the post-processing mitigation
cv_results = run_cross_validation(method="postprocess", cv=5)
print(cv_results["overall"])
```

The returned dictionary contains the accuracy and fairness metrics produced by
`compute_fairness_metrics`. When running cross validation, the per-fold metrics
are included under the `"folds"` key.

## Command Line Interface

Install the package in editable mode and use the `fairness-eval` entry point:

```bash
pip install -e .
fairness-eval --method expgrad --cv 3 --output-json results.json
```

Use `--help` to see all available options. The CLI mirrors the arguments of
`run_pipeline` and `run_cross_validation`.
