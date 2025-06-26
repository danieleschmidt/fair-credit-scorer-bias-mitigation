# Project Architecture

This document describes the layout of the *Fair Credit Scorer: Bias Mitigation* project and highlights the main modules in the code base.

```
fair-credit-scorer-bias-mitigation/
├── architecture/              # Generated dependency diagram and summary
├── data/                      # Input dataset and artifacts
├── src/                       # Python source code
│   ├── __init__.py            # Utility exports for architecture tools
│   ├── architecture_review.py # Generates diagram of module relationships
│   ├── baseline_model.py      # Logistic regression model
│   ├── bias_mitigator.py      # Bias mitigation utilities
│   ├── data_loader_preprocessor.py  # Data loading and preprocessing helpers
│   ├── evaluate_fairness.py   # Pipeline orchestrating training and evaluation
│   ├── fairness_metrics.py    # Collection of fairness metrics
│   └── fair_credit_scorer_bias_mitigation/
│       ├── __init__.py        # Public API
│       └── cli.py             # CLI entrypoint
└── tests/                     # Pytest suite
```

The `src` directory is both the package root and the location of standalone modules used by the CLI. The installable package `fair_credit_scorer_bias_mitigation` re-exports the core functions so that they can be imported directly from the package root or invoked via the `fairness-eval` console command.

## Design Principles

- **Reproducible experiments** – The data loader can generate a synthetic dataset if no CSV file is supplied, ensuring tests and examples work out-of-the-box.
- **Modular pipeline** – Training, mitigation and evaluation are split into separate modules so new techniques can be swapped in easily.
- **Extensible metrics** – `fairness_metrics.py` centralises computation of both performance and fairness metrics, making it straightforward to add new ones.
- **Command-line interface** – Users can run the full pipeline with different mitigation methods using the `fairness-eval` CLI.

For a visual overview of module dependencies, run:

```bash
python -m src.architecture_review
```

This generates `architecture/diagram.svg` and `architecture/architecture_review.md` which detail the project's internal and external dependencies.
