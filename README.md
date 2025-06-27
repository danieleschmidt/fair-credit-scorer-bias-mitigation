# fair-credit-scorer-bias-mitigation

Fair Credit Scorer: Bias Mitigation in Lending Models
**Version 0.1.0**
This project aims to build a credit scoring model and explore techniques to identify and mitigate demographic bias. The goal is to develop a model that is not only accurate but also fair with respect to specified protected attributes.

## Project Goals
- Develop a baseline credit scoring model.
 - Implement and calculate fairness metrics (e.g., demographic parity difference, equalized odds difference,
  false positive/negative rates and their differences, true positive/negative rate
  differences, accuracy, balanced accuracy, precision, recall, F1,
  false discovery rate and its difference, ROC AUC differences,
  demographic parity and equalized odds ratios,
  false/true positive/negative rate ratios, accuracy ratio,
  and log loss difference).
- Apply at least one bias mitigation technique (e.g., re-weighting, adversarial debiasing, or a post-processing method).
- Evaluate and compare the model's performance and fairness before and after mitigation.
- Discuss the trade-offs between fairness and accuracy.

## Tech Stack (Planned)
- Python
- Scikit-learn
- Pandas, NumPy
- AIF360 or Fairlearn (for fairness metrics and mitigation algorithms)
- Matplotlib / Seaborn

## Initial File Structure
```
fair-credit-scorer-bias-mitigation/
├── data/                # dataset generated on first run (not versioned)
├── notebooks/
│   └── fairness_exploration.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader_preprocessor.py
│   ├── baseline_model.py
│   ├── fairness_metrics.py
│   ├── bias_mitigator.py
│   └── evaluate_fairness.py
├── tests/
│   ├── __init__.py
│   └── test_fairness_metrics.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation
Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # for development tools
pre-commit install  # set up git hooks for linting
# The hook runs Ruff and Bandit automatically before each commit
```

## Usage
The dataset is generated automatically the first time you run the pipeline.
Run the packaged CLI to train the model and print fairness metrics:
```bash
fairness-eval  # add --help for options
```
Choose a training method with `--method`. Options are `baseline`,
`reweight`, `postprocess`, or `expgrad`. Use `--test-size` to adjust the train/test
split (default 0.3) and `--random-state` for reproducible splits.
Specify `--data-path` to load or save the dataset at a different location.
Pass `--output-json metrics.json` to also save the results to a file. The JSON
contains the overall and by‑group metrics in nested dictionaries so it can be
easily parsed.
Use `--cv N` to evaluate with `N`-fold cross-validation instead of a single split.
Provide `--threshold T` to apply a custom probability threshold `T` when
converting model scores to predicted labels.
Use `--verbose` to enable debug-level logging for more detailed output.
When cross-validation is enabled, the script prints the average metrics across all folds and
also computes their standard deviation. `--output-json` will write these aggregated results,
including the per-fold metrics and fold-level statistics, to the specified path.
Interactive exploration is available in `notebooks/fairness_exploration.ipynb`,
which demonstrates running the pipeline with each mitigation approach.
The `run_pipeline` function used by the CLI also returns a dictionary of the
accuracy and fairness metrics so you can incorporate the results programmatically.
For more examples see [API_USAGE_GUIDE.md](API_USAGE_GUIDE.md).

## Architecture Overview
The project is organized around a simple data pipeline and model training flow:

1. **Data Loading** – `data_loader_preprocessor.py` provides helper functions
   to generate or load a CSV dataset and split it into train and test sets.
2. **Model Training** – `baseline_model.py` trains a logistic regression model
   on the provided features. Additional bias mitigation utilities are implemented
   in `bias_mitigator.py`.
3. **Fairness Evaluation** – `evaluate_fairness.py` orchestrates the pipeline,
   applying mitigation techniques and computing metrics via `fairness_metrics.py`.

Run the architecture review tool to generate a dependency diagram and summary:

```bash
python -m src.architecture_review
```

This will create `architecture/diagram.svg` and `architecture/architecture_review.md`
which document module relationships and external dependencies.
For a full description of the project layout see [ARCHITECTURE.md](ARCHITECTURE.md).

## Testing
Run the unit tests with coverage:
```bash
python -m src.run_tests
```

## Findings
Initial experiments show that the baseline logistic regression model reaches
around 0.83 accuracy but exhibits notable group disparity. Applying the simple
sample reweighting strategy improves the equalized odds difference from roughly
0.28 to 0.21 and makes selection rates more similar across protected groups,
with accuracy dropping slightly to about 0.79. This demonstrates the trade-off
between fairness and performance when using basic mitigation techniques.
For a deeper discussion, see [TRADEOFFS.md](TRADEOFFS.md).
See [CHANGELOG.md](CHANGELOG.md) for a list of recent updates.
