# fair-credit-scorer-bias-mitigation

Fair Credit Scorer: Bias Mitigation in Lending Models
This project aims to build a credit scoring model and explore techniques to identify and mitigate demographic bias. The goal is to develop a model that is not only accurate but also fair with respect to specified protected attributes.

## Project Goals
- Develop a baseline credit scoring model.
- Implement and calculate fairness metrics (e.g., demographic parity difference, equalized odds difference,
  false positive/negative rate differences, accuracy difference).
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
```

## Usage
The dataset is generated automatically the first time you run the pipeline. Execute the evaluation script to train the model and print fairness metrics:
```bash
python -m src.evaluate_fairness  # add --help for options
```
Choose a training method with `--method`. Options are `baseline`,
`reweight`, or `postprocess`. Use `--test-size` to adjust the train/test
split (default 0.3) and `--random-state` for reproducible splits.
Pass `--output-json metrics.json` to also save the results to a file.
Interactive exploration is available in `notebooks/fairness_exploration.ipynb`,
which demonstrates running the pipeline with each mitigation approach.
The `run_pipeline` function used by the CLI also returns a dictionary of the
accuracy and fairness metrics so you can incorporate the results programmatically.

## Testing
Run the unit tests with pytest:
```bash
pytest -q
```

## Findings
Initial experiments show that the baseline logistic regression model reaches
around 0.83 accuracy but exhibits notable group disparity. Applying the simple
sample reweighting strategy improves the equalized odds difference from roughly
0.28 to 0.21 and makes selection rates more similar across protected groups,
with accuracy dropping slightly to about 0.79. This demonstrates the trade-off
between fairness and performance when using basic mitigation techniques.
