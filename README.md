# fair-credit-scorer-bias-mitigation

Fair Credit Scorer: Bias Mitigation in Lending Models + DevSecOps Automation
**Version 0.2.0**

This project serves two primary purposes:
1. **Fair Credit Scoring**: Build a credit scoring model and explore techniques to identify and mitigate demographic bias
2. **DevSecOps Automation**: Demonstrate autonomous repository hygiene management with the included repo-hygiene-bot

The goal is to develop a model that is not only accurate but also fair with respect to specified protected attributes, while showcasing modern DevSecOps practices through automated repository management.

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

## 🤖 Repository Hygiene Bot

This project includes an **autonomous DevSecOps repository hygiene bot** that applies standardized security and development practices across GitHub repositories. The bot demonstrates advanced automation capabilities for managing repository compliance at scale.

### Key Features

- **🔒 Automated Security**: Deploys CodeQL, Dependabot, and OpenSSF Scorecard
- **📊 SBOM Generation**: Creates Software Bills of Materials for supply chain transparency  
- **📝 Community Standards**: Adds LICENSE, CODE_OF_CONDUCT, CONTRIBUTING, and SECURITY files
- **🏷️ Metadata Management**: Standardizes descriptions, topics, and homepages
- **📋 README Enhancement**: Injects status badges and required documentation sections
- **📈 Compliance Metrics**: Tracks hygiene status across all repositories

### Quick Start

```bash
# Set up environment
export GITHUB_TOKEN="your_token_here"
export GITHUB_USER="your_username"

# Run bot with dry-run (safe preview)
./scripts/run-hygiene-bot.sh --dry-run --verbose

# Apply changes to all repositories
./scripts/run-hygiene-bot.sh

# Process single repository
python -m src.repo_hygiene_cli --single-repo my-repo
```

### Automated Execution

The bot can run automatically via GitHub Actions (workflow file needs to be created manually due to permissions). See the [complete documentation](docs/REPO_HYGIENE_BOT.md) for setup instructions.

### Documentation

- **[Complete Bot Documentation](docs/REPO_HYGIENE_BOT.md)** - Comprehensive guide and API reference
- **[Configuration Guide](config/repo-hygiene.yaml)** - Customization options
- **[Architecture Overview](ARCHITECTURE.md)** - System design and components

### What It Does

The bot systematically applies DevSecOps best practices by:

1. **Metadata Updates**: Ensures descriptions, topics, and homepages are set
2. **Community Files**: Creates LICENSE, CODE_OF_CONDUCT, CONTRIBUTING, SECURITY files
3. **Security Workflows**: Deploys CodeQL, Dependabot, OpenSSF Scorecard scanning
4. **SBOM & Signing**: Generates Software Bills of Materials with Cosign signing
5. **README Standards**: Adds badges and required documentation sections
6. **Compliance Tracking**: Collects metrics and generates reports

All changes are made via pull requests with detailed explanations, ensuring full audit trails and allowing manual review before merging.

### Sample Output

```
🎯 Repository Hygiene Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Overall Statistics:
   Total repositories processed: 25
   Fully compliant repositories: 18
   Repositories needing improvements: 7
   Overall compliance rate: 72.0%

🔍 Issues Found:
   Missing descriptions: 2
   Missing licenses: 3
   Missing CodeQL scanning: 5
   Missing Dependabot: 4
   Insufficient topics (<5): 6
   Missing SBOM workflow: 7
```

This bot serves as a reference implementation for autonomous DevSecOps practices, demonstrating how to scale security and compliance across large repository portfolios.
