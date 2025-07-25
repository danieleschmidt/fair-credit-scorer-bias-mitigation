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

## Model Explainability Features

The system now includes SHAP-based model explainability features for understanding
credit scoring decisions and identifying potential bias.

### Using the ModelExplainer Class

```python
from src.model_explainability import ModelExplainer
from src.baseline_model import train_baseline_model
import pandas as pd

# Train a model
X_train = pd.DataFrame({
    'age': [25, 35, 45],
    'income': [30000, 50000, 70000],
    'credit_score': [600, 700, 800]
})
y_train = [0, 1, 1]

model = train_baseline_model(X_train, y_train)

# Initialize explainer
explainer = ModelExplainer(model, X_train)

# Explain a prediction
test_instance = pd.DataFrame({
    'age': [30],
    'income': [40000],
    'credit_score': [650]
})

explanation = explainer.explain_prediction(test_instance)
print("SHAP values:", explanation['shap_values'])
print("Feature importance:", explanation['feature_importance'])
```

### Explainability API Endpoints

For production use, the system provides REST API endpoints for model explanations:

```python
from src.explainability_api import ExplainabilityAPI

# Initialize API
api = ExplainabilityAPI()

# Load a model (in production, this would load from file)
api._load_model('model.pkl')

# Explain a prediction
features = {
    'age': 35,
    'income': 50000,
    'credit_score': 700,
    'debt_to_income': 0.2
}

explanation = api.explain_prediction(features)
print("API Response:", explanation)

# Get global feature importance
importance = api.get_feature_importance()
print("Feature Importance:", importance)
```

### CLI Testing

Test the explainability API from command line:

```bash
cd src/
python explainability_api.py --test
```

Or explain specific features:

```bash
cd src/
python explainability_api.py --features '{"age": 35, "income": 50000, "credit_score": 700}'
```

### Flask Web API (Optional)

If Flask is installed, you can create web endpoints:

```python
from src.explainability_api import create_flask_app

app = create_flask_app()
if app:
    app.run(debug=True)
```

Available endpoints:
- `GET /health` - API health check
- `POST /explain` - Explain a prediction (JSON body with features)
- `GET /feature-importance` - Get global feature importance
- `POST /load-model` - Load a model from path
