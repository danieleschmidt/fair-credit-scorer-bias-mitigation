"""Public API for the Fair Credit Scorer package."""

__version__ = "0.2.0"

from architecture_review import ArchitectureReview
from baseline_model import evaluate_model, train_baseline_model
from bias_mitigator import (
    expgrad_demographic_parity,
    postprocess_equalized_odds,
    reweight_samples,
)
from data_loader_preprocessor import load_credit_data, load_credit_dataset
from evaluate_fairness import run_cross_validation, run_pipeline
from fairness_metrics import compute_fairness_metrics

__all__ = [
    "__version__",
    "ArchitectureReview",
    "compute_fairness_metrics",
    "evaluate_model",
    "expgrad_demographic_parity",
    "load_credit_data",
    "load_credit_dataset",
    "postprocess_equalized_odds",
    "reweight_samples",
    "run_cross_validation",
    "run_pipeline",
    "train_baseline_model",
]
