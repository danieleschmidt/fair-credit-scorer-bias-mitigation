"""Utility and analysis tools for the Fair Credit Scorer project."""

from typing import TYPE_CHECKING

__all__ = [
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

if TYPE_CHECKING:  # pragma: no cover - used for type checking only
    from .architecture_review import ArchitectureReview
    from .baseline_model import evaluate_model, train_baseline_model
    from .bias_mitigator import (
        expgrad_demographic_parity,
        postprocess_equalized_odds,
        reweight_samples,
    )
    from .data_loader_preprocessor import load_credit_data, load_credit_dataset
    from .evaluate_fairness import run_cross_validation, run_pipeline
    from .fairness_metrics import compute_fairness_metrics


def __getattr__(name: str):
    """Lazily import objects so heavy dependencies load only when required."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    module_map = {
        "ArchitectureReview": ("architecture_review", "ArchitectureReview"),
        "evaluate_model": ("baseline_model", "evaluate_model"),
        "train_baseline_model": ("baseline_model", "train_baseline_model"),
        "expgrad_demographic_parity": ("bias_mitigator", "expgrad_demographic_parity"),
        "postprocess_equalized_odds": ("bias_mitigator", "postprocess_equalized_odds"),
        "reweight_samples": ("bias_mitigator", "reweight_samples"),
        "load_credit_data": ("data_loader_preprocessor", "load_credit_data"),
        "load_credit_dataset": ("data_loader_preprocessor", "load_credit_dataset"),
        "run_cross_validation": ("evaluate_fairness", "run_cross_validation"),
        "run_pipeline": ("evaluate_fairness", "run_pipeline"),
        "compute_fairness_metrics": ("fairness_metrics", "compute_fairness_metrics"),
    }
    module_name, attr = module_map[name]
    module = importlib.import_module(f".{module_name}", __name__)
    return getattr(module, attr)
