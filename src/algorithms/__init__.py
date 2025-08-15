"""
Advanced algorithms package for fair credit scoring.

This package contains sophisticated algorithms for bias detection,
fairness optimization, and advanced machine learning techniques.

Modules:
    - bias_detection: Real-time bias detection algorithms
    - fairness_optimization: Multi-objective fairness optimization
    - ensemble: Ensemble methods for improved fairness-accuracy tradeoffs
    - adaptive: Adaptive learning algorithms that adjust to data drift
"""

__version__ = "0.2.0"

from .bias_detection import DriftDetectionAlgorithm, RealTimeBiasDetector
from .ensemble import FairEnsemble, StackedFairnessModel
from .fairness_optimization import FairnessOptimizer, MultiObjectiveOptimizer

__all__ = [
    "RealTimeBiasDetector",
    "DriftDetectionAlgorithm",
    "FairnessOptimizer",
    "MultiObjectiveOptimizer",
    "FairEnsemble",
    "StackedFairnessModel"
]
