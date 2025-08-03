"""
Advanced features package for fair credit scoring.

This package contains cutting-edge features including automated optimization,
advanced analytics, and experimental fairness techniques.

Modules:
    - optimization: Automated hyperparameter and fairness optimization
    - analytics: Advanced analytics and reporting capabilities
    - experimental: Experimental fairness techniques and research features
    - automation: Automated model management and lifecycle optimization
"""

__version__ = "0.2.0"

from .optimization import AutoFairnessOptimizer, HyperparameterTuner
from .analytics import AdvancedAnalytics, FairnessReporter
from .automation import ModelLifecycleManager, AutoRetrainer

__all__ = [
    "AutoFairnessOptimizer",
    "HyperparameterTuner",
    "AdvancedAnalytics", 
    "FairnessReporter",
    "ModelLifecycleManager",
    "AutoRetrainer"
]