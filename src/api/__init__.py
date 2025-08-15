"""
Fair Credit Scorer API Package

This package provides REST API endpoints for the fair credit scoring system,
including bias monitoring, model explanations, and fairness evaluations.

Modules:
    - fairness_api: Core fairness evaluation and bias monitoring endpoints
    - model_registry: Model versioning and management API
    - monitoring: Real-time bias monitoring and alerting
    - auth: Authentication and authorization middleware
"""

__version__ = "0.2.0"

from .fairness_api import FairnessAPI
from .model_registry import ModelRegistry
from .monitoring import BiasMonitor

__all__ = ["FairnessAPI", "ModelRegistry", "BiasMonitor"]
