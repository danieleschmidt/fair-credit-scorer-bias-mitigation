"""
Business logic services for fair credit scoring.

This package contains core business services that implement the primary
functionality of the fair credit scoring system.

Services:
    - feature_engineering: Advanced feature engineering pipeline
    - bias_detection: Real-time bias detection algorithms
    - model_validation: Comprehensive model validation
    - remediation: Automated bias remediation strategies
"""

__version__ = "0.2.0"

from .bias_detection import BiasDetectionService
from .feature_engineering import FeatureEngineeringService
from .model_validation import ModelValidationService
from .remediation import RemediationService

__all__ = [
    "FeatureEngineeringService",
    "BiasDetectionService",
    "ModelValidationService",
    "RemediationService"
]
