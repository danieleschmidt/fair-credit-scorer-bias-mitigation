"""Fair Credit Scorer — bias detection and mitigation toolkit.

Core modules:
    BiasDetector   — measure fairness metrics across protected attributes
    BiasMitigator  — reweighting and threshold-adjustment strategies
    FairnessReport — structured audit report generation
"""

from .bias_detector import BiasDetector
from .bias_mitigator import BiasMitigator
from .fairness_report import FairnessReport

__version__ = "1.0.0"
__all__ = ["BiasDetector", "BiasMitigator", "FairnessReport"]
