"""
Research framework for fair machine learning algorithms.

This module provides comprehensive tools for conducting reproducible fairness research,
including experimental design, statistical validation, and publication-ready results.
"""

from .benchmarking_suite import (
    FairnessBenchmarkSuite,
    PerformanceMetrics,
    StandardDatasets,
)
from .experimental_framework import (
    ExperimentalFramework,
    ResearchProtocol,
    StatisticalValidation,
)
from .reproducibility_manager import (
    ExperimentConfig,
    ReproducibilityManager,
    ResultsManager,
)

__all__ = [
    'ExperimentalFramework',
    'ResearchProtocol',
    'StatisticalValidation',
    'FairnessBenchmarkSuite',
    'StandardDatasets',
    'PerformanceMetrics',
    'ReproducibilityManager',
    'ExperimentConfig',
    'ResultsManager'
]
