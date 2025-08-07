"""
Research framework for fair machine learning algorithms.

This module provides comprehensive tools for conducting reproducible fairness research,
including experimental design, statistical validation, and publication-ready results.
"""

from .experimental_framework import (
    ExperimentalFramework,
    ResearchProtocol,
    StatisticalValidation
)

from .benchmarking_suite import (
    FairnessBenchmarkSuite,
    StandardDatasets,
    PerformanceMetrics
)

from .reproducibility_manager import (
    ReproducibilityManager,
    ExperimentConfig,
    ResultsManager
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