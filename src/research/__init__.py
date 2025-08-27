"""
Research framework for fair machine learning algorithms.

This module provides comprehensive tools for conducting reproducible fairness research,
including experimental design, statistical validation, and publication-ready results.
"""

# Import only the new components to avoid import errors with existing modules
try:
    from .emergent_fairness_consciousness import (
        EmergentFairnessNeuron,
        EmergentFairnessConsciousness,
        demonstrate_emergent_fairness_consciousness
    )
    from .temporal_fairness_preservation import (
        TemporalFairnessMemory,
        TemporalFairnessPreserver,
        demonstrate_temporal_fairness_preservation
    )
    _new_components_available = True
except ImportError:
    _new_components_available = False

# Try to import existing components
try:
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
    _existing_components_available = True
except ImportError:
    _existing_components_available = False

# Dynamic __all__ based on available components
__all__ = []

if _new_components_available:
    __all__.extend([
        'EmergentFairnessNeuron',
        'EmergentFairnessConsciousness',
        'demonstrate_emergent_fairness_consciousness',
        'TemporalFairnessMemory',
        'TemporalFairnessPreserver',
        'demonstrate_temporal_fairness_preservation'
    ])

if _existing_components_available:
    __all__.extend([
        'ExperimentalFramework',
        'ResearchProtocol',
        'StatisticalValidation',
        'FairnessBenchmarkSuite',
        'StandardDatasets',
        'PerformanceMetrics',
        'ReproducibilityManager',
        'ExperimentConfig',
        'ResultsManager'
    ])
