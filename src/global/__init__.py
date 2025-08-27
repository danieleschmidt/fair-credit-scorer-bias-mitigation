"""
Global-First Implementation Components.

This module provides internationalization, localization, and global compliance
support for the Fair Credit Scorer system.
"""

# Import new global-first components
try:
    from .internationalization import (
        InternationalizationManager,
        GlobalFairnessValidator,
        SupportedLanguage,
        ComplianceRegion,
        CulturalFairnessStandard,
        demonstrate_global_fairness_framework
    )
    _new_global_available = True
except ImportError:
    _new_global_available = False

# Try to import existing components
try:
    from .cultural_fairness import (
        CrossCulturalValidator,
        CulturalContextManager,
        CulturalFairnessFramework,
    )
    from .international_compliance import (
        ComplianceManager,
        PrivacyFramework,
        RegulationChecker,
    )
    from .multi_dataset_validation import (
        CrossDatasetValidator,
        GlobalBenchmarkSuite,
        UniversalMetrics,
    )
    _existing_global_available = True
except ImportError:
    _existing_global_available = False

# Dynamic __all__ based on available components
__all__ = []

if _new_global_available:
    __all__.extend([
        'InternationalizationManager',
        'GlobalFairnessValidator', 
        'SupportedLanguage',
        'ComplianceRegion',
        'CulturalFairnessStandard',
        'demonstrate_global_fairness_framework'
    ])

if _existing_global_available:
    __all__.extend([
        'CulturalFairnessFramework',
        'CrossCulturalValidator',
        'CulturalContextManager',
        'ComplianceManager',
        'RegulationChecker',
        'PrivacyFramework',
        'GlobalBenchmarkSuite',
        'CrossDatasetValidator',
        'UniversalMetrics'
    ])
