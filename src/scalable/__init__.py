"""
Scalable Research Platform for Fairness ML.

This module provides distributed computing capabilities, cloud integration,
and performance optimization for large-scale fairness research.
"""

from .distributed_computing import (
    DistributedFairnessFramework,
    TaskScheduler,
    ComputeCluster
)

from .cloud_integration import (
    CloudExperimentManager,
    ResourceManager,
    AutoScaler
)

from .performance_optimizer import (
    PerformanceOptimizer,
    ModelOptimizer,
    DataOptimizer
)

__all__ = [
    'DistributedFairnessFramework',
    'TaskScheduler',
    'ComputeCluster',
    'CloudExperimentManager',
    'ResourceManager',
    'AutoScaler',
    'PerformanceOptimizer',
    'ModelOptimizer',
    'DataOptimizer'
]