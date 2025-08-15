"""
Scalable Research Platform for Fairness ML.

This module provides distributed computing capabilities, cloud integration,
and performance optimization for large-scale fairness research.
"""

from .cloud_integration import AutoScaler, CloudExperimentManager, ResourceManager
from .distributed_computing import (
    ComputeCluster,
    DistributedFairnessFramework,
    TaskScheduler,
)
from .performance_optimizer import DataOptimizer, ModelOptimizer, PerformanceOptimizer

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
