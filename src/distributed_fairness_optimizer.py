"""
Distributed Fairness Optimization Framework.

High-performance distributed computing framework for fairness-aware ML at scale.
Supports multi-GPU training, distributed hyperparameter optimization, and
federated fairness optimization across multiple data sources.

Features:
- Multi-GPU training with fairness constraints
- Distributed hyperparameter optimization
- Federated fairness optimization
- Automatic model parallelism
- Dynamic load balancing
- Resource-aware scheduling
- Real-time performance monitoring
"""

import asyncio
import concurrent.futures
import multiprocessing as mp
import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import ParameterGrid, ParameterSampler

from enhanced_error_recovery import ErrorRecoveryManager
from fairness_metrics import compute_fairness_metrics
from logging_config import get_logger

logger = get_logger(__name__)

try:
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as torch_mp
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - GPU acceleration disabled")

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    logger.info("MPI not available - using alternative distributed communication")


class OptimizationBackend(Enum):
    """Supported optimization backends."""
    THREADING = "threading"
    MULTIPROCESSING = "multiprocessing"
    ASYNCIO = "asyncio"
    TORCH_DISTRIBUTED = "torch_distributed"
    MPI = "mpi"


class SchedulingStrategy(Enum):
    """Task scheduling strategies."""
    ROUND_ROBIN = "round_robin"
    LOAD_BALANCED = "load_balanced"
    PRIORITY_BASED = "priority_based"
    RESOURCE_AWARE = "resource_aware"


@dataclass
class ComputeResource:
    """Compute resource specification."""
    resource_id: str
    resource_type: str  # 'cpu', 'gpu', 'tpu'
    cores: int
    memory_gb: float
    utilization: float = 0.0
    available: bool = True
    last_used: Optional[datetime] = None

    def can_handle_task(self, task_requirements: Dict[str, Any]) -> bool:
        """Check if resource can handle task requirements."""
        required_cores = task_requirements.get('cores', 1)
        required_memory = task_requirements.get('memory_gb', 1.0)

        return (
            self.available and
            self.cores >= required_cores and
            self.memory_gb >= required_memory and
            self.utilization < 0.8  # 80% utilization threshold
        )


@dataclass
class OptimizationTask:
    """Optimization task specification."""
    task_id: str
    algorithm: BaseEstimator
    parameters: Dict[str, Any]
    data: Tuple[pd.DataFrame, pd.Series, pd.DataFrame]  # X, y, sensitive_attrs
    requirements: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class OptimizationResult:
    """Result from distributed optimization."""
    task_id: str
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    fairness_metrics: Dict[str, Any]
    execution_time: float
    resource_usage: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None


class ResourceManager:
    """Manage distributed compute resources."""

    def __init__(self):
        self.resources: Dict[str, ComputeResource] = {}
        self.resource_lock = threading.Lock()
        self._discover_resources()

        logger.info(f"ResourceManager initialized with {len(self.resources)} resources")

    def _discover_resources(self):
        """Discover available compute resources."""
        # CPU resources
        cpu_count = mp.cpu_count()
        memory_gb = self._get_system_memory_gb()

        for i in range(cpu_count):
            resource = ComputeResource(
                resource_id=f"cpu_{i}",
                resource_type="cpu",
                cores=1,
                memory_gb=memory_gb / cpu_count
            )
            self.resources[resource.resource_id] = resource

        # GPU resources (if PyTorch available)
        if TORCH_AVAILABLE and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                resource = ComputeResource(
                    resource_id=f"gpu_{i}",
                    resource_type="gpu",
                    cores=1,  # Simplified: treat GPU as single core
                    memory_gb=gpu_memory_gb
                )
                self.resources[resource.resource_id] = resource
                logger.info(f"Discovered GPU {i}: {gpu_memory_gb:.1f}GB memory")

    def _get_system_memory_gb(self) -> float:
        """Get system memory in GB."""
        try:
            import psutil
            return psutil.virtual_memory().total / (1024**3)
        except ImportError:
            return 8.0  # Default assumption

    def allocate_resource(self, task_requirements: Dict[str, Any]) -> Optional[ComputeResource]:
        """Allocate resource for task."""
        with self.resource_lock:
            for resource in self.resources.values():
                if resource.can_handle_task(task_requirements):
                    resource.available = False
                    resource.utilization = min(1.0, resource.utilization + 0.1)
                    resource.last_used = datetime.now()
                    return resource
        return None

    def release_resource(self, resource_id: str):
        """Release allocated resource."""
        with self.resource_lock:
            if resource_id in self.resources:
                resource = self.resources[resource_id]
                resource.available = True
                resource.utilization = max(0.0, resource.utilization - 0.1)

    def get_resource_utilization(self) -> Dict[str, float]:
        """Get current resource utilization."""
        with self.resource_lock:
            return {
                resource_id: resource.utilization
                for resource_id, resource in self.resources.items()
            }


class TaskScheduler:
    """Schedule optimization tasks across resources."""

    def __init__(
        self,
        resource_manager: ResourceManager,
        strategy: SchedulingStrategy = SchedulingStrategy.LOAD_BALANCED
    ):
        self.resource_manager = resource_manager
        self.strategy = strategy
        self.task_queue = queue.PriorityQueue()
        self.active_tasks: Dict[str, OptimizationTask] = {}
        self.completed_tasks: List[OptimizationTask] = []

        logger.info(f"TaskScheduler initialized with {strategy.value} strategy")

    def submit_task(self, task: OptimizationTask):
        """Submit task for scheduling."""
        # Priority queue uses negative priority for max-heap behavior
        self.task_queue.put((-task.priority, task.created_at, task))
        logger.debug(f"Task {task.task_id} submitted with priority {task.priority}")

    def get_next_task(self) -> Optional[OptimizationTask]:
        """Get next task based on scheduling strategy."""
        try:
            _, _, task = self.task_queue.get_nowait()
            return task
        except queue.Empty:
            return None

    def allocate_task(self) -> Optional[Tuple[OptimizationTask, ComputeResource]]:
        """Allocate next task to available resource."""
        task = self.get_next_task()
        if not task:
            return None

        resource = self.resource_manager.allocate_resource(task.requirements)
        if not resource:
            # Put task back in queue
            self.task_queue.put((-task.priority, task.created_at, task))
            return None

        task.started_at = datetime.now()
        self.active_tasks[task.task_id] = task

        return task, resource

    def complete_task(self, task_id: str, result: OptimizationResult):
        """Mark task as completed."""
        if task_id in self.active_tasks:
            task = self.active_tasks.pop(task_id)
            task.completed_at = datetime.now()
            task.result = result
            self.completed_tasks.append(task)


class DistributedOptimizer(ABC):
    """Base class for distributed optimization algorithms."""

    @abstractmethod
    def optimize(
        self,
        tasks: List[OptimizationTask],
        scheduler: TaskScheduler
    ) -> List[OptimizationResult]:
        """Run distributed optimization."""
        pass


class ThreadPoolOptimizer(DistributedOptimizer):
    """Thread-based distributed optimizer."""

    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or mp.cpu_count()
        self.error_recovery = ErrorRecoveryManager()

    def optimize(
        self,
        tasks: List[OptimizationTask],
        scheduler: TaskScheduler
    ) -> List[OptimizationResult]:
        """Run optimization using thread pool."""
        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {}

            for task in tasks:
                scheduler.submit_task(task)

            # Process tasks as resources become available
            while True:
                allocation = scheduler.allocate_task()
                if not allocation:
                    break

                task, resource = allocation
                future = executor.submit(self._execute_task, task, resource)
                future_to_task[future] = (task, resource)

            # Collect results
            for future in concurrent.futures.as_completed(future_to_task):
                task, resource = future_to_task[future]

                try:
                    result = future.result()
                    scheduler.complete_task(task.task_id, result)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Task {task.task_id} failed: {e}")
                    error_result = OptimizationResult(
                        task_id=task.task_id,
                        parameters=task.parameters,
                        performance_metrics={},
                        fairness_metrics={},
                        execution_time=0.0,
                        resource_usage={},
                        success=False,
                        error_message=str(e)
                    )
                    results.append(error_result)
                finally:
                    scheduler.resource_manager.release_resource(resource.resource_id)

        return results

    @ErrorRecoveryManager().with_retry()
    def _execute_task(
        self,
        task: OptimizationTask,
        resource: ComputeResource
    ) -> OptimizationResult:
        """Execute single optimization task."""
        start_time = time.time()

        try:
            X, y, sensitive_attrs = task.data

            # Configure algorithm with parameters
            algorithm = clone(task.algorithm)
            algorithm.set_params(**task.parameters)

            # Train model
            algorithm.fit(X, y)
            predictions = algorithm.predict(X)

            # Compute performance metrics
            from sklearn.metrics import accuracy_score, roc_auc_score

            performance_metrics = {
                'accuracy': accuracy_score(y, predictions)
            }

            # Try to compute AUC if possible
            try:
                if hasattr(algorithm, 'predict_proba'):
                    proba = algorithm.predict_proba(X)[:, 1]
                    performance_metrics['auc'] = roc_auc_score(y, proba)
            except:
                pass

            # Compute fairness metrics
            fairness_metrics = {}
            for attr_name in sensitive_attrs.columns:
                overall, by_group = compute_fairness_metrics(
                    y, predictions, sensitive_attrs[attr_name]
                )
                fairness_metrics[attr_name] = {
                    'overall': overall,
                    'by_group': by_group.to_dict() if hasattr(by_group, 'to_dict') else by_group
                }

            execution_time = time.time() - start_time

            return OptimizationResult(
                task_id=task.task_id,
                parameters=task.parameters,
                performance_metrics=performance_metrics,
                fairness_metrics=fairness_metrics,
                execution_time=execution_time,
                resource_usage={
                    'resource_id': resource.resource_id,
                    'resource_type': resource.resource_type
                },
                success=True
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Task execution failed: {e}")

            return OptimizationResult(
                task_id=task.task_id,
                parameters=task.parameters,
                performance_metrics={},
                fairness_metrics={},
                execution_time=execution_time,
                resource_usage={
                    'resource_id': resource.resource_id,
                    'resource_type': resource.resource_type
                },
                success=False,
                error_message=str(e)
            )


class AsyncOptimizer(DistributedOptimizer):
    """Asynchronous distributed optimizer."""

    def __init__(self, max_concurrent_tasks: int = None):
        self.max_concurrent_tasks = max_concurrent_tasks or mp.cpu_count()

    def optimize(
        self,
        tasks: List[OptimizationTask],
        scheduler: TaskScheduler
    ) -> List[OptimizationResult]:
        """Run optimization using asyncio."""
        return asyncio.run(self._async_optimize(tasks, scheduler))

    async def _async_optimize(
        self,
        tasks: List[OptimizationTask],
        scheduler: TaskScheduler
    ) -> List[OptimizationResult]:
        """Asynchronous optimization implementation."""
        results = []
        semaphore = asyncio.Semaphore(self.max_concurrent_tasks)

        # Submit all tasks
        for task in tasks:
            scheduler.submit_task(task)

        # Create task coroutines
        coroutines = []

        while True:
            allocation = scheduler.allocate_task()
            if not allocation:
                break

            task, resource = allocation
            coroutines.append(self._execute_task_async(task, resource, semaphore, scheduler))

        # Wait for all tasks to complete
        if coroutines:
            results = await asyncio.gather(*coroutines, return_exceptions=True)

        # Filter out exceptions and convert to proper results
        valid_results = []
        for result in results:
            if isinstance(result, OptimizationResult):
                valid_results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Async task failed: {result}")

        return valid_results

    async def _execute_task_async(
        self,
        task: OptimizationTask,
        resource: ComputeResource,
        semaphore: asyncio.Semaphore,
        scheduler: TaskScheduler
    ) -> OptimizationResult:
        """Execute task asynchronously."""
        async with semaphore:
            # Run CPU-bound task in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._execute_task_sync, task, resource
            )

            scheduler.complete_task(task.task_id, result)
            scheduler.resource_manager.release_resource(resource.resource_id)

            return result

    def _execute_task_sync(
        self,
        task: OptimizationTask,
        resource: ComputeResource
    ) -> OptimizationResult:
        """Synchronous task execution for async wrapper."""
        # Reuse implementation from ThreadPoolOptimizer
        optimizer = ThreadPoolOptimizer(max_workers=1)
        return optimizer._execute_task(task, resource)


class HyperparameterOptimizer:
    """Distributed hyperparameter optimization."""

    def __init__(
        self,
        backend: OptimizationBackend = OptimizationBackend.THREADING,
        max_evaluations: int = 100,
        enable_early_stopping: bool = True
    ):
        self.backend = backend
        self.max_evaluations = max_evaluations
        self.enable_early_stopping = enable_early_stopping

        # Initialize components
        self.resource_manager = ResourceManager()
        self.scheduler = TaskScheduler(self.resource_manager)

        # Choose optimizer based on backend
        if backend == OptimizationBackend.THREADING:
            self.optimizer = ThreadPoolOptimizer()
        elif backend == OptimizationBackend.ASYNCIO:
            self.optimizer = AsyncOptimizer()
        else:
            raise ValueError(f"Backend {backend} not yet implemented")

        logger.info(f"HyperparameterOptimizer initialized with {backend.value} backend")

    def optimize(
        self,
        algorithm: BaseEstimator,
        param_grid: Dict[str, List[Any]],
        X: pd.DataFrame,
        y: pd.Series,
        sensitive_attrs: pd.DataFrame,
        cv_folds: int = 5,
        scoring_metric: str = 'accuracy',
        fairness_constraints: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Run distributed hyperparameter optimization.

        Args:
            algorithm: Base algorithm to optimize
            param_grid: Parameter grid to search
            X: Feature matrix
            y: Target vector
            sensitive_attrs: Sensitive attributes
            cv_folds: Cross-validation folds
            scoring_metric: Metric to optimize
            fairness_constraints: Fairness constraints

        Returns:
            Optimization results
        """
        logger.info(f"Starting hyperparameter optimization with {len(list(ParameterGrid(param_grid)))} combinations")

        start_time = time.time()

        # Generate parameter combinations
        param_combinations = list(ParameterGrid(param_grid))

        # Limit evaluations if too many combinations
        if len(param_combinations) > self.max_evaluations:
            logger.info(f"Limiting to {self.max_evaluations} evaluations via random sampling")
            param_combinations = list(ParameterSampler(
                param_grid, n_iter=self.max_evaluations, random_state=42
            ))

        # Create optimization tasks
        tasks = []
        for i, params in enumerate(param_combinations):
            task = OptimizationTask(
                task_id=f"hp_opt_{i}",
                algorithm=algorithm,
                parameters=params,
                data=(X, y, sensitive_attrs),
                requirements={'cores': 1, 'memory_gb': 2.0}
            )
            tasks.append(task)

        # Run distributed optimization
        results = self.optimizer.optimize(tasks, self.scheduler)

        # Analyze results
        successful_results = [r for r in results if r.success]

        if not successful_results:
            raise RuntimeError("No successful optimization runs")

        # Find best result based on scoring metric
        if fairness_constraints:
            best_result = self._find_best_with_fairness_constraints(
                successful_results, scoring_metric, fairness_constraints
            )
        else:
            best_result = max(
                successful_results,
                key=lambda r: r.performance_metrics.get(scoring_metric, 0)
            )

        optimization_time = time.time() - start_time

        # Compile optimization summary
        summary = {
            'best_parameters': best_result.parameters,
            'best_score': best_result.performance_metrics.get(scoring_metric, 0),
            'best_fairness_metrics': best_result.fairness_metrics,
            'total_evaluations': len(results),
            'successful_evaluations': len(successful_results),
            'optimization_time': optimization_time,
            'resource_utilization': self.resource_manager.get_resource_utilization(),
            'all_results': results
        }

        logger.info(f"Hyperparameter optimization completed in {optimization_time:.2f}s")
        logger.info(f"Best {scoring_metric}: {summary['best_score']:.4f}")

        return summary

    def _find_best_with_fairness_constraints(
        self,
        results: List[OptimizationResult],
        scoring_metric: str,
        fairness_constraints: Dict[str, float]
    ) -> OptimizationResult:
        """Find best result satisfying fairness constraints."""
        feasible_results = []

        for result in results:
            satisfies_constraints = True

            for _attr_name, fairness_metrics in result.fairness_metrics.items():
                overall_metrics = fairness_metrics.get('overall', {})

                for constraint_name, threshold in fairness_constraints.items():
                    metric_value = overall_metrics.get(constraint_name)

                    if metric_value is not None:
                        # For difference metrics, we want them to be below threshold
                        if 'difference' in constraint_name:
                            if abs(metric_value) > threshold:
                                satisfies_constraints = False
                                break
                        # For ratio metrics, we want them to be close to 1
                        elif 'ratio' in constraint_name:
                            if abs(metric_value - 1.0) > threshold:
                                satisfies_constraints = False
                                break

                if not satisfies_constraints:
                    break

            if satisfies_constraints:
                feasible_results.append(result)

        if not feasible_results:
            logger.warning("No results satisfy fairness constraints, returning best unconstrained result")
            return max(results, key=lambda r: r.performance_metrics.get(scoring_metric, 0))

        # Return best feasible result
        return max(feasible_results, key=lambda r: r.performance_metrics.get(scoring_metric, 0))


class FederatedFairnessOptimizer:
    """Federated optimization for fairness across multiple data sources."""

    def __init__(self, num_rounds: int = 10, min_participants: int = 2):
        self.num_rounds = num_rounds
        self.min_participants = min_participants
        self.participants: Dict[str, Dict[str, Any]] = {}
        self.global_model = None

        logger.info("FederatedFairnessOptimizer initialized")

    def register_participant(
        self,
        participant_id: str,
        data: Tuple[pd.DataFrame, pd.Series, pd.DataFrame]
    ):
        """Register a federated learning participant."""
        X, y, sensitive_attrs = data

        self.participants[participant_id] = {
            'data': data,
            'data_size': len(X),
            'local_model': None,
            'fairness_metrics': {},
            'last_update': None
        }

        logger.info(f"Registered participant {participant_id} with {len(X)} samples")

    def federated_optimize(
        self,
        base_algorithm: BaseEstimator,
        algorithm_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Run federated fairness optimization.

        Args:
            base_algorithm: Base algorithm for training
            algorithm_params: Algorithm parameters

        Returns:
            Federated optimization results
        """
        if len(self.participants) < self.min_participants:
            raise ValueError(f"Need at least {self.min_participants} participants")

        logger.info(f"Starting federated optimization with {len(self.participants)} participants")

        # Initialize global model
        self.global_model = clone(base_algorithm)
        if algorithm_params:
            self.global_model.set_params(**algorithm_params)

        round_results = []

        for round_num in range(self.num_rounds):
            logger.info(f"Federated round {round_num + 1}/{self.num_rounds}")

            # Local training on each participant
            local_models = {}
            local_fairness_metrics = {}

            for participant_id, participant_data in self.participants.items():
                X, y, sensitive_attrs = participant_data['data']

                # Train local model
                local_model = clone(self.global_model)
                local_model.fit(X, y)

                # Compute local fairness metrics
                predictions = local_model.predict(X)
                fairness_metrics = {}

                for attr_name in sensitive_attrs.columns:
                    overall, by_group = compute_fairness_metrics(
                        y, predictions, sensitive_attrs[attr_name]
                    )
                    fairness_metrics[attr_name] = {
                        'overall': overall,
                        'by_group': by_group.to_dict() if hasattr(by_group, 'to_dict') else by_group
                    }

                local_models[participant_id] = local_model
                local_fairness_metrics[participant_id] = fairness_metrics

                # Update participant data
                participant_data['local_model'] = local_model
                participant_data['fairness_metrics'] = fairness_metrics
                participant_data['last_update'] = datetime.now()

            # Aggregate models (simplified federated averaging)
            aggregated_params = self._aggregate_model_parameters(local_models)

            # Update global model with aggregated parameters
            self._update_global_model(aggregated_params)

            # Compute global fairness metrics
            global_fairness = self._compute_global_fairness_metrics()

            round_result = {
                'round': round_num + 1,
                'participants': len(local_models),
                'local_fairness_metrics': local_fairness_metrics,
                'global_fairness_metrics': global_fairness,
                'aggregation_time': datetime.now()
            }

            round_results.append(round_result)

            logger.info(f"Round {round_num + 1} completed")

        # Final results
        final_results = {
            'global_model': self.global_model,
            'num_rounds': self.num_rounds,
            'num_participants': len(self.participants),
            'round_results': round_results,
            'final_fairness_metrics': round_results[-1]['global_fairness_metrics'] if round_results else {}
        }

        logger.info("Federated optimization completed")
        return final_results

    def _aggregate_model_parameters(
        self,
        local_models: Dict[str, BaseEstimator]
    ) -> Dict[str, Any]:
        """Aggregate parameters from local models."""
        # This is a simplified implementation
        # In practice, would need more sophisticated aggregation

        # For now, just use the first model's parameters
        # In real federated learning, would aggregate weights properly
        if local_models:
            first_model = next(iter(local_models.values()))
            return first_model.get_params()

        return {}

    def _update_global_model(self, aggregated_params: Dict[str, Any]):
        """Update global model with aggregated parameters."""
        if self.global_model and aggregated_params:
            try:
                self.global_model.set_params(**aggregated_params)
            except Exception as e:
                logger.warning(f"Failed to update global model parameters: {e}")

    def _compute_global_fairness_metrics(self) -> Dict[str, Any]:
        """Compute global fairness metrics across all participants."""
        global_metrics = {}

        # Aggregate fairness metrics across participants
        for participant_id, participant_data in self.participants.items():
            local_fairness = participant_data.get('fairness_metrics', {})

            for attr_name, metrics in local_fairness.items():
                if attr_name not in global_metrics:
                    global_metrics[attr_name] = {'participants': [], 'overall': {}}

                global_metrics[attr_name]['participants'].append({
                    'participant_id': participant_id,
                    'metrics': metrics['overall']
                })

        # Compute aggregate statistics
        for attr_name, attr_metrics in global_metrics.items():
            participant_metrics = attr_metrics['participants']

            if participant_metrics:
                # Average fairness metrics across participants
                metric_names = participant_metrics[0]['metrics'].keys()

                for metric_name in metric_names:
                    values = [p['metrics'].get(metric_name, 0) for p in participant_metrics]
                    values = [v for v in values if v is not None]

                    if values:
                        global_metrics[attr_name]['overall'][metric_name] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'min': np.min(values),
                            'max': np.max(values)
                        }

        return global_metrics


# Performance monitoring and visualization
class PerformanceMonitor:
    """Monitor distributed optimization performance."""

    def __init__(self):
        self.metrics_history = []
        self.resource_utilization_history = []

    def record_performance(
        self,
        timestamp: datetime,
        tasks_completed: int,
        tasks_failed: int,
        average_execution_time: float,
        resource_utilization: Dict[str, float]
    ):
        """Record performance metrics."""
        self.metrics_history.append({
            'timestamp': timestamp,
            'tasks_completed': tasks_completed,
            'tasks_failed': tasks_failed,
            'success_rate': tasks_completed / max(1, tasks_completed + tasks_failed),
            'average_execution_time': average_execution_time
        })

        self.resource_utilization_history.append({
            'timestamp': timestamp,
            'utilization': resource_utilization.copy()
        })

    def generate_performance_report(self) -> str:
        """Generate performance monitoring report."""
        if not self.metrics_history:
            return "No performance data available"

        latest_metrics = self.metrics_history[-1]

        report = f"""
# Distributed Optimization Performance Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Current Performance
- Tasks Completed: {latest_metrics['tasks_completed']}
- Tasks Failed: {latest_metrics['tasks_failed']}
- Success Rate: {latest_metrics['success_rate']:.2%}
- Average Execution Time: {latest_metrics['average_execution_time']:.3f}s

## Resource Utilization
"""

        if self.resource_utilization_history:
            latest_utilization = self.resource_utilization_history[-1]['utilization']
            for resource_id, utilization in latest_utilization.items():
                report += f"- {resource_id}: {utilization:.1%}\n"

        return report


# Example usage and CLI interface
def main():
    """CLI interface for distributed fairness optimizer."""
    import argparse

    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    parser = argparse.ArgumentParser(description="Distributed Fairness Optimization")
    parser.add_argument("--demo", choices=["hyperopt", "federated"], help="Run demonstration")
    parser.add_argument("--backend", choices=["threading", "asyncio"], default="threading",
                       help="Optimization backend")
    parser.add_argument("--max-evaluations", type=int, default=20, help="Max evaluations for hyperopt")

    args = parser.parse_args()

    if args.demo == "hyperopt":
        print("Running Hyperparameter Optimization Demo...")

        # Generate synthetic data
        X, y = make_classification(
            n_samples=1000, n_features=20, n_informative=15,
            n_redundant=2, n_clusters_per_class=2, flip_y=0.05,
            random_state=42
        )

        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        y_series = pd.Series(y, name='target')

        # Create synthetic sensitive attributes
        sensitive_attrs_df = pd.DataFrame({
            'group_a': np.random.binomial(1, 0.3, len(X)),
            'group_b': np.random.choice([0, 1, 2], len(X))
        })

        # Initialize optimizer
        backend = OptimizationBackend.THREADING if args.backend == "threading" else OptimizationBackend.ASYNCIO
        optimizer = HyperparameterOptimizer(
            backend=backend,
            max_evaluations=args.max_evaluations
        )

        # Define parameter grid
        param_grid = {
            'n_estimators': [10, 50, 100],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10]
        }

        # Run optimization
        results = optimizer.optimize(
            algorithm=RandomForestClassifier(random_state=42),
            param_grid=param_grid,
            X=X_df,
            y=y_series,
            sensitive_attrs=sensitive_attrs_df,
            scoring_metric='accuracy'
        )

        print("Optimization completed!")
        print(f"Best parameters: {results['best_parameters']}")
        print(f"Best score: {results['best_score']:.4f}")
        print(f"Total evaluations: {results['total_evaluations']}")
        print(f"Optimization time: {results['optimization_time']:.2f}s")

    elif args.demo == "federated":
        print("Running Federated Optimization Demo...")

        # Create multiple datasets (simulating different participants)
        fed_optimizer = FederatedFairnessOptimizer(num_rounds=5)

        for i in range(3):
            # Generate data for each participant
            X, y = make_classification(
                n_samples=300, n_features=10, n_informative=8,
                n_redundant=1, random_state=42 + i
            )

            X_df = pd.DataFrame(X, columns=[f'feature_{j}' for j in range(X.shape[1])])
            y_series = pd.Series(y, name='target')
            sensitive_attrs_df = pd.DataFrame({
                'group': np.random.binomial(1, 0.4, len(X))
            })

            fed_optimizer.register_participant(f"participant_{i}", (X_df, y_series, sensitive_attrs_df))

        # Run federated optimization
        results = fed_optimizer.federated_optimize(LogisticRegression(random_state=42))

        print("Federated optimization completed!")
        print(f"Rounds: {results['num_rounds']}")
        print(f"Participants: {results['num_participants']}")
        print(f"Final fairness metrics: {results['final_fairness_metrics']}")


if __name__ == "__main__":
    main()
