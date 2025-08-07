"""
Distributed Computing Framework for Fairness Research.

Provides distributed processing capabilities for large-scale fairness experiments,
including parallel algorithm evaluation, distributed hyperparameter optimization,
and multi-node fairness analysis.

Research contributions:
- Distributed fairness evaluation across multiple compute nodes
- Parallel bias detection algorithms for large datasets
- Scalable multi-objective optimization for fairness constraints
- Fault-tolerant experiment execution with automatic recovery
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import warnings
from datetime import datetime
import json
import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
import socket
import os
from pathlib import Path

from ..logging_config import get_logger
from ..fairness_metrics import compute_fairness_metrics

logger = get_logger(__name__)


class TaskStatus(Enum):
    """Status of distributed tasks."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ComputeNodeType(Enum):
    """Types of compute nodes."""
    CPU = "cpu"
    GPU = "gpu"
    HIGH_MEMORY = "high_memory"
    NETWORK_OPTIMIZED = "network_optimized"


class DistributionStrategy(Enum):
    """Distribution strategies for tasks."""
    ROUND_ROBIN = "round_robin"
    LOAD_BALANCED = "load_balanced"
    RESOURCE_AWARE = "resource_aware"
    LOCALITY_AWARE = "locality_aware"


@dataclass
class ComputeNode:
    """Represents a compute node in the cluster."""
    node_id: str
    hostname: str
    port: int
    node_type: ComputeNodeType
    cpu_cores: int
    memory_gb: float
    gpu_count: int = 0
    current_load: float = 0.0
    is_available: bool = True
    capabilities: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'node_id': self.node_id,
            'hostname': self.hostname,
            'port': self.port,
            'node_type': self.node_type.value,
            'cpu_cores': self.cpu_cores,
            'memory_gb': self.memory_gb,
            'gpu_count': self.gpu_count,
            'current_load': self.current_load,
            'is_available': self.is_available,
            'capabilities': self.capabilities
        }
    
    def get_resource_score(self, task_requirements: Dict[str, Any]) -> float:
        """Calculate resource fitness score for a task."""
        score = 0.0
        
        # CPU requirement
        cpu_req = task_requirements.get('cpu_cores', 1)
        if self.cpu_cores >= cpu_req:
            score += 1.0 - (cpu_req / self.cpu_cores)
        else:
            return 0.0  # Cannot handle task
        
        # Memory requirement
        memory_req = task_requirements.get('memory_gb', 1.0)
        if self.memory_gb >= memory_req:
            score += 1.0 - (memory_req / self.memory_gb)
        else:
            return 0.0  # Cannot handle task
        
        # GPU requirement
        gpu_req = task_requirements.get('gpu_count', 0)
        if gpu_req > 0 and self.gpu_count >= gpu_req:
            score += 2.0  # Bonus for GPU availability
        elif gpu_req > 0 and self.gpu_count < gpu_req:
            return 0.0  # Cannot handle GPU task
        
        # Load penalty
        score *= (1.0 - min(self.current_load, 0.9))
        
        # Availability check
        if not self.is_available:
            return 0.0
        
        return score


@dataclass
class DistributedTask:
    """Represents a distributed task."""
    task_id: str
    task_type: str
    function_name: str
    args: List[Any]
    kwargs: Dict[str, Any]
    requirements: Dict[str, Any]
    priority: int = 1
    max_retries: int = 3
    timeout_seconds: int = 3600
    status: TaskStatus = TaskStatus.PENDING
    assigned_node: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'function_name': self.function_name,
            'requirements': self.requirements,
            'priority': self.priority,
            'max_retries': self.max_retries,
            'timeout_seconds': self.timeout_seconds,
            'status': self.status.value,
            'assigned_node': self.assigned_node,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error': self.error,
            'retry_count': self.retry_count
        }


class TaskScheduler:
    """
    Intelligent task scheduler for distributed fairness computations.
    
    Implements resource-aware scheduling with load balancing and
    fault tolerance for fairness research workloads.
    """
    
    def __init__(
        self,
        strategy: DistributionStrategy = DistributionStrategy.RESOURCE_AWARE,
        max_concurrent_tasks: int = 10
    ):
        """
        Initialize task scheduler.
        
        Args:
            strategy: Distribution strategy to use
            max_concurrent_tasks: Maximum concurrent tasks per node
        """
        self.strategy = strategy
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # Task queues
        self.pending_tasks: queue.PriorityQueue = queue.PriorityQueue()
        self.running_tasks: Dict[str, DistributedTask] = {}
        self.completed_tasks: Dict[str, DistributedTask] = {}
        self.failed_tasks: Dict[str, DistributedTask] = {}
        
        # Compute nodes
        self.compute_nodes: Dict[str, ComputeNode] = {}
        self.node_task_counts: Dict[str, int] = {}
        
        # Scheduling thread
        self.scheduler_thread: Optional[threading.Thread] = None
        self.is_running = False
        
        logger.info(f"TaskScheduler initialized with strategy: {strategy.value}")
    
    def add_compute_node(self, node: ComputeNode):
        """Add a compute node to the cluster."""
        self.compute_nodes[node.node_id] = node
        self.node_task_counts[node.node_id] = 0
        logger.info(f"Added compute node: {node.node_id} ({node.hostname})")
    
    def remove_compute_node(self, node_id: str):
        """Remove a compute node from the cluster."""
        if node_id in self.compute_nodes:
            # Reassign running tasks
            tasks_to_reassign = [
                task for task in self.running_tasks.values()
                if task.assigned_node == node_id
            ]
            
            for task in tasks_to_reassign:
                self._reassign_task(task)
            
            del self.compute_nodes[node_id]
            del self.node_task_counts[node_id]
            logger.info(f"Removed compute node: {node_id}")
    
    def submit_task(self, task: DistributedTask) -> str:
        """Submit a task for distributed execution."""
        # Add to pending queue with priority
        priority = -task.priority  # Negative for max heap behavior
        self.pending_tasks.put((priority, time.time(), task))
        
        logger.debug(f"Submitted task: {task.task_id}")
        return task.task_id
    
    def start_scheduler(self):
        """Start the task scheduler."""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Task scheduler started")
    
    def stop_scheduler(self):
        """Stop the task scheduler."""
        self.is_running = False
        
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5.0)
        
        logger.info("Task scheduler stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.is_running:
            try:
                # Process pending tasks
                self._process_pending_tasks()
                
                # Check running tasks for completion/timeout
                self._check_running_tasks()
                
                # Update node loads
                self._update_node_loads()
                
                time.sleep(1.0)  # Schedule every second
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(5.0)  # Back off on error
    
    def _process_pending_tasks(self):
        """Process pending tasks for assignment."""
        while not self.pending_tasks.empty():
            try:
                _, _, task = self.pending_tasks.get_nowait()
                
                # Find best node for task
                best_node = self._select_node_for_task(task)
                
                if best_node:
                    self._assign_task_to_node(task, best_node)
                else:
                    # No suitable node available, put back in queue
                    priority = -task.priority
                    self.pending_tasks.put((priority, time.time(), task))
                    break  # Wait for next cycle
                    
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error processing pending task: {e}")
    
    def _check_running_tasks(self):
        """Check running tasks for completion or timeout."""
        completed_tasks = []
        
        for task_id, task in self.running_tasks.items():
            # Check for timeout
            if task.started_at:
                elapsed = (datetime.now() - task.started_at).total_seconds()
                if elapsed > task.timeout_seconds:
                    self._handle_task_timeout(task)
                    completed_tasks.append(task_id)
                    continue
            
            # Check for completion (in real implementation, this would be via messaging)
            # For demonstration, we'll simulate task completion
            if self._simulate_task_completion(task):
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                self.completed_tasks[task_id] = task
                completed_tasks.append(task_id)
                
                # Update node task count
                if task.assigned_node:
                    self.node_task_counts[task.assigned_node] -= 1
        
        # Remove completed tasks from running
        for task_id in completed_tasks:
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
    
    def _select_node_for_task(self, task: DistributedTask) -> Optional[ComputeNode]:
        """Select the best node for a task based on strategy."""
        available_nodes = [
            node for node in self.compute_nodes.values()
            if node.is_available and 
               self.node_task_counts.get(node.node_id, 0) < self.max_concurrent_tasks
        ]
        
        if not available_nodes:
            return None
        
        if self.strategy == DistributionStrategy.ROUND_ROBIN:
            return min(available_nodes, key=lambda n: self.node_task_counts.get(n.node_id, 0))
        
        elif self.strategy == DistributionStrategy.LOAD_BALANCED:
            return min(available_nodes, key=lambda n: n.current_load)
        
        elif self.strategy == DistributionStrategy.RESOURCE_AWARE:
            # Select node with highest resource fitness score
            node_scores = [
                (node, node.get_resource_score(task.requirements))
                for node in available_nodes
            ]
            
            # Filter nodes that can handle the task
            capable_nodes = [(node, score) for node, score in node_scores if score > 0]
            
            if not capable_nodes:
                return None
            
            return max(capable_nodes, key=lambda x: x[1])[0]
        
        else:
            # Default to first available
            return available_nodes[0]
    
    def _assign_task_to_node(self, task: DistributedTask, node: ComputeNode):
        """Assign a task to a specific node."""
        task.assigned_node = node.node_id
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        
        self.running_tasks[task.task_id] = task
        self.node_task_counts[node.node_id] = self.node_task_counts.get(node.node_id, 0) + 1
        
        # In real implementation, would send task to node via RPC/messaging
        self._simulate_task_execution(task, node)
        
        logger.debug(f"Assigned task {task.task_id} to node {node.node_id}")
    
    def _simulate_task_execution(self, task: DistributedTask, node: ComputeNode):
        """Simulate task execution (placeholder for real distributed execution)."""
        # In a real implementation, this would send the task to the node
        # via RPC, REST API, or message queue
        logger.debug(f"Simulating execution of task {task.task_id} on node {node.node_id}")
    
    def _simulate_task_completion(self, task: DistributedTask) -> bool:
        """Simulate task completion (placeholder)."""
        # For demonstration, randomly complete tasks after some time
        if task.started_at:
            elapsed = (datetime.now() - task.started_at).total_seconds()
            # Complete tasks after 5-15 seconds for demo
            completion_time = 5 + (hash(task.task_id) % 10)
            return elapsed > completion_time
        return False
    
    def _handle_task_timeout(self, task: DistributedTask):
        """Handle task timeout."""
        logger.warning(f"Task {task.task_id} timed out after {task.timeout_seconds}s")
        
        task.status = TaskStatus.FAILED
        task.error = f"Timeout after {task.timeout_seconds} seconds"
        task.completed_at = datetime.now()
        
        # Update node task count
        if task.assigned_node:
            self.node_task_counts[task.assigned_node] -= 1
        
        # Retry if within retry limit
        if task.retry_count < task.max_retries:
            task.retry_count += 1
            task.status = TaskStatus.PENDING
            task.assigned_node = None
            task.started_at = None
            task.error = None
            
            # Re-queue for retry
            priority = -task.priority
            self.pending_tasks.put((priority, time.time(), task))
            
            logger.info(f"Re-queued task {task.task_id} for retry ({task.retry_count}/{task.max_retries})")
        else:
            self.failed_tasks[task.task_id] = task
            logger.error(f"Task {task.task_id} failed after {task.max_retries} retries")
    
    def _reassign_task(self, task: DistributedTask):
        """Reassign a task to a different node."""
        old_node = task.assigned_node
        task.assigned_node = None
        task.status = TaskStatus.PENDING
        task.started_at = None
        
        # Re-queue task
        priority = -task.priority
        self.pending_tasks.put((priority, time.time(), task))
        
        logger.info(f"Reassigned task {task.task_id} from node {old_node}")
    
    def _update_node_loads(self):
        """Update current load for all nodes."""
        for node_id, node in self.compute_nodes.items():
            task_count = self.node_task_counts.get(node_id, 0)
            node.current_load = min(1.0, task_count / self.max_concurrent_tasks)
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get current cluster status."""
        return {
            'nodes': {node_id: node.to_dict() for node_id, node in self.compute_nodes.items()},
            'task_counts': {
                'pending': self.pending_tasks.qsize(),
                'running': len(self.running_tasks),
                'completed': len(self.completed_tasks),
                'failed': len(self.failed_tasks)
            },
            'node_task_counts': self.node_task_counts,
            'scheduler_running': self.is_running
        }


class ComputeCluster:
    """
    Manages a cluster of compute nodes for distributed fairness research.
    
    Provides high-level interface for cluster management and
    distributed task execution.
    """
    
    def __init__(self, cluster_name: str = "fairness_cluster"):
        """
        Initialize compute cluster.
        
        Args:
            cluster_name: Name of the cluster
        """
        self.cluster_name = cluster_name
        self.scheduler = TaskScheduler()
        self.task_results: Dict[str, Any] = {}
        
        # Initialize with local node
        self._add_local_node()
        
        logger.info(f"ComputeCluster '{cluster_name}' initialized")
    
    def _add_local_node(self):
        """Add local machine as a compute node."""
        local_node = ComputeNode(
            node_id="local",
            hostname=socket.gethostname(),
            port=8080,
            node_type=ComputeNodeType.CPU,
            cpu_cores=multiprocessing.cpu_count(),
            memory_gb=8.0,  # Simplified
            capabilities=["fairness_computation", "bias_detection", "optimization"]
        )
        
        self.scheduler.add_compute_node(local_node)
    
    def add_remote_node(
        self,
        node_id: str,
        hostname: str,
        port: int,
        node_type: ComputeNodeType,
        cpu_cores: int,
        memory_gb: float,
        gpu_count: int = 0
    ):
        """Add a remote compute node to the cluster."""
        node = ComputeNode(
            node_id=node_id,
            hostname=hostname,
            port=port,
            node_type=node_type,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            gpu_count=gpu_count,
            capabilities=["fairness_computation", "bias_detection"]
        )
        
        self.scheduler.add_compute_node(node)
        logger.info(f"Added remote node: {node_id}")
    
    def start_cluster(self):
        """Start the compute cluster."""
        self.scheduler.start_scheduler()
        logger.info(f"Cluster '{self.cluster_name}' started")
    
    def stop_cluster(self):
        """Stop the compute cluster."""
        self.scheduler.stop_scheduler()
        logger.info(f"Cluster '{self.cluster_name}' stopped")
    
    def submit_fairness_evaluation(
        self,
        algorithm_name: str,
        algorithm_params: Dict[str, Any],
        dataset_partition: Dict[str, Any],
        evaluation_config: Dict[str, Any]
    ) -> str:
        """Submit a fairness evaluation task."""
        task = DistributedTask(
            task_id=f"fairness_eval_{algorithm_name}_{int(time.time())}",
            task_type="fairness_evaluation",
            function_name="evaluate_fairness_distributed",
            args=[algorithm_name, algorithm_params, dataset_partition],
            kwargs=evaluation_config,
            requirements={
                'cpu_cores': 2,
                'memory_gb': 4.0,
                'timeout_seconds': 1800
            },
            priority=2
        )
        
        return self.scheduler.submit_task(task)
    
    def submit_bias_detection(
        self,
        data_partition: Dict[str, Any],
        detection_config: Dict[str, Any]
    ) -> str:
        """Submit a bias detection task."""
        task = DistributedTask(
            task_id=f"bias_detect_{int(time.time())}",
            task_type="bias_detection",
            function_name="detect_bias_distributed",
            args=[data_partition],
            kwargs=detection_config,
            requirements={
                'cpu_cores': 1,
                'memory_gb': 2.0,
                'timeout_seconds': 900
            },
            priority=3
        )
        
        return self.scheduler.submit_task(task)
    
    def submit_hyperparameter_optimization(
        self,
        algorithm_name: str,
        param_space: Dict[str, Any],
        dataset_config: Dict[str, Any],
        optimization_config: Dict[str, Any]
    ) -> str:
        """Submit hyperparameter optimization task."""
        task = DistributedTask(
            task_id=f"hyperparam_opt_{algorithm_name}_{int(time.time())}",
            task_type="hyperparameter_optimization",
            function_name="optimize_hyperparameters_distributed",
            args=[algorithm_name, param_space, dataset_config],
            kwargs=optimization_config,
            requirements={
                'cpu_cores': 4,
                'memory_gb': 8.0,
                'timeout_seconds': 3600
            },
            priority=1  # Highest priority
        )
        
        return self.scheduler.submit_task(task)
    
    def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get result of a completed task."""
        # Check completed tasks
        if task_id in self.scheduler.completed_tasks:
            task = self.scheduler.completed_tasks[task_id]
            return {
                'status': task.status.value,
                'result': task.result,
                'execution_time': (
                    task.completed_at - task.started_at
                ).total_seconds() if task.started_at and task.completed_at else None,
                'assigned_node': task.assigned_node
            }
        
        # Check failed tasks
        if task_id in self.scheduler.failed_tasks:
            task = self.scheduler.failed_tasks[task_id]
            return {
                'status': task.status.value,
                'error': task.error,
                'retry_count': task.retry_count,
                'assigned_node': task.assigned_node
            }
        
        # Check running tasks
        if task_id in self.scheduler.running_tasks:
            task = self.scheduler.running_tasks[task_id]
            return {
                'status': task.status.value,
                'assigned_node': task.assigned_node,
                'started_at': task.started_at.isoformat() if task.started_at else None
            }
        
        return None
    
    def wait_for_task(self, task_id: str, timeout: int = 3600) -> Dict[str, Any]:
        """Wait for a task to complete and return results."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            result = self.get_task_result(task_id)
            
            if result and result['status'] in ['completed', 'failed']:
                return result
            
            time.sleep(1.0)
        
        raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")
    
    def get_cluster_metrics(self) -> Dict[str, Any]:
        """Get comprehensive cluster metrics."""
        status = self.scheduler.get_cluster_status()
        
        # Calculate additional metrics
        total_tasks = (
            status['task_counts']['pending'] +
            status['task_counts']['running'] +
            status['task_counts']['completed'] +
            status['task_counts']['failed']
        )
        
        success_rate = (
            status['task_counts']['completed'] / total_tasks
            if total_tasks > 0 else 0.0
        )
        
        # Node utilization
        node_utilizations = []
        for node_id, node_info in status['nodes'].items():
            task_count = status['node_task_counts'].get(node_id, 0)
            utilization = task_count / self.scheduler.max_concurrent_tasks
            node_utilizations.append(utilization)
        
        avg_utilization = np.mean(node_utilizations) if node_utilizations else 0.0
        
        return {
            'cluster_name': self.cluster_name,
            'total_nodes': len(status['nodes']),
            'total_tasks': total_tasks,
            'success_rate': success_rate,
            'average_node_utilization': avg_utilization,
            'task_distribution': status['task_counts'],
            'node_details': status['nodes'],
            'scheduler_running': status['scheduler_running']
        }


class DistributedFairnessFramework:
    """
    High-level framework for distributed fairness research.
    
    Provides easy-to-use interface for conducting large-scale
    fairness experiments across distributed compute resources.
    """
    
    def __init__(
        self,
        cluster_name: str = "fairness_research_cluster",
        enable_auto_scaling: bool = False
    ):
        """
        Initialize distributed fairness framework.
        
        Args:
            cluster_name: Name of the compute cluster
            enable_auto_scaling: Enable automatic scaling of compute resources
        """
        self.cluster_name = cluster_name
        self.enable_auto_scaling = enable_auto_scaling
        
        # Initialize cluster
        self.cluster = ComputeCluster(cluster_name)
        
        # Experiment tracking
        self.experiments: Dict[str, Dict[str, Any]] = {}
        self.active_experiments: Set[str] = set()
        
        logger.info(f"DistributedFairnessFramework initialized: {cluster_name}")
    
    def start_framework(self):
        """Start the distributed framework."""
        self.cluster.start_cluster()
        logger.info("Distributed fairness framework started")
    
    def stop_framework(self):
        """Stop the distributed framework."""
        self.cluster.stop_cluster()
        logger.info("Distributed fairness framework stopped")
    
    def run_distributed_benchmark(
        self,
        algorithms: List[Dict[str, Any]],
        datasets: List[Dict[str, Any]],
        evaluation_metrics: List[str],
        experiment_name: str = "distributed_benchmark"
    ) -> str:
        """
        Run a distributed fairness benchmark across algorithms and datasets.
        
        Args:
            algorithms: List of algorithm configurations
            datasets: List of dataset configurations
            evaluation_metrics: Metrics to evaluate
            experiment_name: Name for the experiment
            
        Returns:
            Experiment ID
        """
        experiment_id = f"{experiment_name}_{int(time.time())}"
        
        logger.info(f"Starting distributed benchmark: {experiment_id}")
        
        # Create experiment tracking
        self.experiments[experiment_id] = {
            'name': experiment_name,
            'algorithms': algorithms,
            'datasets': datasets,
            'metrics': evaluation_metrics,
            'task_ids': [],
            'started_at': datetime.now().isoformat(),
            'status': 'running'
        }
        
        self.active_experiments.add(experiment_id)
        
        # Submit evaluation tasks for each algorithm-dataset pair
        task_ids = []
        
        for algorithm in algorithms:
            for dataset in datasets:
                task_id = self.cluster.submit_fairness_evaluation(
                    algorithm_name=algorithm['name'],
                    algorithm_params=algorithm['params'],
                    dataset_partition=dataset,
                    evaluation_config={
                        'metrics': evaluation_metrics,
                        'experiment_id': experiment_id
                    }
                )
                
                task_ids.append(task_id)
        
        self.experiments[experiment_id]['task_ids'] = task_ids
        
        logger.info(f"Submitted {len(task_ids)} evaluation tasks for experiment {experiment_id}")
        return experiment_id
    
    def run_distributed_bias_scan(
        self,
        dataset_config: Dict[str, Any],
        detection_methods: List[str],
        experiment_name: str = "bias_scan"
    ) -> str:
        """
        Run distributed bias detection scan across a dataset.
        
        Args:
            dataset_config: Dataset configuration
            detection_methods: Bias detection methods to use
            experiment_name: Name for the experiment
            
        Returns:
            Experiment ID
        """
        experiment_id = f"{experiment_name}_{int(time.time())}"
        
        logger.info(f"Starting distributed bias scan: {experiment_id}")
        
        # Partition dataset for distributed processing
        partitions = self._partition_dataset(dataset_config, num_partitions=4)
        
        # Create experiment tracking
        self.experiments[experiment_id] = {
            'name': experiment_name,
            'dataset': dataset_config,
            'detection_methods': detection_methods,
            'partitions': len(partitions),
            'task_ids': [],
            'started_at': datetime.now().isoformat(),
            'status': 'running'
        }
        
        self.active_experiments.add(experiment_id)
        
        # Submit bias detection tasks for each partition
        task_ids = []
        
        for i, partition in enumerate(partitions):
            for method in detection_methods:
                task_id = self.cluster.submit_bias_detection(
                    data_partition=partition,
                    detection_config={
                        'method': method,
                        'partition_id': i,
                        'experiment_id': experiment_id
                    }
                )
                
                task_ids.append(task_id)
        
        self.experiments[experiment_id]['task_ids'] = task_ids
        
        logger.info(f"Submitted {len(task_ids)} bias detection tasks for experiment {experiment_id}")
        return experiment_id
    
    def run_distributed_hyperparameter_optimization(
        self,
        algorithm_name: str,
        parameter_space: Dict[str, Any],
        dataset_config: Dict[str, Any],
        optimization_objective: str = "fairness_accuracy_trade_off",
        max_evaluations: int = 100,
        experiment_name: str = "hyperparameter_opt"
    ) -> str:
        """
        Run distributed hyperparameter optimization for fairness algorithms.
        
        Args:
            algorithm_name: Name of algorithm to optimize
            parameter_space: Parameter space to explore
            dataset_config: Dataset configuration
            optimization_objective: Objective function to optimize
            max_evaluations: Maximum number of parameter combinations to evaluate
            experiment_name: Name for the experiment
            
        Returns:
            Experiment ID
        """
        experiment_id = f"{experiment_name}_{algorithm_name}_{int(time.time())}"
        
        logger.info(f"Starting distributed hyperparameter optimization: {experiment_id}")
        
        # Create experiment tracking
        self.experiments[experiment_id] = {
            'name': experiment_name,
            'algorithm': algorithm_name,
            'parameter_space': parameter_space,
            'dataset': dataset_config,
            'objective': optimization_objective,
            'max_evaluations': max_evaluations,
            'task_ids': [],
            'started_at': datetime.now().isoformat(),
            'status': 'running'
        }
        
        self.active_experiments.add(experiment_id)
        
        # Submit hyperparameter optimization task
        task_id = self.cluster.submit_hyperparameter_optimization(
            algorithm_name=algorithm_name,
            param_space=parameter_space,
            dataset_config=dataset_config,
            optimization_config={
                'objective': optimization_objective,
                'max_evaluations': max_evaluations,
                'experiment_id': experiment_id
            }
        )
        
        self.experiments[experiment_id]['task_ids'] = [task_id]
        
        logger.info(f"Submitted hyperparameter optimization task: {task_id}")
        return experiment_id
    
    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get status of a distributed experiment."""
        if experiment_id not in self.experiments:
            return {'error': f'Experiment {experiment_id} not found'}
        
        experiment = self.experiments[experiment_id]
        task_ids = experiment['task_ids']
        
        # Get status of all tasks
        task_statuses = {}
        for task_id in task_ids:
            result = self.cluster.get_task_result(task_id)
            if result:
                task_statuses[task_id] = result['status']
            else:
                task_statuses[task_id] = 'pending'
        
        # Calculate overall progress
        completed_tasks = sum(1 for status in task_statuses.values() if status == 'completed')
        failed_tasks = sum(1 for status in task_statuses.values() if status == 'failed')
        running_tasks = sum(1 for status in task_statuses.values() if status == 'running')
        pending_tasks = sum(1 for status in task_statuses.values() if status == 'pending')
        
        total_tasks = len(task_ids)
        progress = (completed_tasks / total_tasks) if total_tasks > 0 else 0.0
        
        # Update experiment status
        if completed_tasks + failed_tasks == total_tasks:
            experiment['status'] = 'completed'
            experiment['completed_at'] = datetime.now().isoformat()
            self.active_experiments.discard(experiment_id)
        
        return {
            'experiment_id': experiment_id,
            'name': experiment['name'],
            'status': experiment['status'],
            'progress': progress,
            'task_summary': {
                'total': total_tasks,
                'completed': completed_tasks,
                'failed': failed_tasks,
                'running': running_tasks,
                'pending': pending_tasks
            },
            'task_statuses': task_statuses,
            'started_at': experiment['started_at'],
            'completed_at': experiment.get('completed_at')
        }
    
    def wait_for_experiment(self, experiment_id: str, timeout: int = 7200) -> Dict[str, Any]:
        """Wait for an experiment to complete."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_experiment_status(experiment_id)
            
            if status.get('status') == 'completed':
                return self.get_experiment_results(experiment_id)
            
            time.sleep(5.0)  # Check every 5 seconds
        
        raise TimeoutError(f"Experiment {experiment_id} did not complete within {timeout} seconds")
    
    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get comprehensive results from a completed experiment."""
        if experiment_id not in self.experiments:
            return {'error': f'Experiment {experiment_id} not found'}
        
        experiment = self.experiments[experiment_id]
        task_ids = experiment['task_ids']
        
        # Collect results from all tasks
        task_results = {}
        for task_id in task_ids:
            result = self.cluster.get_task_result(task_id)
            if result:
                task_results[task_id] = result
        
        # Aggregate results based on experiment type
        if 'algorithms' in experiment:  # Benchmark experiment
            results = self._aggregate_benchmark_results(task_results, experiment)
        elif 'detection_methods' in experiment:  # Bias scan experiment
            results = self._aggregate_bias_scan_results(task_results, experiment)
        elif 'parameter_space' in experiment:  # Hyperparameter optimization
            results = self._aggregate_optimization_results(task_results, experiment)
        else:
            results = {'task_results': task_results}
        
        results.update({
            'experiment_id': experiment_id,
            'experiment_config': experiment,
            'total_tasks': len(task_ids),
            'successful_tasks': len([r for r in task_results.values() if r.get('status') == 'completed'])
        })
        
        return results
    
    def _partition_dataset(self, dataset_config: Dict[str, Any], num_partitions: int = 4) -> List[Dict[str, Any]]:
        """Partition dataset for distributed processing."""
        # Simplified partitioning - in practice would handle actual data partitioning
        partitions = []
        
        for i in range(num_partitions):
            partition = dataset_config.copy()
            partition['partition_id'] = i
            partition['partition_count'] = num_partitions
            partitions.append(partition)
        
        return partitions
    
    def _aggregate_benchmark_results(
        self, 
        task_results: Dict[str, Any], 
        experiment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Aggregate results from benchmark experiment."""
        # Simplified aggregation - in practice would process actual results
        algorithms = experiment['algorithms']
        datasets = experiment['datasets']
        
        aggregated = {
            'benchmark_summary': {
                'algorithms_tested': len(algorithms),
                'datasets_tested': len(datasets),
                'total_evaluations': len(algorithms) * len(datasets)
            },
            'algorithm_performance': {},
            'dataset_analysis': {}
        }
        
        return aggregated
    
    def _aggregate_bias_scan_results(
        self, 
        task_results: Dict[str, Any], 
        experiment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Aggregate results from bias scan experiment."""
        detection_methods = experiment['detection_methods']
        partitions = experiment['partitions']
        
        aggregated = {
            'bias_scan_summary': {
                'detection_methods': len(detection_methods),
                'data_partitions': partitions,
                'total_scans': len(detection_methods) * partitions
            },
            'bias_findings': {},
            'method_comparison': {}
        }
        
        return aggregated
    
    def _aggregate_optimization_results(
        self, 
        task_results: Dict[str, Any], 
        experiment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Aggregate results from hyperparameter optimization."""
        algorithm = experiment['algorithm']
        objective = experiment['objective']
        
        aggregated = {
            'optimization_summary': {
                'algorithm': algorithm,
                'objective': objective,
                'evaluations_completed': len([r for r in task_results.values() if r.get('status') == 'completed'])
            },
            'best_parameters': {},
            'optimization_history': {}
        }
        
        return aggregated
    
    def get_framework_metrics(self) -> Dict[str, Any]:
        """Get comprehensive framework metrics."""
        cluster_metrics = self.cluster.get_cluster_metrics()
        
        # Add experiment metrics
        total_experiments = len(self.experiments)
        active_experiments = len(self.active_experiments)
        completed_experiments = len([e for e in self.experiments.values() if e.get('status') == 'completed'])
        
        framework_metrics = {
            'framework_name': 'DistributedFairnessFramework',
            'cluster_metrics': cluster_metrics,
            'experiment_metrics': {
                'total_experiments': total_experiments,
                'active_experiments': active_experiments,
                'completed_experiments': completed_experiments
            },
            'active_experiment_list': list(self.active_experiments)
        }
        
        return framework_metrics


# Example usage and CLI interface
def main():
    """CLI interface for distributed computing framework."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Distributed Fairness Computing Framework")
    parser.add_argument("--demo", action="store_true", help="Run demonstration")
    parser.add_argument("--cluster-name", default="demo_cluster", help="Cluster name")
    parser.add_argument("--nodes", type=int, default=2, help="Number of simulated nodes")
    
    args = parser.parse_args()
    
    if args.demo:
        print("🚀 Starting Distributed Fairness Framework Demo")
        
        # Initialize framework
        framework = DistributedFairnessFramework(
            cluster_name=args.cluster_name,
            enable_auto_scaling=False
        )
        
        # Add simulated remote nodes
        for i in range(1, args.nodes):
            framework.cluster.add_remote_node(
                node_id=f"node_{i}",
                hostname=f"compute-node-{i}",
                port=8080 + i,
                node_type=ComputeNodeType.CPU,
                cpu_cores=4,
                memory_gb=8.0
            )
        
        # Start framework
        framework.start_framework()
        
        print(f"✅ Framework started with {args.nodes} nodes")
        
        # Demo 1: Distributed benchmark
        print("\n📊 Demo 1: Distributed Benchmark")
        
        algorithms = [
            {'name': 'LogisticRegression', 'params': {'max_iter': 1000}},
            {'name': 'RandomForest', 'params': {'n_estimators': 100}}
        ]
        
        datasets = [
            {'name': 'adult_income', 'size': 10000},
            {'name': 'german_credit', 'size': 1000}
        ]
        
        benchmark_id = framework.run_distributed_benchmark(
            algorithms=algorithms,
            datasets=datasets,
            evaluation_metrics=['accuracy', 'demographic_parity', 'equalized_odds'],
            experiment_name="demo_benchmark"
        )
        
        print(f"   Submitted benchmark experiment: {benchmark_id}")
        
        # Demo 2: Distributed bias scan
        print("\n🔍 Demo 2: Distributed Bias Scan")
        
        bias_scan_id = framework.run_distributed_bias_scan(
            dataset_config={'name': 'synthetic_biased', 'size': 50000},
            detection_methods=['statistical', 'ml_based', 'causal'],
            experiment_name="demo_bias_scan"
        )
        
        print(f"   Submitted bias scan experiment: {bias_scan_id}")
        
        # Demo 3: Hyperparameter optimization
        print("\n⚙️ Demo 3: Distributed Hyperparameter Optimization")
        
        hyperparam_id = framework.run_distributed_hyperparameter_optimization(
            algorithm_name="FairClassifier",
            parameter_space={
                'C': [0.1, 1.0, 10.0],
                'fairness_constraint': [0.1, 0.05, 0.01]
            },
            dataset_config={'name': 'adult_income', 'size': 10000},
            max_evaluations=20,
            experiment_name="demo_hyperparam_opt"
        )
        
        print(f"   Submitted hyperparameter optimization: {hyperparam_id}")
        
        # Monitor experiments
        print("\n⏳ Monitoring experiment progress...")
        
        experiments = [benchmark_id, bias_scan_id, hyperparam_id]
        
        for i in range(30):  # Monitor for 30 seconds
            print(f"\n--- Status Update {i+1} ---")
            
            for exp_id in experiments:
                status = framework.get_experiment_status(exp_id)
                print(f"   {status['name']}: {status['progress']:.1%} complete "
                      f"({status['task_summary']['completed']}/{status['task_summary']['total']} tasks)")
            
            # Show cluster metrics
            metrics = framework.get_framework_metrics()
            print(f"   Cluster utilization: {metrics['cluster_metrics']['average_node_utilization']:.1%}")
            
            time.sleep(2.0)
            
            # Check if all completed
            all_completed = all(
                framework.get_experiment_status(exp_id)['status'] == 'completed'
                for exp_id in experiments
            )
            
            if all_completed:
                print("\n✅ All experiments completed!")
                break
        
        # Show final results
        print("\n📋 Final Results Summary:")
        
        for exp_id in experiments:
            results = framework.get_experiment_results(exp_id)
            print(f"\n   {results['experiment_config']['name']}:")
            print(f"     - Total tasks: {results['total_tasks']}")
            print(f"     - Successful: {results['successful_tasks']}")
            
            if 'benchmark_summary' in results:
                summary = results['benchmark_summary']
                print(f"     - Algorithms tested: {summary['algorithms_tested']}")
                print(f"     - Datasets tested: {summary['datasets_tested']}")
            
            elif 'bias_scan_summary' in results:
                summary = results['bias_scan_summary']
                print(f"     - Detection methods: {summary['detection_methods']}")
                print(f"     - Data partitions: {summary['data_partitions']}")
            
            elif 'optimization_summary' in results:
                summary = results['optimization_summary']
                print(f"     - Algorithm optimized: {summary['algorithm']}")
                print(f"     - Evaluations: {summary['evaluations_completed']}")
        
        # Stop framework
        framework.stop_framework()
        print(f"\n🔴 Framework stopped")
        print("Demo completed successfully! 🎉")


if __name__ == "__main__":
    main()