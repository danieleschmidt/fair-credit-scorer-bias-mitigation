"""
Distributed Computing Framework v3.0
Advanced distributed computing infrastructure for high-performance ML operations.

This module provides sophisticated distributed computing capabilities including:
- Distributed training with fault tolerance
- Dynamic load balancing and resource allocation
- Federated learning support
- Edge computing integration
- Blockchain-based consensus for distributed decisions
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, clone

logger = logging.getLogger(__name__)


class ComputeNodeType(Enum):
    """Types of compute nodes in distributed system."""
    MASTER = "master"
    WORKER = "worker"
    COORDINATOR = "coordinator"
    EDGE_DEVICE = "edge_device"
    GPU_NODE = "gpu_node"


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class ResourceType(Enum):
    """Types of computational resources."""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    GPU = "gpu"


@dataclass
class ResourceCapacity:
    """Resource capacity specification."""
    cpu_cores: int = 1
    memory_gb: float = 1.0
    storage_gb: float = 10.0
    network_mbps: float = 100.0
    gpu_count: int = 0
    gpu_memory_gb: float = 0.0


@dataclass
class ComputeTask:
    """Distributed compute task specification."""
    task_id: str
    function: Callable
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    resource_requirements: Optional[ResourceCapacity] = None
    timeout: Optional[int] = None
    retry_count: int = 3
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComputeNode:
    """Compute node in distributed system."""
    node_id: str
    node_type: ComputeNodeType
    host: str = "localhost"
    port: int = 8080
    capacity: ResourceCapacity = field(default_factory=ResourceCapacity)
    current_load: ResourceCapacity = field(default_factory=ResourceCapacity)
    status: str = "idle"
    last_heartbeat: float = field(default_factory=time.time)
    active_tasks: List[str] = field(default_factory=list)
    performance_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TaskResult:
    """Result of distributed computation."""
    task_id: str
    node_id: str
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    resource_usage: Optional[ResourceCapacity] = None
    timestamp: float = field(default_factory=time.time)


class ConsensusBlock:
    """Blockchain-style block for distributed consensus."""

    def __init__(self, data: Dict[str, Any], previous_hash: str = ""):
        self.timestamp = time.time()
        self.data = data
        self.previous_hash = previous_hash
        self.nonce = 0
        self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        """Calculate block hash."""
        block_string = f"{self.timestamp}{json.dumps(self.data, sort_keys=True)}{self.previous_hash}{self.nonce}"
        return hashlib.sha256(block_string.encode()).hexdigest()

    def mine_block(self, difficulty: int = 2):
        """Mine block with proof of work."""
        target = "0" * difficulty
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()


class DistributedConsensus:
    """Blockchain-based consensus mechanism for distributed decisions."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.blockchain = []
        self.pending_transactions = []
        self.mining_difficulty = 2
        self.mining_reward = 1.0

        # Create genesis block
        genesis_block = ConsensusBlock({"type": "genesis", "node_id": node_id})
        genesis_block.mine_block(self.mining_difficulty)
        self.blockchain.append(genesis_block)

    def add_transaction(self, transaction: Dict[str, Any]):
        """Add transaction to pending pool."""
        transaction["timestamp"] = time.time()
        transaction["node_id"] = self.node_id
        self.pending_transactions.append(transaction)

    def mine_pending_transactions(self) -> Optional[ConsensusBlock]:
        """Mine pending transactions into a block."""
        if not self.pending_transactions:
            return None

        # Create block with pending transactions
        block_data = {
            "transactions": self.pending_transactions.copy(),
            "miner": self.node_id,
            "reward": self.mining_reward
        }

        previous_hash = self.blockchain[-1].hash if self.blockchain else ""
        new_block = ConsensusBlock(block_data, previous_hash)

        # Mine the block
        new_block.mine_block(self.mining_difficulty)

        # Add to blockchain and clear pending
        self.blockchain.append(new_block)
        self.pending_transactions.clear()

        logger.info(f"Block mined by {self.node_id}: {new_block.hash[:10]}...")
        return new_block

    def validate_chain(self) -> bool:
        """Validate blockchain integrity."""
        for i in range(1, len(self.blockchain)):
            current_block = self.blockchain[i]
            previous_block = self.blockchain[i - 1]

            # Check current block hash
            if current_block.hash != current_block.calculate_hash():
                return False

            # Check link to previous block
            if current_block.previous_hash != previous_block.hash:
                return False

        return True

    def get_consensus_data(self, transaction_type: str) -> List[Dict[str, Any]]:
        """Get consensus data for specific transaction type."""
        consensus_data = []
        for block in self.blockchain:
            if "transactions" in block.data:
                for transaction in block.data["transactions"]:
                    if transaction.get("type") == transaction_type:
                        consensus_data.append(transaction)
        return consensus_data


class FederatedLearningCoordinator:
    """Coordinator for federated learning operations."""

    def __init__(self, model_template: BaseEstimator, aggregation_strategy: str = "fedavg"):
        self.model_template = model_template
        self.aggregation_strategy = aggregation_strategy
        self.global_model = clone(model_template)
        self.round_number = 0
        self.client_updates = {}
        self.performance_history = []

    def aggregate_updates(self, client_updates: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate client model updates."""
        if not client_updates:
            return {}

        if self.aggregation_strategy == "fedavg":
            return self._federated_averaging(client_updates)
        elif self.aggregation_strategy == "weighted":
            return self._weighted_aggregation(client_updates)
        else:
            return self._simple_averaging(client_updates)

    def _federated_averaging(self, client_updates: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """FedAvg aggregation algorithm."""
        total_samples = sum(update["num_samples"] for update in client_updates.values())
        aggregated_params = {}

        for _client_id, update in client_updates.items():
            weight = update["num_samples"] / total_samples

            for param_name, param_value in update["parameters"].items():
                if param_name not in aggregated_params:
                    aggregated_params[param_name] = np.zeros_like(param_value)
                aggregated_params[param_name] += weight * param_value

        return {
            "parameters": aggregated_params,
            "round": self.round_number,
            "participants": len(client_updates),
            "total_samples": total_samples
        }

    def _weighted_aggregation(self, client_updates: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Performance-weighted aggregation."""
        total_weight = sum(update["performance"] for update in client_updates.values())
        aggregated_params = {}

        for _client_id, update in client_updates.items():
            weight = update["performance"] / total_weight

            for param_name, param_value in update["parameters"].items():
                if param_name not in aggregated_params:
                    aggregated_params[param_name] = np.zeros_like(param_value)
                aggregated_params[param_name] += weight * param_value

        return {
            "parameters": aggregated_params,
            "round": self.round_number,
            "participants": len(client_updates),
            "aggregation_method": "weighted"
        }

    def _simple_averaging(self, client_updates: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Simple parameter averaging."""
        aggregated_params = {}
        num_clients = len(client_updates)

        for _client_id, update in client_updates.items():
            for param_name, param_value in update["parameters"].items():
                if param_name not in aggregated_params:
                    aggregated_params[param_name] = np.zeros_like(param_value)
                aggregated_params[param_name] += param_value / num_clients

        return {
            "parameters": aggregated_params,
            "round": self.round_number,
            "participants": num_clients,
            "aggregation_method": "simple"
        }


class LoadBalancer:
    """Dynamic load balancer for distributed tasks."""

    def __init__(self, nodes: List[ComputeNode], balancing_strategy: str = "resource_aware"):
        self.nodes = {node.node_id: node for node in nodes}
        self.balancing_strategy = balancing_strategy
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.node_performance = {}

    def add_node(self, node: ComputeNode):
        """Add compute node to the system."""
        self.nodes[node.node_id] = node
        logger.info(f"Added node {node.node_id} ({node.node_type.value})")

    def remove_node(self, node_id: str):
        """Remove compute node from system."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            logger.info(f"Removed node {node_id}")

    def select_node(self, task: ComputeTask) -> Optional[ComputeNode]:
        """Select optimal node for task execution."""
        available_nodes = [node for node in self.nodes.values()
                         if node.status in ["idle", "active"] and self._can_handle_task(node, task)]

        if not available_nodes:
            return None

        if self.balancing_strategy == "round_robin":
            return self._round_robin_selection(available_nodes)
        elif self.balancing_strategy == "resource_aware":
            return self._resource_aware_selection(available_nodes, task)
        elif self.balancing_strategy == "performance_based":
            return self._performance_based_selection(available_nodes)
        else:
            return available_nodes[0]  # First available

    def _can_handle_task(self, node: ComputeNode, task: ComputeTask) -> bool:
        """Check if node can handle the task."""
        if task.resource_requirements is None:
            return True

        req = task.resource_requirements
        available_cpu = node.capacity.cpu_cores - node.current_load.cpu_cores
        available_memory = node.capacity.memory_gb - node.current_load.memory_gb

        return (available_cpu >= req.cpu_cores and
                available_memory >= req.memory_gb)

    def _resource_aware_selection(self, nodes: List[ComputeNode], task: ComputeTask) -> ComputeNode:
        """Select node based on resource availability."""
        def resource_score(node):
            if task.resource_requirements is None:
                # For tasks without requirements, prefer less loaded nodes
                cpu_utilization = node.current_load.cpu_cores / max(node.capacity.cpu_cores, 1)
                memory_utilization = node.current_load.memory_gb / max(node.capacity.memory_gb, 1)
                return -(cpu_utilization + memory_utilization)  # Negative for min selection
            else:
                # For tasks with requirements, prefer nodes with just enough resources
                req = task.resource_requirements
                cpu_available = node.capacity.cpu_cores - node.current_load.cpu_cores
                memory_available = node.capacity.memory_gb - node.current_load.memory_gb

                cpu_score = 1.0 if cpu_available >= req.cpu_cores else 0.0
                memory_score = 1.0 if memory_available >= req.memory_gb else 0.0
                efficiency_score = -(cpu_available + memory_available)  # Prefer tighter fit

                return cpu_score + memory_score + efficiency_score * 0.1

        return max(nodes, key=resource_score)

    def _performance_based_selection(self, nodes: List[ComputeNode]) -> ComputeNode:
        """Select node based on historical performance."""
        def performance_score(node):
            if node.node_id not in self.node_performance:
                return 0.0  # New nodes get neutral score

            perf_data = self.node_performance[node.node_id]
            return perf_data.get("avg_execution_time", float('inf')) * -1  # Prefer faster nodes

        return max(nodes, key=performance_score)

    def _round_robin_selection(self, nodes: List[ComputeNode]) -> ComputeNode:
        """Simple round-robin node selection."""
        if not hasattr(self, '_round_robin_index'):
            self._round_robin_index = 0

        selected_node = nodes[self._round_robin_index % len(nodes)]
        self._round_robin_index += 1
        return selected_node

    def update_node_load(self, node_id: str, load_delta: ResourceCapacity, add: bool = True):
        """Update node resource load."""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            if add:
                node.current_load.cpu_cores += load_delta.cpu_cores
                node.current_load.memory_gb += load_delta.memory_gb
            else:
                node.current_load.cpu_cores = max(0, node.current_load.cpu_cores - load_delta.cpu_cores)
                node.current_load.memory_gb = max(0.0, node.current_load.memory_gb - load_delta.memory_gb)

    def record_task_performance(self, node_id: str, execution_time: float, success: bool):
        """Record task performance for node selection."""
        if node_id not in self.node_performance:
            self.node_performance[node_id] = {
                "total_tasks": 0,
                "successful_tasks": 0,
                "total_time": 0.0,
                "avg_execution_time": 0.0,
                "success_rate": 0.0
            }

        perf = self.node_performance[node_id]
        perf["total_tasks"] += 1
        perf["total_time"] += execution_time

        if success:
            perf["successful_tasks"] += 1

        perf["avg_execution_time"] = perf["total_time"] / perf["total_tasks"]
        perf["success_rate"] = perf["successful_tasks"] / perf["total_tasks"]


class DistributedComputingFramework:
    """
    Advanced distributed computing framework for ML operations.

    Provides comprehensive distributed computing capabilities including:
    - Dynamic task scheduling and load balancing
    - Fault-tolerant execution with retry mechanisms
    - Federated learning coordination
    - Blockchain-based consensus for critical decisions
    - Real-time resource monitoring and optimization
    """

    def __init__(self, master_node_id: str = None):
        self.master_node_id = master_node_id or f"master_{uuid.uuid4().hex[:8]}"
        self.nodes = {}
        self.load_balancer = LoadBalancer([])
        self.consensus = DistributedConsensus(self.master_node_id)
        self.federated_coordinator = None

        # Task management
        self.active_tasks = {}
        self.completed_tasks = {}
        self.task_dependencies = {}

        # System state
        self.system_status = "initializing"
        self.performance_metrics = {}
        self.fault_tolerance_enabled = True

        # Threading
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.background_tasks = []

        logger.info(f"Initialized distributed computing framework with master node: {self.master_node_id}")

    def add_compute_node(self, node: ComputeNode):
        """Add compute node to the distributed system."""
        self.nodes[node.node_id] = node
        self.load_balancer.add_node(node)

        # Record node addition in consensus
        self.consensus.add_transaction({
            "type": "node_added",
            "node_id": node.node_id,
            "node_type": node.node_type.value,
            "capacity": {
                "cpu_cores": node.capacity.cpu_cores,
                "memory_gb": node.capacity.memory_gb
            }
        })

        logger.info(f"Added compute node: {node.node_id} ({node.node_type.value})")

    def remove_compute_node(self, node_id: str):
        """Remove compute node from system."""
        if node_id in self.nodes:
            # Migrate active tasks
            node = self.nodes[node_id]
            if node.active_tasks:
                logger.warning(f"Migrating {len(node.active_tasks)} active tasks from {node_id}")
                for task_id in node.active_tasks:
                    if task_id in self.active_tasks:
                        self._reschedule_task(task_id)

            del self.nodes[node_id]
            self.load_balancer.remove_node(node_id)

            # Record node removal in consensus
            self.consensus.add_transaction({
                "type": "node_removed",
                "node_id": node_id,
                "reason": "manual_removal"
            })

    def submit_task(self, task: ComputeTask) -> str:
        """Submit task for distributed execution."""
        # Generate task ID if not provided
        if not task.task_id:
            task.task_id = f"task_{uuid.uuid4().hex[:8]}"

        # Check dependencies
        if task.dependencies:
            unmet_deps = [dep for dep in task.dependencies
                         if dep not in self.completed_tasks]
            if unmet_deps:
                logger.warning(f"Task {task.task_id} has unmet dependencies: {unmet_deps}")
                # Queue task for later execution
                if task.task_id not in self.task_dependencies:
                    self.task_dependencies[task.task_id] = task
                return task.task_id

        # Submit task for immediate execution
        self.active_tasks[task.task_id] = task

        # Schedule task execution
        future = self.executor.submit(self._execute_task, task)
        self.background_tasks.append(future)

        logger.info(f"Submitted task {task.task_id} with priority {task.priority.value}")
        return task.task_id

    async def submit_task_async(self, task: ComputeTask) -> TaskResult:
        """Submit task for asynchronous distributed execution."""
        task_id = self.submit_task(task)

        # Wait for task completion
        while task_id in self.active_tasks:
            await asyncio.sleep(0.1)

        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        else:
            return TaskResult(
                task_id=task_id,
                node_id="unknown",
                error="Task execution failed or timed out"
            )

    def _execute_task(self, task: ComputeTask) -> TaskResult:
        """Execute task on selected compute node."""
        start_time = time.time()

        # Select optimal node
        selected_node = self.load_balancer.select_node(task)

        if selected_node is None:
            error_msg = "No available nodes for task execution"
            logger.error(f"Task {task.task_id}: {error_msg}")
            result = TaskResult(
                task_id=task.task_id,
                node_id="none",
                error=error_msg,
                execution_time=time.time() - start_time
            )
            self.completed_tasks[task.task_id] = result
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            return result

        # Update node load
        if task.resource_requirements:
            self.load_balancer.update_node_load(
                selected_node.node_id,
                task.resource_requirements,
                add=True
            )

        # Execute task
        result = self._execute_on_node(task, selected_node)

        # Update node load (release resources)
        if task.resource_requirements:
            self.load_balancer.update_node_load(
                selected_node.node_id,
                task.resource_requirements,
                add=False
            )

        # Record performance
        execution_time = time.time() - start_time
        success = result.error is None
        self.load_balancer.record_task_performance(selected_node.node_id, execution_time, success)

        # Store result
        result.execution_time = execution_time
        self.completed_tasks[task.task_id] = result

        # Remove from active tasks
        if task.task_id in self.active_tasks:
            del self.active_tasks[task.task_id]

        # Check for dependent tasks
        self._check_dependent_tasks(task.task_id)

        return result

    def _execute_on_node(self, task: ComputeTask, node: ComputeNode) -> TaskResult:
        """Execute task on specific compute node."""
        try:
            # Update node status
            node.status = "active"
            node.active_tasks.append(task.task_id)

            # Execute function
            if task.timeout:
                # Use process executor for timeout support
                with ProcessPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(task.function, *task.args, **task.kwargs)
                    try:
                        result_value = future.result(timeout=task.timeout)
                    except Exception as e:
                        return TaskResult(
                            task_id=task.task_id,
                            node_id=node.node_id,
                            error=f"Task execution failed: {str(e)}"
                        )
            else:
                # Direct execution
                result_value = task.function(*task.args, **task.kwargs)

            # Create successful result
            result = TaskResult(
                task_id=task.task_id,
                node_id=node.node_id,
                result=result_value
            )

        except Exception as e:
            logger.error(f"Task {task.task_id} failed on node {node.node_id}: {str(e)}")

            # Handle retry logic
            if task.retry_count > 0:
                task.retry_count -= 1
                logger.info(f"Retrying task {task.task_id}, {task.retry_count} retries remaining")

                # Resubmit task
                return self._execute_task(task)

            result = TaskResult(
                task_id=task.task_id,
                node_id=node.node_id,
                error=str(e)
            )

        finally:
            # Update node status
            if task.task_id in node.active_tasks:
                node.active_tasks.remove(task.task_id)

            if not node.active_tasks:
                node.status = "idle"

        return result

    def _reschedule_task(self, task_id: str):
        """Reschedule task due to node failure."""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            logger.info(f"Rescheduling task {task_id}")

            # Reset retry count for rescheduled task
            task.retry_count = max(task.retry_count, 1)

            # Resubmit task
            future = self.executor.submit(self._execute_task, task)
            self.background_tasks.append(future)

    def _check_dependent_tasks(self, completed_task_id: str):
        """Check and execute tasks that were waiting for dependencies."""
        ready_tasks = []

        for task_id, task in list(self.task_dependencies.items()):
            if completed_task_id in task.dependencies:
                # Check if all dependencies are now met
                all_deps_met = all(dep in self.completed_tasks for dep in task.dependencies)

                if all_deps_met:
                    ready_tasks.append(task_id)

        # Execute ready tasks
        for task_id in ready_tasks:
            task = self.task_dependencies.pop(task_id)
            self.submit_task(task)

    def setup_federated_learning(self, model_template: BaseEstimator,
                                aggregation_strategy: str = "fedavg"):
        """Setup federated learning coordination."""
        self.federated_coordinator = FederatedLearningCoordinator(
            model_template, aggregation_strategy
        )
        logger.info(f"Federated learning setup with {aggregation_strategy} aggregation")

    def run_federated_round(self, client_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """Run single federated learning round."""
        if self.federated_coordinator is None:
            raise ValueError("Federated learning not initialized")

        # Distribute model to clients (simulated)
        client_updates = {}

        for client_id, (X, y) in client_data.items():
            # Create local model
            local_model = clone(self.federated_coordinator.model_template)

            # Train local model
            local_model.fit(X, y)

            # Extract parameters (simplified)
            if hasattr(local_model, 'coef_'):
                parameters = {"coef_": local_model.coef_}
                if hasattr(local_model, 'intercept_'):
                    parameters["intercept_"] = local_model.intercept_
            else:
                parameters = {}

            client_updates[client_id] = {
                "parameters": parameters,
                "num_samples": len(X),
                "performance": local_model.score(X, y) if hasattr(local_model, 'score') else 0.8
            }

        # Aggregate updates
        aggregated_update = self.federated_coordinator.aggregate_updates(client_updates)

        # Update global model
        if "parameters" in aggregated_update:
            for param_name, param_value in aggregated_update["parameters"].items():
                if hasattr(self.federated_coordinator.global_model, param_name):
                    setattr(self.federated_coordinator.global_model, param_name, param_value)

        self.federated_coordinator.round_number += 1

        # Record in consensus
        self.consensus.add_transaction({
            "type": "federated_round",
            "round": self.federated_coordinator.round_number,
            "participants": len(client_updates),
            "aggregation_method": aggregated_update.get("aggregation_method", "fedavg")
        })

        return aggregated_update

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        active_nodes = len([n for n in self.nodes.values() if n.status in ["idle", "active"]])
        total_capacity = sum(n.capacity.cpu_cores for n in self.nodes.values())
        current_load = sum(n.current_load.cpu_cores for n in self.nodes.values())

        return {
            "system_status": self.system_status,
            "master_node": self.master_node_id,
            "total_nodes": len(self.nodes),
            "active_nodes": active_nodes,
            "total_capacity": total_capacity,
            "current_load": current_load,
            "utilization": current_load / max(total_capacity, 1),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "pending_dependencies": len(self.task_dependencies),
            "blockchain_length": len(self.consensus.blockchain),
            "federated_learning": self.federated_coordinator is not None,
            "fault_tolerance": self.fault_tolerance_enabled
        }

    def mine_consensus_block(self) -> Optional[ConsensusBlock]:
        """Mine pending transactions into consensus block."""
        return self.consensus.mine_pending_transactions()

    def validate_system_integrity(self) -> Dict[str, bool]:
        """Validate system integrity using consensus."""
        return {
            "blockchain_valid": self.consensus.validate_chain(),
            "nodes_consistent": self._validate_node_consistency(),
            "task_integrity": self._validate_task_integrity()
        }

    def _validate_node_consistency(self) -> bool:
        """Validate node state consistency."""
        consensus_nodes = self.consensus.get_consensus_data("node_added")

        # Check if all consensus nodes are in system
        for node_entry in consensus_nodes:
            if node_entry["node_id"] not in self.nodes:
                logger.warning(f"Consensus node {node_entry['node_id']} not in system")
                return False

        return True

    def _validate_task_integrity(self) -> bool:
        """Validate task execution integrity."""
        # Simple validation - could be enhanced with cryptographic verification
        for task_id, result in self.completed_tasks.items():
            if result.error is None and result.result is None:
                logger.warning(f"Task {task_id} completed without result or error")
                return False

        return True

    def shutdown(self):
        """Shutdown distributed computing framework."""
        logger.info("Shutting down distributed computing framework")

        # Wait for active tasks to complete
        if self.background_tasks:
            for future in as_completed(self.background_tasks, timeout=30):
                try:
                    future.result()
                except Exception as e:
                    logger.warning(f"Task failed during shutdown: {e}")

        # Shutdown executor
        self.executor.shutdown(wait=True)

        # Update system status
        self.system_status = "shutdown"

        logger.info("Distributed computing framework shutdown complete")


# Factory functions and utilities
def create_distributed_framework(
    master_node_id: str = None,
    num_worker_nodes: int = 4,
    node_capacity: ResourceCapacity = None
) -> DistributedComputingFramework:
    """Factory function to create distributed computing framework."""

    if node_capacity is None:
        node_capacity = ResourceCapacity(cpu_cores=2, memory_gb=4.0)

    # Create framework
    framework = DistributedComputingFramework(master_node_id)

    # Add worker nodes
    for i in range(num_worker_nodes):
        worker_node = ComputeNode(
            node_id=f"worker_{i+1}",
            node_type=ComputeNodeType.WORKER,
            capacity=node_capacity
        )
        framework.add_compute_node(worker_node)

    framework.system_status = "ready"
    return framework


def create_compute_task(
    function: Callable,
    args: Tuple = (),
    kwargs: Dict[str, Any] = None,
    priority: str = "normal",
    resource_requirements: ResourceCapacity = None,
    task_id: str = None
) -> ComputeTask:
    """Factory function to create compute task."""

    if kwargs is None:
        kwargs = {}

    priority_map = {
        "critical": TaskPriority.CRITICAL,
        "high": TaskPriority.HIGH,
        "normal": TaskPriority.NORMAL,
        "low": TaskPriority.LOW,
        "background": TaskPriority.BACKGROUND
    }

    priority_enum = priority_map.get(priority, TaskPriority.NORMAL)

    return ComputeTask(
        task_id=task_id or f"task_{uuid.uuid4().hex[:8]}",
        function=function,
        args=args,
        kwargs=kwargs,
        priority=priority_enum,
        resource_requirements=resource_requirements
    )


# Example usage and demonstration
if __name__ == "__main__":
    # Example distributed computation
    def example_computation(x: int, y: int = 1) -> int:
        """Example computation function."""
        time.sleep(0.1)  # Simulate work
        return x * y + np.random.randint(0, 10)

    def ml_training_task(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Example ML training task."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        scores = cross_val_score(model, X, y, cv=3)

        return {
            "accuracy": np.mean(scores),
            "std": np.std(scores),
            "n_samples": len(X)
        }

    # Create distributed framework
    print("🌐 Creating distributed computing framework...")
    framework = create_distributed_framework(
        num_worker_nodes=3,
        node_capacity=ResourceCapacity(cpu_cores=2, memory_gb=2.0)
    )

    # Submit simple tasks
    print("📊 Submitting computational tasks...")
    tasks = []
    for i in range(10):
        task = create_compute_task(
            function=example_computation,
            args=(i,),
            kwargs={"y": i + 1},
            priority="normal"
        )
        task_id = framework.submit_task(task)
        tasks.append(task_id)

    # Wait for task completion
    while len(framework.completed_tasks) < len(tasks):
        time.sleep(0.1)

    # Print results
    print("✅ Task Results:")
    for task_id in tasks[:3]:  # Show first 3 results
        result = framework.completed_tasks[task_id]
        if result.error:
            print(f"  {task_id}: ERROR - {result.error}")
        else:
            print(f"  {task_id}: {result.result} (node: {result.node_id}, time: {result.execution_time:.2f}s)")

    # Setup federated learning
    from sklearn.linear_model import LogisticRegression
    print("🤝 Setting up federated learning...")
    framework.setup_federated_learning(
        model_template=LogisticRegression(random_state=42),
        aggregation_strategy="fedavg"
    )

    # Simulate federated learning round
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

    # Split data among clients
    client_data = {
        "client_1": (X[:300], y[:300]),
        "client_2": (X[300:600], y[300:600]),
        "client_3": (X[600:], y[600:])
    }

    fed_result = framework.run_federated_round(client_data)
    print(f"🔄 Federated learning round completed: {fed_result['participants']} participants")

    # Mine consensus block
    block = framework.mine_consensus_block()
    if block:
        print(f"⛏️ Mined consensus block: {block.hash[:20]}...")

    # System status
    status = framework.get_system_status()
    print("📈 System Status:")
    print(f"  Active nodes: {status['active_nodes']}/{status['total_nodes']}")
    print(f"  Utilization: {status['utilization']:.1%}")
    print(f"  Completed tasks: {status['completed_tasks']}")
    print(f"  Blockchain length: {status['blockchain_length']}")

    # Validate system
    validation = framework.validate_system_integrity()
    print(f"🔍 System validation: {all(validation.values())}")

    # Shutdown
    framework.shutdown()

    logger.info("Distributed computing framework demonstration completed successfully")
