"""
Cloud Integration for Scalable Fairness Research.

Provides cloud computing integration for large-scale fairness experiments,
including auto-scaling, resource optimization, and cost management.

Research contributions:
- Cloud-native fairness research infrastructure
- Automatic resource scaling based on computational demands
- Cost-optimized experiment execution across cloud providers
- Multi-cloud deployment strategies for research resilience
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..logging_config import get_logger

logger = get_logger(__name__)


class CloudProvider(Enum):
    """Supported cloud providers."""
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    LOCAL = "local"


class InstanceType(Enum):
    """Cloud instance types."""
    COMPUTE_OPTIMIZED = "compute_optimized"
    MEMORY_OPTIMIZED = "memory_optimized"
    GPU_ENABLED = "gpu_enabled"
    GENERAL_PURPOSE = "general_purpose"
    SPOT_INSTANCE = "spot_instance"


class ScalingPolicy(Enum):
    """Auto-scaling policies."""
    CPU_UTILIZATION = "cpu_utilization"
    QUEUE_LENGTH = "queue_length"
    CUSTOM_METRIC = "custom_metric"
    PREDICTIVE = "predictive"


@dataclass
class CloudResource:
    """Represents a cloud resource."""
    resource_id: str
    provider: CloudProvider
    instance_type: InstanceType
    region: str
    cpu_cores: int
    memory_gb: float
    gpu_count: int = 0
    hourly_cost: float = 0.0
    is_spot_instance: bool = False
    current_status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'resource_id': self.resource_id,
            'provider': self.provider.value,
            'instance_type': self.instance_type.value,
            'region': self.region,
            'cpu_cores': self.cpu_cores,
            'memory_gb': self.memory_gb,
            'gpu_count': self.gpu_count,
            'hourly_cost': self.hourly_cost,
            'is_spot_instance': self.is_spot_instance,
            'current_status': self.current_status,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class ScalingRule:
    """Auto-scaling rule definition."""
    rule_id: str
    policy: ScalingPolicy
    metric_name: str
    threshold_up: float
    threshold_down: float
    scale_up_count: int = 1
    scale_down_count: int = 1
    cooldown_minutes: int = 5
    min_instances: int = 1
    max_instances: int = 10

    def should_scale_up(self, metric_value: float) -> bool:
        """Check if should scale up based on metric value."""
        return metric_value > self.threshold_up

    def should_scale_down(self, metric_value: float) -> bool:
        """Check if should scale down based on metric value."""
        return metric_value < self.threshold_down


class CloudProviderAdapter(ABC):
    """Abstract adapter for cloud providers."""

    @abstractmethod
    def launch_instance(
        self,
        instance_type: InstanceType,
        region: str,
        configuration: Dict[str, Any]
    ) -> CloudResource:
        """Launch a new cloud instance."""
        pass

    @abstractmethod
    def terminate_instance(self, resource_id: str) -> bool:
        """Terminate a cloud instance."""
        pass

    @abstractmethod
    def get_instance_status(self, resource_id: str) -> str:
        """Get status of a cloud instance."""
        pass

    @abstractmethod
    def get_pricing_info(
        self,
        instance_type: InstanceType,
        region: str
    ) -> Dict[str, float]:
        """Get pricing information for instance type."""
        pass


class MockCloudAdapter(CloudProviderAdapter):
    """Mock cloud provider adapter for testing."""

    def __init__(self, provider: CloudProvider):
        self.provider = provider
        self.instances: Dict[str, CloudResource] = {}
        self.pricing = self._initialize_pricing()

    def _initialize_pricing(self) -> Dict[str, Dict[str, float]]:
        """Initialize mock pricing data."""
        return {
            InstanceType.GENERAL_PURPOSE.value: {"us-east-1": 0.096, "eu-west-1": 0.108},
            InstanceType.COMPUTE_OPTIMIZED.value: {"us-east-1": 0.192, "eu-west-1": 0.216},
            InstanceType.MEMORY_OPTIMIZED.value: {"us-east-1": 0.384, "eu-west-1": 0.432},
            InstanceType.GPU_ENABLED.value: {"us-east-1": 3.06, "eu-west-1": 3.45},
            InstanceType.SPOT_INSTANCE.value: {"us-east-1": 0.031, "eu-west-1": 0.035}
        }

    def launch_instance(
        self,
        instance_type: InstanceType,
        region: str,
        configuration: Dict[str, Any]
    ) -> CloudResource:
        """Launch a mock cloud instance."""
        resource_id = f"{self.provider.value}_{instance_type.value}_{int(time.time())}"

        # Simulate instance specifications based on type
        specs = self._get_instance_specs(instance_type)
        pricing = self.pricing.get(instance_type.value, {}).get(region, 0.1)

        resource = CloudResource(
            resource_id=resource_id,
            provider=self.provider,
            instance_type=instance_type,
            region=region,
            cpu_cores=specs['cpu_cores'],
            memory_gb=specs['memory_gb'],
            gpu_count=specs['gpu_count'],
            hourly_cost=pricing,
            is_spot_instance=instance_type == InstanceType.SPOT_INSTANCE,
            current_status="launching"
        )

        self.instances[resource_id] = resource

        logger.info(f"Launched {self.provider.value} instance: {resource_id}")
        return resource

    def terminate_instance(self, resource_id: str) -> bool:
        """Terminate a mock cloud instance."""
        if resource_id in self.instances:
            self.instances[resource_id].current_status = "terminating"
            # Simulate termination delay
            time.sleep(0.1)
            del self.instances[resource_id]
            logger.info(f"Terminated instance: {resource_id}")
            return True
        return False

    def get_instance_status(self, resource_id: str) -> str:
        """Get mock instance status."""
        if resource_id in self.instances:
            resource = self.instances[resource_id]

            # Simulate status transitions
            if resource.current_status == "launching":
                # Simulate launch time
                elapsed = (datetime.now() - resource.created_at).total_seconds()
                if elapsed > 30:  # 30 seconds to launch
                    resource.current_status = "running"

            return resource.current_status

        return "not_found"

    def get_pricing_info(
        self,
        instance_type: InstanceType,
        region: str
    ) -> Dict[str, float]:
        """Get mock pricing information."""
        base_price = self.pricing.get(instance_type.value, {}).get(region, 0.1)

        return {
            'hourly_cost': base_price,
            'monthly_cost': base_price * 24 * 30,
            'spot_price': base_price * 0.3 if instance_type != InstanceType.SPOT_INSTANCE else base_price
        }

    def _get_instance_specs(self, instance_type: InstanceType) -> Dict[str, int]:
        """Get mock instance specifications."""
        specs = {
            InstanceType.GENERAL_PURPOSE: {'cpu_cores': 4, 'memory_gb': 16, 'gpu_count': 0},
            InstanceType.COMPUTE_OPTIMIZED: {'cpu_cores': 8, 'memory_gb': 16, 'gpu_count': 0},
            InstanceType.MEMORY_OPTIMIZED: {'cpu_cores': 4, 'memory_gb': 64, 'gpu_count': 0},
            InstanceType.GPU_ENABLED: {'cpu_cores': 8, 'memory_gb': 32, 'gpu_count': 1},
            InstanceType.SPOT_INSTANCE: {'cpu_cores': 2, 'memory_gb': 8, 'gpu_count': 0}
        }

        return specs.get(instance_type, {'cpu_cores': 2, 'memory_gb': 8, 'gpu_count': 0})


class ResourceManager:
    """
    Manages cloud resources for fairness research workloads.
    
    Handles resource allocation, cost optimization, and lifecycle management
    across multiple cloud providers.
    """

    def __init__(
        self,
        default_provider: CloudProvider = CloudProvider.AWS,
        default_region: str = "us-east-1"
    ):
        """
        Initialize resource manager.
        
        Args:
            default_provider: Default cloud provider
            default_region: Default region
        """
        self.default_provider = default_provider
        self.default_region = default_region

        # Cloud adapters
        self.adapters: Dict[CloudProvider, CloudProviderAdapter] = {
            CloudProvider.AWS: MockCloudAdapter(CloudProvider.AWS),
            CloudProvider.AZURE: MockCloudAdapter(CloudProvider.AZURE),
            CloudProvider.GCP: MockCloudAdapter(CloudProvider.GCP)
        }

        # Resource tracking
        self.active_resources: Dict[str, CloudResource] = {}
        self.resource_usage: Dict[str, List[Dict[str, Any]]] = {}
        self.total_cost: float = 0.0

        logger.info(f"ResourceManager initialized with provider: {default_provider.value}")

    def launch_research_cluster(
        self,
        cluster_config: Dict[str, Any],
        provider: Optional[CloudProvider] = None,
        region: Optional[str] = None
    ) -> List[CloudResource]:
        """
        Launch a cluster of resources for research workloads.
        
        Args:
            cluster_config: Cluster configuration
            provider: Cloud provider (uses default if None)
            region: Region (uses default if None)
            
        Returns:
            List of launched resources
        """
        provider = provider or self.default_provider
        region = region or self.default_region

        logger.info(f"Launching research cluster with {cluster_config}")

        adapter = self.adapters[provider]
        launched_resources = []

        # Launch instances based on configuration
        for instance_spec in cluster_config.get('instances', []):
            instance_type = InstanceType(instance_spec['type'])
            count = instance_spec.get('count', 1)

            for _ in range(count):
                resource = adapter.launch_instance(
                    instance_type=instance_type,
                    region=region,
                    configuration=instance_spec.get('config', {})
                )

                self.active_resources[resource.resource_id] = resource
                launched_resources.append(resource)

        logger.info(f"Launched {len(launched_resources)} instances")
        return launched_resources

    def terminate_cluster(self, resource_ids: List[str]) -> int:
        """
        Terminate a cluster of resources.
        
        Args:
            resource_ids: List of resource IDs to terminate
            
        Returns:
            Number of successfully terminated resources
        """
        logger.info(f"Terminating cluster with {len(resource_ids)} resources")

        terminated_count = 0

        for resource_id in resource_ids:
            if resource_id in self.active_resources:
                resource = self.active_resources[resource_id]
                adapter = self.adapters[resource.provider]

                if adapter.terminate_instance(resource_id):
                    # Calculate final cost
                    uptime_hours = self._calculate_uptime_hours(resource)
                    cost = uptime_hours * resource.hourly_cost
                    self.total_cost += cost

                    # Record usage
                    self._record_resource_usage(resource, uptime_hours, cost)

                    del self.active_resources[resource_id]
                    terminated_count += 1

        logger.info(f"Terminated {terminated_count} resources")
        return terminated_count

    def get_cost_estimate(
        self,
        cluster_config: Dict[str, Any],
        duration_hours: float,
        provider: Optional[CloudProvider] = None,
        region: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get cost estimate for running a cluster configuration.
        
        Args:
            cluster_config: Cluster configuration
            duration_hours: Expected runtime in hours
            provider: Cloud provider
            region: Region
            
        Returns:
            Cost estimate breakdown
        """
        provider = provider or self.default_provider
        region = region or self.default_region

        adapter = self.adapters[provider]

        cost_breakdown = {}
        total_hourly_cost = 0.0

        for instance_spec in cluster_config.get('instances', []):
            instance_type = InstanceType(instance_spec['type'])
            count = instance_spec.get('count', 1)

            pricing = adapter.get_pricing_info(instance_type, region)
            hourly_cost = pricing['hourly_cost'] * count
            total_cost = hourly_cost * duration_hours

            cost_breakdown[f"{instance_type.value}_x{count}"] = {
                'hourly_cost': hourly_cost,
                'total_cost': total_cost,
                'spot_savings': (pricing['hourly_cost'] - pricing.get('spot_price', 0)) * count * duration_hours
            }

            total_hourly_cost += hourly_cost

        return {
            'provider': provider.value,
            'region': region,
            'duration_hours': duration_hours,
            'total_hourly_cost': total_hourly_cost,
            'total_estimated_cost': total_hourly_cost * duration_hours,
            'cost_breakdown': cost_breakdown,
            'potential_spot_savings': sum(item['spot_savings'] for item in cost_breakdown.values())
        }

    def optimize_costs(self, cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize cluster configuration for cost efficiency.
        
        Args:
            cluster_config: Original cluster configuration
            
        Returns:
            Optimized configuration with cost savings
        """
        logger.info("Optimizing cluster configuration for costs")

        optimized_config = cluster_config.copy()
        optimizations = []

        # Optimization 1: Suggest spot instances where applicable
        for instance_spec in optimized_config.get('instances', []):
            if instance_spec['type'] != InstanceType.SPOT_INSTANCE.value:
                if instance_spec.get('fault_tolerant', True):
                    original_type = instance_spec['type']
                    instance_spec['type'] = InstanceType.SPOT_INSTANCE.value
                    instance_spec['original_type'] = original_type

                    optimizations.append({
                        'type': 'spot_instance_conversion',
                        'original_type': original_type,
                        'savings_percentage': 70
                    })

        # Optimization 2: Right-size instances based on workload
        for instance_spec in optimized_config.get('instances', []):
            workload_requirements = instance_spec.get('requirements', {})

            # If CPU requirements are low, suggest smaller instance
            if workload_requirements.get('cpu_intensive', False) is False:
                if instance_spec['type'] == InstanceType.COMPUTE_OPTIMIZED.value:
                    instance_spec['type'] = InstanceType.GENERAL_PURPOSE.value

                    optimizations.append({
                        'type': 'right_sizing',
                        'change': 'downgrade_cpu',
                        'savings_percentage': 50
                    })

        # Optimization 3: Region selection based on cost
        cheapest_region = self._find_cheapest_region(optimized_config)
        if cheapest_region != self.default_region:
            optimizations.append({
                'type': 'region_optimization',
                'suggested_region': cheapest_region,
                'savings_percentage': 15
            })

        return {
            'original_config': cluster_config,
            'optimized_config': optimized_config,
            'optimizations': optimizations,
            'estimated_savings_percentage': sum(opt.get('savings_percentage', 0) for opt in optimizations) / len(optimizations) if optimizations else 0
        }

    def _calculate_uptime_hours(self, resource: CloudResource) -> float:
        """Calculate uptime hours for a resource."""
        uptime_delta = datetime.now() - resource.created_at
        return uptime_delta.total_seconds() / 3600

    def _record_resource_usage(self, resource: CloudResource, uptime_hours: float, cost: float):
        """Record resource usage for analytics."""
        usage_record = {
            'resource_id': resource.resource_id,
            'provider': resource.provider.value,
            'instance_type': resource.instance_type.value,
            'region': resource.region,
            'uptime_hours': uptime_hours,
            'cost': cost,
            'terminated_at': datetime.now().isoformat()
        }

        if resource.resource_id not in self.resource_usage:
            self.resource_usage[resource.resource_id] = []

        self.resource_usage[resource.resource_id].append(usage_record)

    def _find_cheapest_region(self, cluster_config: Dict[str, Any]) -> str:
        """Find the cheapest region for a cluster configuration."""
        regions = ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"]
        region_costs = {}

        for region in regions:
            total_cost = 0.0

            for instance_spec in cluster_config.get('instances', []):
                instance_type = InstanceType(instance_spec['type'])
                count = instance_spec.get('count', 1)

                adapter = self.adapters[self.default_provider]
                pricing = adapter.get_pricing_info(instance_type, region)
                total_cost += pricing['hourly_cost'] * count

            region_costs[region] = total_cost

        return min(region_costs.items(), key=lambda x: x[1])[0]

    def get_resource_metrics(self) -> Dict[str, Any]:
        """Get comprehensive resource metrics."""
        active_count = len(self.active_resources)
        total_usage_records = sum(len(usage) for usage in self.resource_usage.values())

        # Calculate current hourly cost
        current_hourly_cost = sum(
            resource.hourly_cost for resource in self.active_resources.values()
        )

        # Provider distribution
        provider_distribution = {}
        for resource in self.active_resources.values():
            provider = resource.provider.value
            provider_distribution[provider] = provider_distribution.get(provider, 0) + 1

        # Instance type distribution
        instance_type_distribution = {}
        for resource in self.active_resources.values():
            instance_type = resource.instance_type.value
            instance_type_distribution[instance_type] = instance_type_distribution.get(instance_type, 0) + 1

        return {
            'active_resources': active_count,
            'total_cost_to_date': self.total_cost,
            'current_hourly_cost': current_hourly_cost,
            'total_usage_records': total_usage_records,
            'provider_distribution': provider_distribution,
            'instance_type_distribution': instance_type_distribution,
            'resource_details': {rid: resource.to_dict() for rid, resource in self.active_resources.items()}
        }


class AutoScaler:
    """
    Automatic scaling system for research workloads.
    
    Monitors metrics and automatically scales cloud resources up or down
    based on configurable policies and thresholds.
    """

    def __init__(
        self,
        resource_manager: ResourceManager,
        monitoring_interval: int = 60
    ):
        """
        Initialize auto-scaler.
        
        Args:
            resource_manager: Resource manager instance
            monitoring_interval: Monitoring interval in seconds
        """
        self.resource_manager = resource_manager
        self.monitoring_interval = monitoring_interval

        # Scaling rules and state
        self.scaling_rules: Dict[str, ScalingRule] = {}
        self.metrics_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self.last_scaling_action: Dict[str, datetime] = {}

        # Auto-scaler state
        self.is_running = False
        self.scaler_thread: Optional[Any] = None

        logger.info("AutoScaler initialized")

    def add_scaling_rule(self, rule: ScalingRule):
        """Add an auto-scaling rule."""
        self.scaling_rules[rule.rule_id] = rule
        logger.info(f"Added scaling rule: {rule.rule_id}")

    def remove_scaling_rule(self, rule_id: str):
        """Remove an auto-scaling rule."""
        if rule_id in self.scaling_rules:
            del self.scaling_rules[rule_id]
            logger.info(f"Removed scaling rule: {rule_id}")

    def start_auto_scaling(self):
        """Start the auto-scaling monitor."""
        if self.is_running:
            logger.warning("Auto-scaler is already running")
            return

        self.is_running = True

        # In a real implementation, would start monitoring thread
        logger.info("Auto-scaler started")

    def stop_auto_scaling(self):
        """Stop the auto-scaling monitor."""
        self.is_running = False
        logger.info("Auto-scaler stopped")

    def collect_metrics(self) -> Dict[str, float]:
        """Collect current system metrics."""
        # Simulate metric collection
        import random

        # Mock metrics - in practice would collect from monitoring systems
        metrics = {
            'cpu_utilization': random.uniform(20, 95),
            'queue_length': random.randint(0, 50),
            'memory_utilization': random.uniform(30, 85),
            'task_completion_rate': random.uniform(5, 25)
        }

        # Store metrics history
        timestamp = datetime.now()
        for metric_name, value in metrics.items():
            if metric_name not in self.metrics_history:
                self.metrics_history[metric_name] = []

            self.metrics_history[metric_name].append((timestamp, value))

            # Keep only last 100 measurements
            if len(self.metrics_history[metric_name]) > 100:
                self.metrics_history[metric_name] = self.metrics_history[metric_name][-100:]

        return metrics

    def evaluate_scaling_rules(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Evaluate scaling rules against current metrics."""
        scaling_actions = []

        for rule_id, rule in self.scaling_rules.items():
            if rule.metric_name not in metrics:
                continue

            metric_value = metrics[rule.metric_name]
            current_time = datetime.now()

            # Check cooldown period
            last_action_time = self.last_scaling_action.get(rule_id)
            if last_action_time:
                cooldown_elapsed = (current_time - last_action_time).total_seconds() / 60
                if cooldown_elapsed < rule.cooldown_minutes:
                    continue

            # Evaluate scaling conditions
            if rule.should_scale_up(metric_value):
                # Check if we're below max instances
                current_instances = len(self.resource_manager.active_resources)
                if current_instances < rule.max_instances:
                    scaling_actions.append({
                        'rule_id': rule_id,
                        'action': 'scale_up',
                        'metric_name': rule.metric_name,
                        'metric_value': metric_value,
                        'threshold': rule.threshold_up,
                        'scale_count': rule.scale_up_count
                    })

            elif rule.should_scale_down(metric_value):
                # Check if we're above min instances
                current_instances = len(self.resource_manager.active_resources)
                if current_instances > rule.min_instances:
                    scaling_actions.append({
                        'rule_id': rule_id,
                        'action': 'scale_down',
                        'metric_name': rule.metric_name,
                        'metric_value': metric_value,
                        'threshold': rule.threshold_down,
                        'scale_count': rule.scale_down_count
                    })

        return scaling_actions

    def execute_scaling_action(self, action: Dict[str, Any]) -> bool:
        """Execute a scaling action."""
        rule_id = action['rule_id']
        action_type = action['action']
        scale_count = action['scale_count']

        logger.info(f"Executing scaling action: {action_type} by {scale_count} for rule {rule_id}")

        try:
            if action_type == 'scale_up':
                # Launch additional instances
                cluster_config = {
                    'instances': [{
                        'type': InstanceType.GENERAL_PURPOSE.value,
                        'count': scale_count,
                        'fault_tolerant': True
                    }]
                }

                self.resource_manager.launch_research_cluster(cluster_config)

            elif action_type == 'scale_down':
                # Terminate some instances (oldest first)
                active_resources = list(self.resource_manager.active_resources.keys())
                resources_to_terminate = active_resources[:scale_count]

                self.resource_manager.terminate_cluster(resources_to_terminate)

            # Record scaling action
            self.last_scaling_action[rule_id] = datetime.now()

            logger.info("Scaling action completed successfully")
            return True

        except Exception as e:
            logger.error(f"Scaling action failed: {e}")
            return False

    def run_scaling_cycle(self) -> Dict[str, Any]:
        """Run a single scaling evaluation cycle."""
        # Collect metrics
        metrics = self.collect_metrics()

        # Evaluate scaling rules
        scaling_actions = self.evaluate_scaling_rules(metrics)

        # Execute actions
        executed_actions = []
        for action in scaling_actions:
            if self.execute_scaling_action(action):
                executed_actions.append(action)

        return {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'scaling_actions_evaluated': len(scaling_actions),
            'scaling_actions_executed': len(executed_actions),
            'executed_actions': executed_actions
        }

    def get_scaling_history(self, hours: int = 24) -> Dict[str, Any]:
        """Get scaling history for the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        # Filter metrics history
        recent_metrics = {}
        for metric_name, history in self.metrics_history.items():
            recent_history = [
                (timestamp, value) for timestamp, value in history
                if timestamp >= cutoff_time
            ]
            recent_metrics[metric_name] = recent_history

        # Filter scaling actions
        recent_scaling_actions = {}
        for rule_id, last_action_time in self.last_scaling_action.items():
            if last_action_time >= cutoff_time:
                recent_scaling_actions[rule_id] = last_action_time.isoformat()

        return {
            'time_range_hours': hours,
            'metrics_history': recent_metrics,
            'scaling_actions': recent_scaling_actions,
            'active_scaling_rules': list(self.scaling_rules.keys())
        }


class CloudExperimentManager:
    """
    High-level manager for cloud-based fairness experiments.
    
    Orchestrates cloud resources, auto-scaling, and cost optimization
    for large-scale fairness research projects.
    """

    def __init__(
        self,
        project_name: str,
        default_provider: CloudProvider = CloudProvider.AWS,
        enable_auto_scaling: bool = True,
        cost_budget_usd: Optional[float] = None
    ):
        """
        Initialize cloud experiment manager.
        
        Args:
            project_name: Name of the research project
            default_provider: Default cloud provider
            enable_auto_scaling: Enable automatic scaling
            cost_budget_usd: Cost budget in USD (optional)
        """
        self.project_name = project_name
        self.default_provider = default_provider
        self.enable_auto_scaling = enable_auto_scaling
        self.cost_budget_usd = cost_budget_usd

        # Initialize components
        self.resource_manager = ResourceManager(default_provider)
        self.auto_scaler = AutoScaler(self.resource_manager) if enable_auto_scaling else None

        # Experiment tracking
        self.active_experiments: Dict[str, Dict[str, Any]] = {}
        self.experiment_costs: Dict[str, float] = {}

        # Budget tracking
        self.budget_alerts_enabled = cost_budget_usd is not None
        self.budget_threshold_percentage = 80.0  # Alert at 80% of budget

        logger.info(f"CloudExperimentManager initialized for project: {project_name}")

    def create_experiment_infrastructure(
        self,
        experiment_name: str,
        infrastructure_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create cloud infrastructure for an experiment.
        
        Args:
            experiment_name: Name of the experiment
            infrastructure_config: Infrastructure requirements
            
        Returns:
            Infrastructure details and cost estimates
        """
        logger.info(f"Creating infrastructure for experiment: {experiment_name}")

        # Optimize configuration for cost efficiency
        optimization_result = self.resource_manager.optimize_costs(infrastructure_config)
        optimized_config = optimization_result['optimized_config']

        # Get cost estimate
        estimated_duration = infrastructure_config.get('estimated_duration_hours', 8.0)
        cost_estimate = self.resource_manager.get_cost_estimate(
            optimized_config, estimated_duration
        )

        # Check budget constraints
        if self.cost_budget_usd:
            total_spent = sum(self.experiment_costs.values())
            if total_spent + cost_estimate['total_estimated_cost'] > self.cost_budget_usd:
                raise ValueError(
                    f"Experiment would exceed budget. "
                    f"Cost: ${cost_estimate['total_estimated_cost']:.2f}, "
                    f"Remaining budget: ${self.cost_budget_usd - total_spent:.2f}"
                )

        # Launch infrastructure
        launched_resources = self.resource_manager.launch_research_cluster(optimized_config)

        # Track experiment
        self.active_experiments[experiment_name] = {
            'infrastructure_config': infrastructure_config,
            'optimized_config': optimized_config,
            'launched_resources': [r.resource_id for r in launched_resources],
            'cost_estimate': cost_estimate,
            'created_at': datetime.now().isoformat(),
            'status': 'running'
        }

        self.experiment_costs[experiment_name] = 0.0  # Will be updated on termination

        # Setup auto-scaling if enabled
        if self.auto_scaler and infrastructure_config.get('enable_auto_scaling', True):
            self._setup_experiment_scaling_rules(experiment_name, infrastructure_config)

        logger.info(f"Infrastructure created for experiment: {experiment_name}")

        return {
            'experiment_name': experiment_name,
            'launched_resources': len(launched_resources),
            'resource_details': [r.to_dict() for r in launched_resources],
            'cost_estimate': cost_estimate,
            'optimization_applied': optimization_result['optimizations'],
            'auto_scaling_enabled': self.auto_scaler is not None
        }

    def terminate_experiment_infrastructure(self, experiment_name: str) -> Dict[str, Any]:
        """
        Terminate cloud infrastructure for an experiment.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            Termination summary with final costs
        """
        if experiment_name not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_name} not found")

        logger.info(f"Terminating infrastructure for experiment: {experiment_name}")

        experiment = self.active_experiments[experiment_name]
        resource_ids = experiment['launched_resources']

        # Terminate resources
        terminated_count = self.resource_manager.terminate_cluster(resource_ids)

        # Calculate final costs
        final_cost = 0.0
        for resource_id in resource_ids:
            if resource_id in self.resource_manager.resource_usage:
                usage_records = self.resource_manager.resource_usage[resource_id]
                final_cost += sum(record['cost'] for record in usage_records)

        self.experiment_costs[experiment_name] = final_cost

        # Update experiment status
        experiment['status'] = 'terminated'
        experiment['terminated_at'] = datetime.now().isoformat()
        experiment['final_cost'] = final_cost

        logger.info(f"Infrastructure terminated for experiment: {experiment_name}")

        return {
            'experiment_name': experiment_name,
            'terminated_resources': terminated_count,
            'final_cost': final_cost,
            'estimated_cost': experiment['cost_estimate']['total_estimated_cost'],
            'cost_difference': final_cost - experiment['cost_estimate']['total_estimated_cost']
        }

    def monitor_experiment_costs(self) -> Dict[str, Any]:
        """Monitor costs across all active experiments."""
        total_spent = sum(self.experiment_costs.values())

        # Current running costs
        current_hourly_cost = sum(
            resource.hourly_cost
            for resource in self.resource_manager.active_resources.values()
        )

        # Budget status
        budget_status = {}
        if self.cost_budget_usd:
            budget_used_percentage = (total_spent / self.cost_budget_usd) * 100
            remaining_budget = self.cost_budget_usd - total_spent

            budget_status = {
                'budget_total': self.cost_budget_usd,
                'budget_used': total_spent,
                'budget_remaining': remaining_budget,
                'budget_used_percentage': budget_used_percentage,
                'budget_alert': budget_used_percentage > self.budget_threshold_percentage
            }

        # Cost by experiment
        experiment_costs = {}
        for exp_name, exp_data in self.active_experiments.items():
            if exp_data['status'] == 'running':
                # Estimate current cost
                elapsed_hours = self._calculate_experiment_elapsed_hours(exp_data)
                estimated_current_cost = elapsed_hours * sum(
                    resource.hourly_cost
                    for resource_id in exp_data['launched_resources']
                    for resource in [self.resource_manager.active_resources.get(resource_id)]
                    if resource
                )
                experiment_costs[exp_name] = estimated_current_cost
            else:
                experiment_costs[exp_name] = self.experiment_costs.get(exp_name, 0.0)

        return {
            'total_spent': total_spent,
            'current_hourly_cost': current_hourly_cost,
            'budget_status': budget_status,
            'experiment_costs': experiment_costs,
            'active_experiments': len([e for e in self.active_experiments.values() if e['status'] == 'running'])
        }

    def get_cost_optimization_recommendations(self) -> Dict[str, Any]:
        """Get recommendations for cost optimization."""
        recommendations = []
        potential_savings = 0.0

        # Analyze active resources
        for resource in self.resource_manager.active_resources.values():
            # Recommendation 1: Spot instances
            if not resource.is_spot_instance and resource.instance_type != InstanceType.SPOT_INSTANCE:
                spot_savings = resource.hourly_cost * 0.7  # 70% savings
                recommendations.append({
                    'type': 'spot_instance',
                    'resource_id': resource.resource_id,
                    'current_cost_hourly': resource.hourly_cost,
                    'potential_savings_hourly': spot_savings,
                    'description': f"Switch to spot instance for {resource.resource_id}"
                })
                potential_savings += spot_savings

            # Recommendation 2: Right-sizing
            if resource.instance_type == InstanceType.COMPUTE_OPTIMIZED:
                rightsizing_savings = resource.hourly_cost * 0.3  # 30% savings
                recommendations.append({
                    'type': 'rightsizing',
                    'resource_id': resource.resource_id,
                    'current_cost_hourly': resource.hourly_cost,
                    'potential_savings_hourly': rightsizing_savings,
                    'description': f"Downsize to general purpose instance for {resource.resource_id}"
                })
                potential_savings += rightsizing_savings * 0.5  # Conservative estimate

        # Recommendation 3: Regional optimization
        if len(self.resource_manager.active_resources) > 0:
            regional_savings = sum(resource.hourly_cost for resource in self.resource_manager.active_resources.values()) * 0.15
            recommendations.append({
                'type': 'regional_optimization',
                'description': "Consider moving workloads to cheaper regions",
                'potential_savings_hourly': regional_savings
            })
            potential_savings += regional_savings * 0.3  # Conservative estimate

        return {
            'total_recommendations': len(recommendations),
            'potential_hourly_savings': potential_savings,
            'potential_monthly_savings': potential_savings * 24 * 30,
            'recommendations': recommendations
        }

    def _setup_experiment_scaling_rules(
        self,
        experiment_name: str,
        infrastructure_config: Dict[str, Any]
    ):
        """Setup auto-scaling rules for an experiment."""
        if not self.auto_scaler:
            return

        # Default scaling rule based on queue length
        queue_rule = ScalingRule(
            rule_id=f"{experiment_name}_queue_scaling",
            policy=ScalingPolicy.QUEUE_LENGTH,
            metric_name="queue_length",
            threshold_up=infrastructure_config.get('scale_up_queue_threshold', 20),
            threshold_down=infrastructure_config.get('scale_down_queue_threshold', 5),
            scale_up_count=infrastructure_config.get('scale_up_count', 2),
            scale_down_count=infrastructure_config.get('scale_down_count', 1),
            min_instances=infrastructure_config.get('min_instances', 1),
            max_instances=infrastructure_config.get('max_instances', 10)
        )

        self.auto_scaler.add_scaling_rule(queue_rule)

        # CPU utilization rule
        cpu_rule = ScalingRule(
            rule_id=f"{experiment_name}_cpu_scaling",
            policy=ScalingPolicy.CPU_UTILIZATION,
            metric_name="cpu_utilization",
            threshold_up=80.0,
            threshold_down=30.0,
            scale_up_count=1,
            scale_down_count=1,
            cooldown_minutes=10,
            min_instances=infrastructure_config.get('min_instances', 1),
            max_instances=infrastructure_config.get('max_instances', 10)
        )

        self.auto_scaler.add_scaling_rule(cpu_rule)

        logger.info(f"Setup auto-scaling rules for experiment: {experiment_name}")

    def _calculate_experiment_elapsed_hours(self, experiment_data: Dict[str, Any]) -> float:
        """Calculate elapsed hours for an experiment."""
        created_at = datetime.fromisoformat(experiment_data['created_at'])
        elapsed_delta = datetime.now() - created_at
        return elapsed_delta.total_seconds() / 3600

    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all cloud operations."""
        resource_metrics = self.resource_manager.get_resource_metrics()
        cost_monitoring = self.monitor_experiment_costs()
        cost_recommendations = self.get_cost_optimization_recommendations()

        # Auto-scaling status
        auto_scaling_status = {}
        if self.auto_scaler:
            auto_scaling_status = {
                'enabled': True,
                'active_rules': len(self.auto_scaler.scaling_rules),
                'is_running': self.auto_scaler.is_running
            }

        return {
            'project_name': self.project_name,
            'resource_metrics': resource_metrics,
            'cost_monitoring': cost_monitoring,
            'cost_recommendations': cost_recommendations,
            'auto_scaling_status': auto_scaling_status,
            'active_experiments': {
                name: {
                    'status': exp['status'],
                    'resources': len(exp['launched_resources']),
                    'created_at': exp['created_at']
                }
                for name, exp in self.active_experiments.items()
            }
        }


# Example usage and CLI interface
def main():
    """CLI interface for cloud integration."""
    import argparse

    parser = argparse.ArgumentParser(description="Cloud Integration for Fairness Research")
    parser.add_argument("--demo", action="store_true", help="Run demonstration")
    parser.add_argument("--provider", choices=[p.value for p in CloudProvider],
                       default="aws", help="Cloud provider")
    parser.add_argument("--budget", type=float, help="Budget in USD")

    args = parser.parse_args()

    if args.demo:
        print("☁️ Starting Cloud Integration Demo")

        # Initialize cloud experiment manager
        manager = CloudExperimentManager(
            project_name="fairness_research_demo",
            default_provider=CloudProvider(args.provider),
            enable_auto_scaling=True,
            cost_budget_usd=args.budget
        )

        print(f"✅ Cloud manager initialized for provider: {args.provider}")
        if args.budget:
            print(f"💰 Budget set to: ${args.budget}")

        # Demo infrastructure configuration
        infrastructure_config = {
            'instances': [
                {
                    'type': InstanceType.COMPUTE_OPTIMIZED.value,
                    'count': 3,
                    'requirements': {'cpu_intensive': True, 'fault_tolerant': True}
                },
                {
                    'type': InstanceType.MEMORY_OPTIMIZED.value,
                    'count': 1,
                    'requirements': {'memory_intensive': True}
                }
            ],
            'estimated_duration_hours': 4.0,
            'enable_auto_scaling': True,
            'min_instances': 2,
            'max_instances': 8
        }

        print("\n📋 Infrastructure Configuration:")
        for instance_spec in infrastructure_config['instances']:
            print(f"   - {instance_spec['count']}x {instance_spec['type']}")

        # Get cost estimate
        print("\n💵 Cost Analysis:")
        cost_estimate = manager.resource_manager.get_cost_estimate(
            infrastructure_config,
            infrastructure_config['estimated_duration_hours']
        )

        print(f"   Estimated total cost: ${cost_estimate['total_estimated_cost']:.2f}")
        print(f"   Hourly cost: ${cost_estimate['total_hourly_cost']:.2f}")
        if cost_estimate['potential_spot_savings'] > 0:
            print(f"   Potential spot savings: ${cost_estimate['potential_spot_savings']:.2f}")

        # Get optimization recommendations
        print("\n🎯 Cost Optimization:")
        optimization = manager.resource_manager.optimize_costs(infrastructure_config)

        if optimization['optimizations']:
            print(f"   Potential savings: {optimization['estimated_savings_percentage']:.1f}%")
            for opt in optimization['optimizations']:
                print(f"   - {opt['type']}: {opt.get('savings_percentage', 0)}% savings")
        else:
            print("   Configuration already optimized")

        # Create experiment infrastructure
        print("\n🚀 Launching Experiment Infrastructure:")
        try:
            infrastructure_result = manager.create_experiment_infrastructure(
                experiment_name="demo_fairness_experiment",
                infrastructure_config=infrastructure_config
            )

            print(f"   ✅ Launched {infrastructure_result['launched_resources']} resources")
            print(f"   💰 Estimated cost: ${infrastructure_result['cost_estimate']['total_estimated_cost']:.2f}")

            if infrastructure_result['optimization_applied']:
                print("   🎯 Applied optimizations:")
                for opt in infrastructure_result['optimization_applied']:
                    print(f"     - {opt['type']}")

            # Start auto-scaling if enabled
            if manager.auto_scaler:
                manager.auto_scaler.start_auto_scaling()
                print("   📈 Auto-scaling enabled")

            # Simulate experiment runtime
            print("\n⏳ Simulating experiment execution...")
            experiment_name = "demo_fairness_experiment"

            for i in range(5):
                time.sleep(2)  # Simulate work

                # Monitor costs
                cost_status = manager.monitor_experiment_costs()
                print(f"   [{i+1}/5] Current hourly cost: ${cost_status['current_hourly_cost']:.2f}")

                if manager.auto_scaler:
                    # Run scaling cycle
                    scaling_result = manager.auto_scaler.run_scaling_cycle()
                    if scaling_result['scaling_actions_executed'] > 0:
                        print(f"   📊 Executed {scaling_result['scaling_actions_executed']} scaling actions")

            # Get final status
            print("\n📊 Final Status:")
            status = manager.get_comprehensive_status()

            print(f"   Active resources: {status['resource_metrics']['active_resources']}")
            print(f"   Current hourly cost: ${status['cost_monitoring']['current_hourly_cost']:.2f}")

            if status['cost_monitoring']['budget_status']:
                budget_status = status['cost_monitoring']['budget_status']
                print(f"   Budget used: {budget_status['budget_used_percentage']:.1f}%")

            # Show cost recommendations
            recommendations = status['cost_recommendations']
            if recommendations['recommendations']:
                print("\n💡 Cost Optimization Recommendations:")
                print(f"   Potential monthly savings: ${recommendations['potential_monthly_savings']:.2f}")
                for rec in recommendations['recommendations'][:3]:  # Show top 3
                    print(f"   - {rec['description']}")

            # Terminate infrastructure
            print("\n🔄 Terminating experiment infrastructure...")
            termination_result = manager.terminate_experiment_infrastructure(experiment_name)

            print(f"   ✅ Terminated {termination_result['terminated_resources']} resources")
            print(f"   💰 Final cost: ${termination_result['final_cost']:.2f}")

            cost_diff = termination_result['cost_difference']
            if cost_diff > 0:
                print(f"   📈 Over estimate by: ${cost_diff:.2f}")
            else:
                print(f"   📉 Under estimate by: ${abs(cost_diff):.2f}")

        except ValueError as e:
            print(f"   ❌ Error: {e}")

        print("\n✅ Cloud integration demo completed! 🎉")


if __name__ == "__main__":
    main()
