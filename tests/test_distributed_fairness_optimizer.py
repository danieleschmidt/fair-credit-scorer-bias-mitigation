"""
Comprehensive tests for distributed fairness optimization framework.
"""

import asyncio
import time
from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.distributed_fairness_optimizer import (
    AsyncOptimizer,
    ComputeResource,
    FederatedFairnessOptimizer,
    HyperparameterOptimizer,
    OptimizationBackend,
    OptimizationTask,
    PerformanceMonitor,
    ResourceManager,
    SchedulingStrategy,
    TaskScheduler,
    ThreadPoolOptimizer
)


@pytest.fixture
def sample_data():
    """Create sample dataset for testing."""
    np.random.seed(42)
    
    X = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100),
        'feature3': np.random.randint(0, 3, 100),
        'feature4': np.random.random(100)
    })
    
    y = pd.Series(np.random.binomial(1, 0.6, 100), name='target')
    
    sensitive_attrs = pd.DataFrame({
        'group': np.random.choice(['A', 'B'], 100),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
    })
    
    return X, y, sensitive_attrs


@pytest.fixture
def sample_resource():
    """Create sample compute resource."""
    return ComputeResource(
        resource_id="test_cpu_0",
        resource_type="cpu",
        cores=4,
        memory_gb=8.0
    )


class TestComputeResource:
    """Test compute resource functionality."""
    
    def test_resource_creation(self):
        """Test resource creation and properties."""
        resource = ComputeResource(
            resource_id="gpu_0",
            resource_type="gpu",
            cores=1,
            memory_gb=16.0
        )
        
        assert resource.resource_id == "gpu_0"
        assert resource.resource_type == "gpu"
        assert resource.cores == 1
        assert resource.memory_gb == 16.0
        assert resource.available is True
        assert resource.utilization == 0.0
    
    def test_can_handle_task(self):
        """Test resource capability checking."""
        resource = ComputeResource(
            resource_id="test_cpu",
            resource_type="cpu",
            cores=4,
            memory_gb=8.0,
            utilization=0.5
        )
        
        # Task within resource limits
        light_task = {'cores': 2, 'memory_gb': 4.0}
        assert resource.can_handle_task(light_task) is True
        
        # Task exceeding core limit
        heavy_task = {'cores': 8, 'memory_gb': 4.0}
        assert resource.can_handle_task(heavy_task) is False
        
        # Task exceeding memory limit
        memory_task = {'cores': 2, 'memory_gb': 16.0}
        assert resource.can_handle_task(memory_task) is False
        
        # High utilization resource
        resource.utilization = 0.9
        assert resource.can_handle_task(light_task) is False


class TestResourceManager:
    """Test resource manager functionality."""
    
    def test_resource_discovery(self):
        """Test automatic resource discovery."""
        manager = ResourceManager()
        
        # Should discover at least CPU resources
        assert len(manager.resources) > 0
        
        # Check CPU resources exist
        cpu_resources = [r for r in manager.resources.values() if r.resource_type == "cpu"]
        assert len(cpu_resources) > 0
        
        # All resources should be initially available
        for resource in manager.resources.values():
            assert resource.available is True
            assert resource.utilization == 0.0
    
    def test_resource_allocation_and_release(self):
        """Test resource allocation and release."""
        manager = ResourceManager()
        
        # Allocate resource
        task_requirements = {'cores': 1, 'memory_gb': 2.0}
        resource = manager.allocate_resource(task_requirements)
        
        assert resource is not None
        assert resource.available is False
        assert resource.utilization > 0
        
        # Release resource
        manager.release_resource(resource.resource_id)
        assert resource.available is True
        assert resource.utilization >= 0  # May not be exactly 0 due to adjustment
    
    def test_resource_allocation_failure(self):
        """Test resource allocation when no resources available."""
        manager = ResourceManager()
        
        # Make all resources unavailable
        for resource in manager.resources.values():
            resource.available = False
        
        # Should fail to allocate
        task_requirements = {'cores': 1, 'memory_gb': 1.0}
        resource = manager.allocate_resource(task_requirements)
        assert resource is None
    
    def test_get_resource_utilization(self):
        """Test resource utilization reporting."""
        manager = ResourceManager()
        
        utilization = manager.get_resource_utilization()
        
        assert isinstance(utilization, dict)
        assert len(utilization) == len(manager.resources)
        
        # All should start at 0 utilization
        for util in utilization.values():
            assert util == 0.0


class TestTaskScheduler:
    """Test task scheduler functionality."""
    
    def test_task_submission_and_retrieval(self):
        """Test task submission and retrieval."""
        manager = ResourceManager()
        scheduler = TaskScheduler(manager)
        
        task = OptimizationTask(
            task_id="test_task",
            algorithm=LogisticRegression(),
            parameters={'C': 1.0},
            data=(pd.DataFrame(), pd.Series(), pd.DataFrame()),
            priority=2
        )
        
        scheduler.submit_task(task)
        
        retrieved_task = scheduler.get_next_task()
        assert retrieved_task is not None
        assert retrieved_task.task_id == "test_task"
        assert retrieved_task.priority == 2
    
    def test_task_priority_ordering(self):
        """Test task priority ordering."""
        manager = ResourceManager()
        scheduler = TaskScheduler(manager)
        
        # Submit tasks with different priorities
        low_priority_task = OptimizationTask(
            task_id="low",
            algorithm=LogisticRegression(),
            parameters={},
            data=(pd.DataFrame(), pd.Series(), pd.DataFrame()),
            priority=1
        )
        
        high_priority_task = OptimizationTask(
            task_id="high",
            algorithm=LogisticRegression(),
            parameters={},
            data=(pd.DataFrame(), pd.Series(), pd.DataFrame()),
            priority=3
        )
        
        # Submit in low-priority-first order
        scheduler.submit_task(low_priority_task)
        scheduler.submit_task(high_priority_task)
        
        # Should retrieve high priority first
        first_task = scheduler.get_next_task()
        assert first_task.task_id == "high"
        
        second_task = scheduler.get_next_task()
        assert second_task.task_id == "low"
    
    def test_task_allocation_with_resources(self):
        """Test task allocation with resource availability."""
        manager = ResourceManager()
        scheduler = TaskScheduler(manager)
        
        task = OptimizationTask(
            task_id="test_task",
            algorithm=LogisticRegression(),
            parameters={},
            data=(pd.DataFrame(), pd.Series(), pd.DataFrame()),
            requirements={'cores': 1, 'memory_gb': 1.0}
        )
        
        scheduler.submit_task(task)
        
        allocation = scheduler.allocate_task()
        assert allocation is not None
        
        allocated_task, allocated_resource = allocation
        assert allocated_task.task_id == "test_task"
        assert allocated_resource is not None
        assert allocated_resource.available is False
    
    def test_task_completion_tracking(self):
        """Test task completion tracking."""
        manager = ResourceManager()
        scheduler = TaskScheduler(manager)
        
        # Submit and allocate task
        task = OptimizationTask(
            task_id="test_task",
            algorithm=LogisticRegression(),
            parameters={},
            data=(pd.DataFrame(), pd.Series(), pd.DataFrame())
        )
        
        scheduler.submit_task(task)
        allocation = scheduler.allocate_task()
        assert allocation is not None
        
        allocated_task, _ = allocation
        
        # Complete task
        from src.distributed_fairness_optimizer import OptimizationResult
        result = OptimizationResult(
            task_id="test_task",
            parameters={},
            performance_metrics={'accuracy': 0.8},
            fairness_metrics={},
            execution_time=1.0,
            resource_usage={},
            success=True
        )
        
        scheduler.complete_task("test_task", result)
        
        # Task should be in completed list
        assert len(scheduler.completed_tasks) == 1
        assert scheduler.completed_tasks[0].task_id == "test_task"
        assert scheduler.completed_tasks[0].result == result


class TestThreadPoolOptimizer:
    """Test thread pool optimizer functionality."""
    
    def test_single_task_execution(self, sample_data):
        """Test execution of single optimization task."""
        X, y, sensitive_attrs = sample_data
        
        optimizer = ThreadPoolOptimizer(max_workers=2)
        manager = ResourceManager()
        scheduler = TaskScheduler(manager)
        
        task = OptimizationTask(
            task_id="test_task",
            algorithm=LogisticRegression(),
            parameters={'C': 1.0, 'random_state': 42},
            data=(X, y, sensitive_attrs)
        )
        
        results = optimizer.optimize([task], scheduler)
        
        assert len(results) == 1
        result = results[0]
        assert result.success is True
        assert result.task_id == "test_task"
        assert 'accuracy' in result.performance_metrics
        assert len(result.fairness_metrics) > 0
        assert result.execution_time > 0
    
    def test_multiple_task_execution(self, sample_data):
        """Test execution of multiple optimization tasks."""
        X, y, sensitive_attrs = sample_data
        
        optimizer = ThreadPoolOptimizer(max_workers=2)
        manager = ResourceManager()
        scheduler = TaskScheduler(manager)
        
        # Create multiple tasks with different parameters
        tasks = []
        for i, c_value in enumerate([0.1, 1.0, 10.0]):
            task = OptimizationTask(
                task_id=f"task_{i}",
                algorithm=LogisticRegression(),
                parameters={'C': c_value, 'random_state': 42},
                data=(X, y, sensitive_attrs)
            )
            tasks.append(task)
        
        results = optimizer.optimize(tasks, scheduler)
        
        assert len(results) == 3
        
        # All tasks should complete successfully
        successful_results = [r for r in results if r.success]
        assert len(successful_results) == 3
        
        # Check that different parameters were used
        c_values = [r.parameters['C'] for r in results]
        assert set(c_values) == {0.1, 1.0, 10.0}
    
    def test_task_execution_with_failure(self, sample_data):
        """Test handling of task execution failures."""
        X, y, sensitive_attrs = sample_data
        
        optimizer = ThreadPoolOptimizer(max_workers=1)
        manager = ResourceManager()
        scheduler = TaskScheduler(manager)
        
        # Create task with invalid parameters
        task = OptimizationTask(
            task_id="failing_task",
            algorithm=LogisticRegression(),
            parameters={'C': -1.0},  # Invalid parameter
            data=(X, y, sensitive_attrs)
        )
        
        results = optimizer.optimize([task], scheduler)
        
        assert len(results) == 1
        result = results[0]
        assert result.success is False
        assert result.error_message is not None
        assert result.task_id == "failing_task"


class TestAsyncOptimizer:
    """Test async optimizer functionality."""
    
    @pytest.mark.asyncio
    async def test_async_task_execution(self, sample_data):
        """Test async execution of optimization tasks."""
        X, y, sensitive_attrs = sample_data
        
        optimizer = AsyncOptimizer(max_concurrent_tasks=2)
        manager = ResourceManager()
        scheduler = TaskScheduler(manager)
        
        task = OptimizationTask(
            task_id="async_task",
            algorithm=LogisticRegression(),
            parameters={'C': 1.0, 'random_state': 42},
            data=(X, y, sensitive_attrs)
        )
        
        results = optimizer.optimize([task], scheduler)
        
        assert len(results) == 1
        result = results[0]
        assert result.success is True
        assert result.task_id == "async_task"
        assert 'accuracy' in result.performance_metrics
    
    @pytest.mark.asyncio
    async def test_concurrent_async_execution(self, sample_data):
        """Test concurrent async execution."""
        X, y, sensitive_attrs = sample_data
        
        optimizer = AsyncOptimizer(max_concurrent_tasks=3)
        manager = ResourceManager()
        scheduler = TaskScheduler(manager)
        
        # Create multiple tasks
        tasks = []
        for i in range(3):
            task = OptimizationTask(
                task_id=f"async_task_{i}",
                algorithm=LogisticRegression(),
                parameters={'C': 1.0, 'random_state': 42 + i},
                data=(X, y, sensitive_attrs)
            )
            tasks.append(task)
        
        start_time = time.time()
        results = optimizer.optimize(tasks, scheduler)
        execution_time = time.time() - start_time
        
        assert len(results) == 3
        
        # Should be faster than sequential execution (allowing for overhead)
        # This is a rough check - actual speedup depends on system
        assert execution_time < 10.0  # Should complete reasonably quickly


class TestHyperparameterOptimizer:
    """Test hyperparameter optimizer functionality."""
    
    def test_basic_hyperparameter_optimization(self, sample_data):
        """Test basic hyperparameter optimization."""
        X, y, sensitive_attrs = sample_data
        
        optimizer = HyperparameterOptimizer(
            backend=OptimizationBackend.THREADING,
            max_evaluations=6  # Small for testing
        )
        
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'solver': ['liblinear', 'lbfgs']
        }
        
        results = optimizer.optimize(
            algorithm=LogisticRegression(random_state=42),
            param_grid=param_grid,
            X=X,
            y=y,
            sensitive_attrs=sensitive_attrs,
            scoring_metric='accuracy'
        )
        
        assert 'best_parameters' in results
        assert 'best_score' in results
        assert 'total_evaluations' in results
        assert 'successful_evaluations' in results
        assert 'optimization_time' in results
        
        # Should evaluate all combinations (3 * 2 = 6)
        assert results['total_evaluations'] == 6
        assert results['successful_evaluations'] > 0
        
        # Best parameters should be from the grid
        best_params = results['best_parameters']
        assert best_params['C'] in [0.1, 1.0, 10.0]
        assert best_params['solver'] in ['liblinear', 'lbfgs']
    
    def test_hyperparameter_optimization_with_evaluation_limit(self, sample_data):
        """Test hyperparameter optimization with evaluation limit."""
        X, y, sensitive_attrs = sample_data
        
        optimizer = HyperparameterOptimizer(
            backend=OptimizationBackend.THREADING,
            max_evaluations=3  # Limit evaluations
        )
        
        # Large parameter grid
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'solver': ['liblinear', 'lbfgs'],
            'penalty': ['l1', 'l2']
        }
        
        results = optimizer.optimize(
            algorithm=LogisticRegression(random_state=42),
            param_grid=param_grid,
            X=X,
            y=y,
            sensitive_attrs=sensitive_attrs
        )
        
        # Should be limited to max_evaluations
        assert results['total_evaluations'] == 3
    
    def test_hyperparameter_optimization_with_fairness_constraints(self, sample_data):
        """Test hyperparameter optimization with fairness constraints."""
        X, y, sensitive_attrs = sample_data
        
        optimizer = HyperparameterOptimizer(
            backend=OptimizationBackend.THREADING,
            max_evaluations=4
        )
        
        param_grid = {
            'C': [0.1, 1.0],
            'solver': ['liblinear', 'lbfgs']
        }
        
        # Define fairness constraints
        fairness_constraints = {
            'demographic_parity_difference': 0.1,  # Max 10% difference
            'equalized_odds_difference': 0.1
        }
        
        results = optimizer.optimize(
            algorithm=LogisticRegression(random_state=42),
            param_grid=param_grid,
            X=X,
            y=y,
            sensitive_attrs=sensitive_attrs,
            fairness_constraints=fairness_constraints
        )
        
        # Should still find a result (may not satisfy constraints with small data)
        assert 'best_parameters' in results
        assert 'best_fairness_metrics' in results


class TestFederatedFairnessOptimizer:
    """Test federated fairness optimizer functionality."""
    
    def test_participant_registration(self, sample_data):
        """Test participant registration."""
        X, y, sensitive_attrs = sample_data
        
        optimizer = FederatedFairnessOptimizer(num_rounds=3)
        
        # Register participant
        optimizer.register_participant("participant_1", (X, y, sensitive_attrs))
        
        assert "participant_1" in optimizer.participants
        participant_data = optimizer.participants["participant_1"]
        assert participant_data['data_size'] == len(X)
        assert participant_data['local_model'] is None
    
    def test_federated_optimization_with_multiple_participants(self):
        """Test federated optimization with multiple participants."""
        optimizer = FederatedFairnessOptimizer(num_rounds=2, min_participants=2)
        
        # Create data for multiple participants
        np.random.seed(42)
        for i in range(3):
            X = pd.DataFrame({
                'feature1': np.random.normal(i, 1, 50),  # Different distributions
                'feature2': np.random.random(50)
            })
            y = pd.Series(np.random.binomial(1, 0.5 + i * 0.1, 50))
            sensitive_attrs = pd.DataFrame({
                'group': np.random.choice(['A', 'B'], 50)
            })
            
            optimizer.register_participant(f"participant_{i}", (X, y, sensitive_attrs))
        
        # Run federated optimization
        results = optimizer.federated_optimize(LogisticRegression(random_state=42))
        
        assert 'global_model' in results
        assert 'num_rounds' in results
        assert 'num_participants' in results
        assert 'round_results' in results
        assert 'final_fairness_metrics' in results
        
        assert results['num_rounds'] == 2
        assert results['num_participants'] == 3
        assert len(results['round_results']) == 2
        
        # Each round should have fairness metrics
        for round_result in results['round_results']:
            assert 'local_fairness_metrics' in round_result
            assert 'global_fairness_metrics' in round_result
    
    def test_federated_optimization_insufficient_participants(self):
        """Test federated optimization with insufficient participants."""
        optimizer = FederatedFairnessOptimizer(min_participants=3)
        
        # Register only one participant
        X = pd.DataFrame({'feature1': [1, 2, 3]})
        y = pd.Series([0, 1, 0])
        sensitive_attrs = pd.DataFrame({'group': ['A', 'B', 'A']})
        
        optimizer.register_participant("participant_1", (X, y, sensitive_attrs))
        
        # Should raise error for insufficient participants
        with pytest.raises(ValueError, match="Need at least 3 participants"):
            optimizer.federated_optimize(LogisticRegression())


class TestPerformanceMonitor:
    """Test performance monitoring functionality."""
    
    def test_performance_recording(self):
        """Test performance metrics recording."""
        monitor = PerformanceMonitor()
        
        timestamp = datetime.now()
        monitor.record_performance(
            timestamp=timestamp,
            tasks_completed=10,
            tasks_failed=2,
            average_execution_time=0.5,
            resource_utilization={'cpu_0': 0.7, 'cpu_1': 0.3}
        )
        
        assert len(monitor.metrics_history) == 1
        assert len(monitor.resource_utilization_history) == 1
        
        metrics = monitor.metrics_history[0]
        assert metrics['tasks_completed'] == 10
        assert metrics['tasks_failed'] == 2
        assert metrics['success_rate'] == 10/12  # 10 successes out of 12 total
        assert metrics['average_execution_time'] == 0.5
    
    def test_performance_report_generation(self):
        """Test performance report generation."""
        monitor = PerformanceMonitor()
        
        # Add some performance data
        timestamp = datetime.now()
        monitor.record_performance(
            timestamp=timestamp,
            tasks_completed=15,
            tasks_failed=3,
            average_execution_time=0.8,
            resource_utilization={'cpu_0': 0.6, 'gpu_0': 0.9}
        )
        
        report = monitor.generate_performance_report()
        
        assert "Distributed Optimization Performance Report" in report
        assert "Current Performance" in report
        assert "Resource Utilization" in report
        assert "Tasks Completed: 15" in report
        assert "Tasks Failed: 3" in report
        assert "cpu_0: 60.0%" in report
        assert "gpu_0: 90.0%" in report


@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration tests with realistic scenarios."""
    
    def test_end_to_end_hyperparameter_optimization(self):
        """Test complete hyperparameter optimization workflow."""
        # Generate larger dataset for more realistic testing
        np.random.seed(42)
        
        X = pd.DataFrame({
            'age': np.random.randint(18, 80, 200),
            'income': np.random.lognormal(10, 1, 200),
            'credit_score': np.random.randint(300, 850, 200),
            'debt_ratio': np.random.beta(2, 5, 200)
        })
        
        # Create target with some correlation to features
        y = pd.Series(
            np.random.binomial(1, 1 / (1 + np.exp(-(X['income'] / 50000 - 1)))),
            name='approved'
        )
        
        sensitive_attrs = pd.DataFrame({
            'gender': np.random.choice(['M', 'F'], 200),
            'race': np.random.choice(['White', 'Black', 'Hispanic'], 200)
        })
        
        # Run hyperparameter optimization
        optimizer = HyperparameterOptimizer(
            backend=OptimizationBackend.THREADING,
            max_evaluations=8
        )
        
        param_grid = {
            'n_estimators': [10, 50],
            'max_depth': [3, 5],
            'min_samples_split': [2, 5]
        }
        
        results = optimizer.optimize(
            algorithm=RandomForestClassifier(random_state=42),
            param_grid=param_grid,
            X=X,
            y=y,
            sensitive_attrs=sensitive_attrs,
            scoring_metric='accuracy'
        )
        
        # Verify comprehensive results
        assert results['successful_evaluations'] == 8
        assert results['best_score'] > 0.5  # Should be better than random
        assert 'n_estimators' in results['best_parameters']
        assert 'max_depth' in results['best_parameters']
        
        # Check fairness metrics are computed
        assert len(results['best_fairness_metrics']) == 2  # Two sensitive attributes
        
        # Verify all results are available
        assert len(results['all_results']) == 8
    
    def test_performance_comparison_different_backends(self, sample_data):
        """Test performance comparison between different backends."""
        X, y, sensitive_attrs = sample_data
        
        param_grid = {'C': [0.1, 1.0, 10.0]}
        
        # Test threading backend
        threading_optimizer = HyperparameterOptimizer(
            backend=OptimizationBackend.THREADING,
            max_evaluations=3
        )
        
        start_time = time.time()
        threading_results = threading_optimizer.optimize(
            algorithm=LogisticRegression(random_state=42),
            param_grid=param_grid,
            X=X,
            y=y,
            sensitive_attrs=sensitive_attrs
        )
        threading_time = time.time() - start_time
        
        # Test async backend
        async_optimizer = HyperparameterOptimizer(
            backend=OptimizationBackend.ASYNCIO,
            max_evaluations=3
        )
        
        start_time = time.time()
        async_results = async_optimizer.optimize(
            algorithm=LogisticRegression(random_state=42),
            param_grid=param_grid,
            X=X,
            y=y,
            sensitive_attrs=sensitive_attrs
        )
        async_time = time.time() - start_time
        
        # Both should complete successfully
        assert threading_results['successful_evaluations'] == 3
        assert async_results['successful_evaluations'] == 3
        
        # Both should find reasonable results
        assert threading_results['best_score'] > 0.4
        assert async_results['best_score'] > 0.4
        
        # Execution times should be reasonable
        assert threading_time < 30.0  # Should complete in reasonable time
        assert async_time < 30.0
    
    def test_distributed_optimization_with_resource_constraints(self, sample_data):
        """Test distributed optimization with limited resources."""
        X, y, sensitive_attrs = sample_data
        
        # Create optimizer with limited resources
        optimizer = HyperparameterOptimizer(
            backend=OptimizationBackend.THREADING,
            max_evaluations=6
        )
        
        # Override resource manager to simulate limited resources
        original_allocate = optimizer.resource_manager.allocate_resource
        allocation_count = 0
        
        def limited_allocate(requirements):
            nonlocal allocation_count
            allocation_count += 1
            # Fail some allocations to simulate resource constraints
            if allocation_count % 3 == 0:
                return None
            return original_allocate(requirements)
        
        optimizer.resource_manager.allocate_resource = limited_allocate
        
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'solver': ['liblinear', 'lbfgs']
        }
        
        results = optimizer.optimize(
            algorithm=LogisticRegression(random_state=42),
            param_grid=param_grid,
            X=X,
            y=y,
            sensitive_attrs=sensitive_attrs
        )
        
        # Should still complete successfully despite resource constraints
        assert results['successful_evaluations'] > 0
        assert 'best_parameters' in results


if __name__ == "__main__":
    pytest.main([__file__])