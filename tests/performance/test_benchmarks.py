"""Performance benchmarks for the fair credit scoring system."""

import time
from pathlib import Path
import tempfile

import pytest

from src.evaluate_fairness import run_pipeline
from src.data_loader_preprocessor import generate_data


class TestPerformanceBenchmarks:
    """Performance benchmarks for core functionality."""

    @pytest.mark.performance
    def test_data_generation_performance(self, benchmark):
        """Benchmark data generation performance."""
        def generate_test_data():
            with tempfile.TemporaryDirectory() as temp_dir:
                data_path = Path(temp_dir) / "benchmark_data.csv"
                return generate_data(str(data_path), n_samples=10000)
        
        result = benchmark(generate_test_data)
        assert len(result) == 10000

    @pytest.mark.performance
    def test_baseline_training_performance(self, benchmark):
        """Benchmark baseline model training performance."""
        def train_baseline():
            with tempfile.TemporaryDirectory() as temp_dir:
                data_path = Path(temp_dir) / "benchmark_data.csv"
                return run_pipeline(
                    method="baseline",
                    data_path=str(data_path),
                    test_size=0.3,
                    random_state=42
                )
        
        result = benchmark(train_baseline)
        assert "accuracy" in result

    @pytest.mark.performance
    def test_fairness_metrics_performance(self, benchmark):
        """Benchmark fairness metrics calculation performance."""
        from src.fairness_metrics import compute_fairness_metrics
        import numpy as np
        
        # Generate test data
        n_samples = 10000
        y_true = np.random.randint(0, 2, n_samples)
        y_pred = np.random.randint(0, 2, n_samples)
        y_prob = np.random.random(n_samples)
        protected_attr = np.random.randint(0, 2, n_samples)
        
        def compute_metrics():
            return compute_fairness_metrics(
                y_true, y_pred, y_prob, protected_attr
            )
        
        result = benchmark(compute_metrics)
        assert "demographic_parity_difference" in result

    @pytest.mark.performance
    def test_large_dataset_performance(self):
        """Test performance with larger datasets."""
        start_time = time.time()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "large_data.csv"
            
            # Generate larger dataset
            data = generate_data(str(data_path), n_samples=50000)
            
            # Run pipeline
            results = run_pipeline(
                method="baseline",
                data_path=str(data_path),
                test_size=0.2,
                random_state=42
            )
        
        elapsed_time = time.time() - start_time
        
        # Should complete within reasonable time (adjust as needed)
        assert elapsed_time < 60  # 60 seconds max
        assert "accuracy" in results

    @pytest.mark.performance
    def test_memory_usage_large_dataset(self):
        """Test memory usage doesn't explode with large datasets."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "memory_test_data.csv"
            
            # Generate moderately large dataset
            generate_data(str(data_path), n_samples=20000)
            
            # Run pipeline
            run_pipeline(
                method="baseline",
                data_path=str(data_path),
                test_size=0.3,
                random_state=42
            )
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (adjust threshold as needed)
        assert memory_increase < 500  # 500 MB increase max