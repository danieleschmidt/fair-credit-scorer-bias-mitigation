"""
Comprehensive tests for the model registry.

Tests model versioning, metadata tracking, and deployment management.
"""

import json
import pickle
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.api.model_registry import ModelRegistry, ModelMetadata
from src.fairness_metrics import compute_fairness_metrics


class TestModelMetadata:
    """Test suite for ModelMetadata class."""
    
    def test_metadata_creation(self):
        """Test metadata object creation."""
        metadata = ModelMetadata(
            name="test_model",
            version="1.0.0",
            algorithm="LogisticRegression",
            training_method="baseline",
            metrics={"accuracy": 0.85, "precision": 0.82},
            fairness_metrics={"demographic_parity_difference": 0.05},
            feature_names=["age", "income", "credit_score"]
        )
        
        assert metadata.name == "test_model"
        assert metadata.version == "1.0.0"
        assert metadata.algorithm == "LogisticRegression"
        assert metadata.metrics["accuracy"] == 0.85
        assert len(metadata.feature_names) == 3
    
    def test_metadata_to_dict(self):
        """Test metadata dictionary conversion."""
        metadata = ModelMetadata(
            name="test_model",
            version="1.0.0",
            algorithm="LogisticRegression",
            training_method="baseline",
            metrics={"accuracy": 0.85},
            fairness_metrics={"demographic_parity_difference": 0.05},
            feature_names=["age", "income"]
        )
        
        data = metadata.to_dict()
        
        assert isinstance(data, dict)
        assert data["name"] == "test_model"
        assert data["version"] == "1.0.0"
        assert "created_at" in data
        assert isinstance(data["created_at"], str)  # ISO format
    
    def test_metadata_from_dict(self):
        """Test metadata creation from dictionary."""
        data = {
            "name": "test_model",
            "version": "1.0.0",
            "algorithm": "LogisticRegression",
            "training_method": "baseline",
            "metrics": {"accuracy": 0.85},
            "fairness_metrics": {"demographic_parity_difference": 0.05},
            "feature_names": ["age", "income"],
            "created_at": "2024-01-01T12:00:00",
            "tags": {"stage": "production"},
            "model_hash": "abc123"
        }
        
        metadata = ModelMetadata.from_dict(data)
        
        assert metadata.name == "test_model"
        assert metadata.version == "1.0.0"
        assert metadata.tags["stage"] == "production"
        assert metadata.model_hash == "abc123"


class TestModelRegistry:
    """Test suite for ModelRegistry class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry = ModelRegistry(self.temp_dir)
        
        # Create sample data
        self.X_test = pd.DataFrame({
            'age': [25, 35, 45],
            'income': [40000, 50000, 60000],
            'credit_score': [650, 700, 750],
            'debt_to_income': [0.3, 0.2, 0.1]
        })
        self.y_test = pd.Series([0, 1, 1])
        self.protected = pd.Series(['A', 'B', 'A'])
        
        # Create sample model
        self.model = LogisticRegression(random_state=42)
        self.model.fit(self.X_test, self.y_test)
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_registry_initialization(self):
        """Test registry initialization."""
        registry_path = Path(self.temp_dir)
        
        assert registry_path.exists()
        assert (registry_path / "models").exists()
        assert (registry_path / "metadata").exists()
        assert (registry_path / "experiments").exists()
        
        config_file = registry_path / "config.json"
        assert config_file.exists()
    
    def test_register_model_without_test_data(self):
        """Test model registration without test data."""
        model_id = self.registry.register_model(
            model=self.model,
            name="test_model",
            version="1.0.0",
            algorithm="LogisticRegression",
            training_method="baseline"
        )
        
        assert model_id == "test_model_v1.0.0"
        
        # Verify model file exists
        model_file = Path(self.temp_dir) / "models" / f"{model_id}.pkl"
        assert model_file.exists()
        
        # Verify metadata file exists
        metadata_file = Path(self.temp_dir) / "metadata" / f"{model_id}.json"
        assert metadata_file.exists()
    
    def test_register_model_with_test_data(self):
        """Test model registration with test data."""
        test_data = (self.X_test, self.y_test, self.protected)
        
        model_id = self.registry.register_model(
            model=self.model,
            name="test_model",
            version="1.0.0",
            algorithm="LogisticRegression",
            training_method="baseline",
            test_data=test_data,
            tags={"stage": "development"}
        )
        
        # Load and verify metadata
        model, metadata = self.registry.load_model("test_model", "1.0.0")
        
        assert metadata.name == "test_model"
        assert metadata.version == "1.0.0"
        assert "accuracy" in metadata.metrics
        assert "demographic_parity_difference" in metadata.fairness_metrics
        assert metadata.tags["stage"] == "development"
        assert len(metadata.feature_names) == 4
    
    def test_load_model(self):
        """Test model loading."""
        # Register a model first
        model_id = self.registry.register_model(
            model=self.model,
            name="test_model",
            version="1.0.0",
            algorithm="LogisticRegression",
            training_method="baseline"
        )
        
        # Load the model
        loaded_model, metadata = self.registry.load_model("test_model", "1.0.0")
        
        assert loaded_model is not None
        assert hasattr(loaded_model, 'predict')
        assert metadata.name == "test_model"
        assert metadata.version == "1.0.0"
    
    def test_load_model_latest_version(self):
        """Test loading latest version of a model."""
        # Register multiple versions
        self.registry.register_model(
            model=self.model,
            name="test_model",
            version="1.0.0",
            algorithm="LogisticRegression",
            training_method="baseline"
        )
        
        self.registry.register_model(
            model=self.model,
            name="test_model",
            version="1.1.0",
            algorithm="LogisticRegression",
            training_method="reweight"
        )
        
        # Load without specifying version (should get latest)
        loaded_model, metadata = self.registry.load_model("test_model")
        
        assert metadata.version == "1.1.0"
        assert metadata.training_method == "reweight"
    
    def test_list_models(self):
        """Test listing models."""
        # Register multiple models
        self.registry.register_model(
            model=self.model,
            name="model_a",
            version="1.0.0",
            algorithm="LogisticRegression",
            training_method="baseline"
        )
        
        self.registry.register_model(
            model=self.model,
            name="model_b",
            version="1.0.0",
            algorithm="LogisticRegression",
            training_method="baseline"
        )
        
        models = self.registry.list_models()
        
        assert len(models) == 2
        model_names = [m["name"] for m in models]
        assert "model_a" in model_names
        assert "model_b" in model_names
    
    def test_list_models_with_filter(self):
        """Test listing models with name filter."""
        # Register models
        self.registry.register_model(
            model=self.model,
            name="credit_model_v1",
            version="1.0.0",
            algorithm="LogisticRegression",
            training_method="baseline"
        )
        
        self.registry.register_model(
            model=self.model,
            name="fraud_model_v1",
            version="1.0.0",
            algorithm="LogisticRegression",
            training_method="baseline"
        )
        
        credit_models = self.registry.list_models(name_filter="credit")
        fraud_models = self.registry.list_models(name_filter="fraud")
        
        assert len(credit_models) == 1
        assert len(fraud_models) == 1
        assert credit_models[0]["name"] == "credit_model_v1"
        assert fraud_models[0]["name"] == "fraud_model_v1"
    
    def test_compare_models(self):
        """Test model comparison."""
        # Register models with test data
        test_data = (self.X_test, self.y_test, self.protected)
        
        model_id1 = self.registry.register_model(
            model=self.model,
            name="model_a",
            version="1.0.0",
            algorithm="LogisticRegression",
            training_method="baseline",
            test_data=test_data
        )
        
        model_id2 = self.registry.register_model(
            model=self.model,
            name="model_b",
            version="1.0.0",
            algorithm="LogisticRegression",
            training_method="reweight",
            test_data=test_data
        )
        
        comparison = self.registry.compare_models([model_id1, model_id2])
        
        assert "models" in comparison
        assert "metrics_comparison" in comparison
        assert "fairness_comparison" in comparison
        assert "recommendation" in comparison
        assert len(comparison["models"]) == 2
    
    def test_promote_model(self):
        """Test model promotion."""
        # Register a model
        model_id = self.registry.register_model(
            model=self.model,
            name="test_model",
            version="1.0.0",
            algorithm="LogisticRegression",
            training_method="baseline"
        )
        
        # Promote to production
        success = self.registry.promote_model(model_id, "production")
        
        assert success == True
        
        # Verify promotion in metadata
        _, metadata = self.registry.load_model("test_model", "1.0.0")
        assert metadata.tags["stage"] == "production"
        assert "promoted_at" in metadata.tags
        
        # Verify production symlink exists
        production_file = Path(self.temp_dir) / "production_model"
        assert production_file.exists()
    
    def test_delete_model(self):
        """Test model deletion."""
        # Register a model
        model_id = self.registry.register_model(
            model=self.model,
            name="test_model",
            version="1.0.0",
            algorithm="LogisticRegression",
            training_method="baseline"
        )
        
        # Verify files exist
        model_file = Path(self.temp_dir) / "models" / f"{model_id}.pkl"
        metadata_file = Path(self.temp_dir) / "metadata" / f"{model_id}.json"
        assert model_file.exists()
        assert metadata_file.exists()
        
        # Delete model
        success = self.registry.delete_model(model_id)
        
        assert success == True
        assert not model_file.exists()
        assert not metadata_file.exists()
    
    def test_get_latest_version(self):
        """Test getting latest version."""
        # Register multiple versions
        self.registry.register_model(
            model=self.model,
            name="test_model",
            version="1.0.0",
            algorithm="LogisticRegression",
            training_method="baseline"
        )
        
        self.registry.register_model(
            model=self.model,
            name="test_model",
            version="1.2.0",
            algorithm="LogisticRegression",
            training_method="baseline"
        )
        
        self.registry.register_model(
            model=self.model,
            name="test_model",
            version="1.1.0",
            algorithm="LogisticRegression",
            training_method="baseline"
        )
        
        latest_version = self.registry.get_latest_version("test_model")
        assert latest_version == "1.2.0"
    
    def test_model_hash_verification(self):
        """Test model hash verification."""
        # Register model
        model_id = self.registry.register_model(
            model=self.model,
            name="test_model",
            version="1.0.0",
            algorithm="LogisticRegression",
            training_method="baseline"
        )
        
        # Load model (should verify hash)
        loaded_model, metadata = self.registry.load_model("test_model", "1.0.0")
        
        # Hash should match
        current_hash = self.registry._compute_model_hash(loaded_model)
        assert current_hash == metadata.model_hash
    
    def test_recommend_best_model(self):
        """Test model recommendation logic."""
        test_data = (self.X_test, self.y_test, self.protected)
        
        # Register models with different performance
        model_id1 = self.registry.register_model(
            model=self.model,
            name="model_a",
            version="1.0.0",
            algorithm="LogisticRegression",
            training_method="baseline",
            test_data=test_data
        )
        
        model_id2 = self.registry.register_model(
            model=self.model,
            name="model_b",
            version="1.0.0",
            algorithm="LogisticRegression",
            training_method="reweight",
            test_data=test_data
        )
        
        recommendation = self.registry._recommend_best_model([model_id1, model_id2])
        
        assert recommendation in [model_id1, model_id2]
    
    def test_error_handling_nonexistent_model(self):
        """Test error handling for non-existent models."""
        with pytest.raises(FileNotFoundError):
            self.registry.load_model("nonexistent_model", "1.0.0")
    
    def test_error_handling_invalid_version(self):
        """Test error handling for invalid versions."""
        # Register a model
        self.registry.register_model(
            model=self.model,
            name="test_model",
            version="1.0.0",
            algorithm="LogisticRegression",
            training_method="baseline"
        )
        
        with pytest.raises(FileNotFoundError):
            self.registry.load_model("test_model", "2.0.0")


@pytest.mark.integration
class TestModelRegistryIntegration:
    """Integration tests for model registry."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry = ModelRegistry(self.temp_dir)
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('src.api.model_registry.compute_fairness_metrics')
    def test_end_to_end_model_lifecycle(self, mock_compute_fairness):
        """Test complete model lifecycle."""
        # Mock fairness metrics computation
        mock_overall = pd.Series({
            'demographic_parity_difference': 0.05,
            'equalized_odds_difference': 0.08,
            'accuracy_difference': 0.02
        })
        mock_by_group = pd.DataFrame({
            'accuracy': [0.85, 0.82]
        })
        mock_compute_fairness.return_value = (mock_overall, mock_by_group)
        
        # Create test data
        X_test = pd.DataFrame({
            'age': [25, 35, 45],
            'income': [40000, 50000, 60000]
        })
        y_test = pd.Series([0, 1, 1])
        protected = pd.Series(['A', 'B', 'A'])
        
        # Create and train model
        model = LogisticRegression(random_state=42)
        model.fit(X_test, y_test)
        
        # 1. Register model
        model_id = self.registry.register_model(
            model=model,
            name="credit_model",
            version="1.0.0",
            algorithm="LogisticRegression",
            training_method="baseline",
            test_data=(X_test, y_test, protected),
            tags={"purpose": "credit_scoring"}
        )
        
        # 2. Load and verify
        loaded_model, metadata = self.registry.load_model("credit_model", "1.0.0")
        assert loaded_model is not None
        assert metadata.name == "credit_model"
        
        # 3. Register improved version
        improved_model_id = self.registry.register_model(
            model=model,
            name="credit_model",
            version="1.1.0",
            algorithm="LogisticRegression",
            training_method="reweight",
            test_data=(X_test, y_test, protected)
        )
        
        # 4. Compare models
        comparison = self.registry.compare_models([model_id, improved_model_id])
        assert len(comparison["models"]) == 2
        
        # 5. Promote best model
        best_model = comparison["recommendation"]
        if best_model:
            success = self.registry.promote_model(best_model, "production")
            assert success == True
        
        # 6. List all models
        models = self.registry.list_models()
        assert len(models) == 2
        
        # 7. Clean up
        self.registry.delete_model(model_id)
        self.registry.delete_model(improved_model_id)


# Performance tests
@pytest.mark.performance
class TestModelRegistryPerformance:
    """Performance tests for model registry."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry = ModelRegistry(self.temp_dir)
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_registration_performance(self):
        """Test model registration performance."""
        # Create test model and data
        X_test = pd.DataFrame({
            'feature_' + str(i): np.random.random(1000)
            for i in range(10)
        })
        y_test = pd.Series(np.random.randint(0, 2, 1000))
        protected = pd.Series(np.random.choice(['A', 'B'], 1000))
        
        model = LogisticRegression(random_state=42)
        model.fit(X_test, y_test)
        
        import time
        
        start_time = time.time()
        
        # Register model
        model_id = self.registry.register_model(
            model=model,
            name="performance_test",
            version="1.0.0",
            algorithm="LogisticRegression",
            training_method="baseline",
            test_data=(X_test, y_test, protected)
        )
        
        end_time = time.time()
        registration_time = end_time - start_time
        
        # Should complete within reasonable time (< 5 seconds)
        assert registration_time < 5.0
        
        # Test loading performance
        start_time = time.time()
        
        loaded_model, metadata = self.registry.load_model("performance_test", "1.0.0")
        
        end_time = time.time()
        loading_time = end_time - start_time
        
        # Loading should be fast (< 1 second)
        assert loading_time < 1.0


# Fixtures
@pytest.fixture
def sample_registry():
    """Create a temporary model registry for testing."""
    temp_dir = tempfile.mkdtemp()
    registry = ModelRegistry(temp_dir)
    yield registry
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_model():
    """Create a sample trained model."""
    X = pd.DataFrame({
        'age': [25, 35, 45],
        'income': [40000, 50000, 60000]
    })
    y = pd.Series([0, 1, 1])
    
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def sample_test_data():
    """Create sample test data."""
    X_test = pd.DataFrame({
        'age': [25, 35, 45],
        'income': [40000, 50000, 60000],
        'credit_score': [650, 700, 750]
    })
    y_test = pd.Series([0, 1, 1])
    protected = pd.Series(['A', 'B', 'A'])
    
    return X_test, y_test, protected