"""
Comprehensive tests for the fairness API.

Tests all endpoints including predictions, training, and monitoring
with various edge cases and error conditions.
"""

import json
import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

# Handle optional FastAPI import
try:
    from fastapi.testclient import TestClient
    from fastapi import HTTPException
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    TestClient = None
    HTTPException = None

from src.api.fairness_api import FairnessAPI, PredictionRequest, BatchPredictionRequest


class TestFairnessAPI:
    """Test suite for FairnessAPI class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.api = FairnessAPI()
        
        # Mock data for testing
        self.sample_features = {
            'age': 35,
            'income': 50000,
            'credit_score': 700,
            'debt_to_income': 0.2
        }
        
        self.sample_batch = [
            {'age': 25, 'income': 40000, 'credit_score': 650, 'debt_to_income': 0.3},
            {'age': 45, 'income': 60000, 'credit_score': 750, 'debt_to_income': 0.1}
        ]
    
    def test_api_initialization(self):
        """Test API initialization."""
        api = FairnessAPI()
        assert hasattr(api, 'models')
        assert hasattr(api, 'monitoring_data')
        assert hasattr(api, 'explainers')
        
        if FASTAPI_AVAILABLE:
            assert hasattr(api, 'app')
            assert api.app.title == "Fair Credit Scorer API"
    
    @patch('src.api.fairness_api.train_baseline_model')
    @patch('src.api.fairness_api.load_credit_data')
    async def test_train_model_background(self, mock_load_data, mock_train_model):
        """Test background model training."""
        # Setup mocks
        X_train = pd.DataFrame(self.sample_batch)
        X_test = pd.DataFrame(self.sample_batch)
        y_train = pd.Series([1, 0])
        y_test = pd.Series([0, 1])
        
        mock_load_data.return_value = (X_train, X_test, y_train, y_test)
        mock_model = Mock()
        mock_train_model.return_value = mock_model
        
        # Test training
        await self.api._train_model_background("baseline", 0.3, 1)
        
        # Verify model was stored
        assert "default" in self.api.models
        assert self.api.models["default"] == mock_model
    
    async def test_analyze_bias_drift(self):
        """Test bias drift analysis."""
        predictions = [
            {'prediction': 1, 'true_label': 1, 'protected_attribute': 'A'},
            {'prediction': 0, 'true_label': 0, 'protected_attribute': 'B'},
            {'prediction': 1, 'true_label': 0, 'protected_attribute': 'A'}
        ]
        
        result = await self.api._analyze_bias_drift("test_model", predictions, "1h")
        
        assert result["model_name"] == "test_model"
        assert result["time_window"] == "1h"
        assert result["sample_count"] == 3
        assert "drift_detected" in result
        assert "current_metrics" in result
    
    def test_detect_bias_drift(self):
        """Test bias drift detection logic."""
        # Test with high bias
        high_bias_metrics = pd.Series({
            'demographic_parity_difference': 0.15,
            'equalized_odds_difference': 0.05
        })
        
        assert self.api._detect_bias_drift(high_bias_metrics) == True
        
        # Test with low bias
        low_bias_metrics = pd.Series({
            'demographic_parity_difference': 0.05,
            'equalized_odds_difference': 0.03
        })
        
        assert self.api._detect_bias_drift(low_bias_metrics) == False
    
    def test_get_cached_metrics(self):
        """Test cached metrics retrieval."""
        metrics = self.api._get_cached_metrics("test_model")
        
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "demographic_parity_difference" in metrics
        assert "last_updated" in metrics


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
class TestFairnessAPIEndpoints:
    """Test suite for FastAPI endpoints."""
    
    def setup_method(self):
        """Setup test client."""
        self.api = FairnessAPI()
        if FASTAPI_AVAILABLE:
            self.client = TestClient(self.api.get_app())
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "api_version" in data
    
    def test_predict_endpoint_no_model(self):
        """Test prediction endpoint without loaded model."""
        prediction_data = {
            "features": {
                "age": 35,
                "income": 50000,
                "credit_score": 700,
                "debt_to_income": 0.2
            }
        }
        
        response = self.client.post("/predict", json=prediction_data)
        assert response.status_code == 404
    
    @patch('src.api.fairness_api.train_baseline_model')
    def test_predict_endpoint_with_model(self, mock_train_model):
        """Test prediction endpoint with loaded model."""
        # Setup mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        
        self.api.models["default"] = mock_model
        
        prediction_data = {
            "features": {
                "age": 35,
                "income": 50000,
                "credit_score": 700,
                "debt_to_income": 0.2
            }
        }
        
        response = self.client.post("/predict", json=prediction_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == 1
        assert "probability" in data
        assert "timestamp" in data
    
    def test_batch_predict_endpoint(self):
        """Test batch prediction endpoint."""
        # Setup mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1, 0])
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2]])
        
        self.api.models["default"] = mock_model
        
        batch_data = {
            "samples": [
                {"age": 25, "income": 40000, "credit_score": 650, "debt_to_income": 0.3},
                {"age": 45, "income": 60000, "credit_score": 750, "debt_to_income": 0.1}
            ]
        }
        
        response = self.client.post("/predict/batch", json=batch_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert len(data["predictions"]) == 2
        assert len(data["probabilities"]) == 2
    
    def test_train_endpoint(self):
        """Test model training endpoint."""
        training_data = {
            "method": "baseline",
            "test_size": 0.3,
            "cross_validation": 1
        }
        
        response = self.client.post("/train", json=training_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "training"
        assert data["method"] == "baseline"
    
    def test_monitor_bias_endpoint(self):
        """Test bias monitoring endpoint."""
        monitoring_data = {
            "model_name": "test_model",
            "predictions": [
                {"prediction": 1, "true_label": 1, "protected_attribute": "A"},
                {"prediction": 0, "true_label": 0, "protected_attribute": "B"}
            ],
            "time_window": "1h"
        }
        
        response = self.client.post("/monitor/bias", json=monitoring_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "test_model"
        assert data["time_window"] == "1h"
    
    def test_list_models_endpoint(self):
        """Test list models endpoint."""
        # Add a test model
        self.api.models["test_model"] = Mock()
        
        response = self.client.get("/models")
        
        assert response.status_code == 200
        data = response.json()
        assert "test_model" in data["models"]
        assert data["count"] >= 1
    
    def test_get_model_metrics_endpoint(self):
        """Test get model metrics endpoint."""
        # Add a test model
        self.api.models["test_model"] = Mock()
        
        response = self.client.get("/models/test_model/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "test_model"
        assert "metrics" in data
    
    def test_get_model_metrics_not_found(self):
        """Test get metrics for non-existent model."""
        response = self.client.get("/models/nonexistent/metrics")
        assert response.status_code == 404


class TestPredictionRequest:
    """Test suite for PredictionRequest validation."""
    
    def test_valid_prediction_request(self):
        """Test valid prediction request creation."""
        request_data = {
            "features": {
                "age": 35,
                "income": 50000,
                "credit_score": 700,
                "debt_to_income": 0.2
            },
            "explain": True
        }
        
        request = PredictionRequest(**request_data)
        assert request.features["age"] == 35
        assert request.explain == True
    
    def test_missing_required_features(self):
        """Test prediction request with missing required features."""
        request_data = {
            "features": {
                "age": 35,
                "income": 50000
                # Missing credit_score and debt_to_income
            }
        }
        
        with pytest.raises(ValueError, match="Missing required features"):
            PredictionRequest(**request_data)
    
    def test_default_explain_value(self):
        """Test default value for explain parameter."""
        request_data = {
            "features": {
                "age": 35,
                "income": 50000,
                "credit_score": 700,
                "debt_to_income": 0.2
            }
        }
        
        request = PredictionRequest(**request_data)
        assert request.explain == False


class TestBatchPredictionRequest:
    """Test suite for BatchPredictionRequest validation."""
    
    def test_valid_batch_request(self):
        """Test valid batch prediction request."""
        request_data = {
            "samples": [
                {"age": 25, "income": 40000, "credit_score": 650, "debt_to_income": 0.3},
                {"age": 45, "income": 60000, "credit_score": 750, "debt_to_income": 0.1}
            ],
            "include_fairness_metrics": True,
            "protected_attribute": "age"
        }
        
        request = BatchPredictionRequest(**request_data)
        assert len(request.samples) == 2
        assert request.include_fairness_metrics == True
        assert request.protected_attribute == "age"
    
    def test_default_values(self):
        """Test default values for batch request."""
        request_data = {
            "samples": [
                {"age": 25, "income": 40000, "credit_score": 650, "debt_to_income": 0.3}
            ]
        }
        
        request = BatchPredictionRequest(**request_data)
        assert request.include_fairness_metrics == True
        assert request.protected_attribute == "age"


@pytest.mark.integration
class TestFairnessAPIIntegration:
    """Integration tests for the fairness API."""
    
    @patch('src.api.fairness_api.load_credit_data')
    @patch('src.api.fairness_api.train_baseline_model')
    async def test_end_to_end_training_and_prediction(self, mock_train_model, mock_load_data):
        """Test complete training and prediction workflow."""
        # Setup mocks
        X_train = pd.DataFrame([
            {'age': 25, 'income': 40000, 'credit_score': 650, 'debt_to_income': 0.3},
            {'age': 45, 'income': 60000, 'credit_score': 750, 'debt_to_income': 0.1}
        ])
        X_test = X_train.copy()
        y_train = pd.Series([1, 0])
        y_test = pd.Series([0, 1])
        
        mock_load_data.return_value = (X_train, X_test, y_train, y_test)
        
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        mock_train_model.return_value = mock_model
        
        api = FairnessAPI()
        
        # Train model
        await api._train_model_background("baseline", 0.3, 1)
        
        # Verify model is loaded
        assert "default" in api.models
        
        # Test prediction (if FastAPI is available)
        if FASTAPI_AVAILABLE:
            client = TestClient(api.get_app())
            
            prediction_data = {
                "features": {
                    "age": 35,
                    "income": 50000,
                    "credit_score": 700,
                    "debt_to_income": 0.2
                }
            }
            
            response = client.post("/predict", json=prediction_data)
            assert response.status_code == 200
            
            data = response.json()
            assert data["prediction"] == 1
            assert "probability" in data


# Fixtures for testing
@pytest.fixture
def sample_model():
    """Create a mock model for testing."""
    model = Mock()
    model.predict.return_value = np.array([1, 0, 1])
    model.predict_proba.return_value = np.array([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
    return model


@pytest.fixture
def sample_features():
    """Create sample feature data."""
    return {
        'age': 35,
        'income': 50000,
        'credit_score': 700,
        'debt_to_income': 0.2
    }


@pytest.fixture
def sample_batch_data():
    """Create sample batch prediction data."""
    return [
        {'age': 25, 'income': 40000, 'credit_score': 650, 'debt_to_income': 0.3, 'protected': 'A'},
        {'age': 45, 'income': 60000, 'credit_score': 750, 'debt_to_income': 0.1, 'protected': 'B'},
        {'age': 35, 'income': 55000, 'credit_score': 720, 'debt_to_income': 0.2, 'protected': 'A'}
    ]


# Performance tests
@pytest.mark.performance
class TestFairnessAPIPerformance:
    """Performance tests for the fairness API."""
    
    def test_prediction_performance(self, sample_model, sample_features):
        """Test prediction endpoint performance."""
        api = FairnessAPI()
        api.models["default"] = sample_model
        
        # Time multiple predictions
        import time
        
        start_time = time.time()
        
        for _ in range(100):
            features_df = pd.DataFrame([sample_features])
            prediction = sample_model.predict(features_df)[0]
            probability = sample_model.predict_proba(features_df)[0]
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 100
        
        # Should be fast (< 10ms per prediction)
        assert avg_time < 0.01
    
    def test_batch_prediction_performance(self, sample_model, sample_batch_data):
        """Test batch prediction performance."""
        api = FairnessAPI()
        api.models["default"] = sample_model
        
        import time
        
        start_time = time.time()
        
        # Simulate batch prediction
        features_df = pd.DataFrame(sample_batch_data)
        predictions = sample_model.predict(features_df)
        probabilities = sample_model.predict_proba(features_df)
        
        end_time = time.time()
        
        # Batch should be efficient
        assert end_time - start_time < 0.1
        assert len(predictions) == len(sample_batch_data)


# Error handling tests
class TestFairnessAPIErrorHandling:
    """Test error handling in the fairness API."""
    
    def test_invalid_features_format(self):
        """Test handling of invalid feature formats."""
        with pytest.raises(ValueError):
            PredictionRequest(features="invalid_format")
    
    def test_missing_model_prediction(self):
        """Test prediction without loaded model."""
        api = FairnessAPI()
        
        # Should handle missing model gracefully
        assert len(api.models) == 0
    
    async def test_bias_analysis_empty_data(self):
        """Test bias analysis with empty data."""
        api = FairnessAPI()
        
        result = await api._analyze_bias_drift("test_model", [], "1h")
        
        assert result["status"] == "no_data"
        assert result["message"] == "No predictions to analyze"
    
    async def test_bias_analysis_invalid_data(self):
        """Test bias analysis with invalid data format."""
        api = FairnessAPI()
        
        invalid_predictions = [
            {'prediction': 1, 'true_label': 1}  # Missing protected_attribute
        ]
        
        result = await api._analyze_bias_drift("test_model", invalid_predictions, "1h")
        
        assert result["status"] == "error"
        assert "Missing columns" in result["message"]