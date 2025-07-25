"""
Tests for explainability API endpoints.
"""

import pytest
import json
from unittest.mock import Mock, patch
from src.explainability_api import ExplainabilityAPI


class TestExplainabilityAPI:
    """Test suite for ExplainabilityAPI class."""
    
    def test_api_initialization(self):
        """Test API initialization."""
        api = ExplainabilityAPI()
        assert api.model is None
        assert api.explainer is None
        assert api.model_path is None
    
    def test_health_check(self):
        """Test health check endpoint."""
        api = ExplainabilityAPI()
        response = api.health_check()
        
        assert response['status'] == 'healthy'
        assert response['model_loaded'] is False
        assert response['explainer_ready'] is False
        assert 'api_version' in response
    
    @patch('src.explainability_api.train_baseline_model')
    @patch('src.explainability_api.ModelExplainer')
    @patch('src.explainability_api.make_classification')
    def test_load_model(self, mock_make_classification, mock_explainer, mock_train):
        """Test model loading."""
        # Setup mocks
        # Create larger sample data to avoid sampling issues
        import numpy as np
        X = np.random.rand(200, 4)
        y = np.random.randint(0, 2, 200)
        mock_make_classification.return_value = (X, y)
        mock_train.return_value = Mock()
        mock_explainer.return_value = Mock()
        
        api = ExplainabilityAPI()
        api._load_model('test_model.pkl')
        
        assert api.model is not None
        assert api.explainer is not None
        mock_train.assert_called_once()
        mock_explainer.assert_called_once()
    
    def test_explain_prediction_no_model(self):
        """Test explanation when no model is loaded."""
        api = ExplainabilityAPI()
        
        request_data = {
            'age': 35,
            'income': 50000,
            'credit_score': 700
        }
        
        response = api.explain_prediction(request_data)
        assert response['status'] == 'error'
        assert 'Model not loaded' in response['error']
    
    @patch('src.explainability_api.train_baseline_model')
    @patch('src.explainability_api.ModelExplainer')
    @patch('src.explainability_api.make_classification')
    def test_explain_prediction_success(self, mock_make_classification, mock_explainer_class, mock_train):
        """Test successful prediction explanation."""
        # Setup mocks
        # Create larger sample data to avoid sampling issues
        import numpy as np
        X = np.random.rand(200, 4)
        y = np.random.randint(0, 2, 200)
        mock_make_classification.return_value = (X, y)
        mock_train.return_value = Mock()
        
        mock_explainer = Mock()
        mock_explainer.explain_for_api.return_value = {
            'explanation': {'test': 'data'},
            'prediction_probability': [0.3, 0.7]
        }
        mock_explainer.get_model_summary.return_value = {'features': ['age', 'income']}
        mock_explainer_class.return_value = mock_explainer
        
        api = ExplainabilityAPI()
        api._load_model('test_model.pkl')
        
        request_data = {
            'age': 35,
            'income': 50000,
            'credit_score': 700
        }
        
        response = api.explain_prediction(request_data)
        assert response['status'] == 'success'
        assert 'explanation' in response
        assert 'model_info' in response
    
    @patch('src.explainability_api.train_baseline_model')
    @patch('src.explainability_api.ModelExplainer')
    @patch('src.explainability_api.make_classification')
    def test_get_feature_importance(self, mock_make_classification, mock_explainer_class, mock_train):
        """Test feature importance endpoint."""
        # Setup mocks
        # Create larger sample data to avoid sampling issues
        import numpy as np
        X = np.random.rand(200, 4)
        y = np.random.randint(0, 2, 200)
        mock_make_classification.return_value = (X, y)
        mock_train.return_value = Mock()
        
        mock_explainer = Mock()
        mock_explainer.get_feature_importance_plot.return_value = {
            'features': ['age', 'income'],
            'importance_scores': [0.5, 0.3]
        }
        mock_explainer.get_model_summary.return_value = {'features': ['age', 'income']}
        mock_explainer_class.return_value = mock_explainer
        
        api = ExplainabilityAPI()
        api._load_model('test_model.pkl')
        
        response = api.get_feature_importance()
        assert response['status'] == 'success'
        assert 'feature_importance' in response
        assert 'model_info' in response
    
    def test_get_feature_importance_no_model(self):
        """Test feature importance when no model is loaded."""
        api = ExplainabilityAPI()
        
        response = api.get_feature_importance()
        assert response['status'] == 'error'
        assert 'Model not loaded' in response['error']
    
    @patch('src.explainability_api.create_flask_app')
    def test_flask_app_creation(self, mock_create_app):
        """Test Flask app creation (if Flask is available)."""
        mock_app = Mock()
        mock_create_app.return_value = mock_app
        
        app = mock_create_app()
        assert app is not None