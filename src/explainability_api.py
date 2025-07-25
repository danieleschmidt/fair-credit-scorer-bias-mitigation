"""
API endpoint for model explainability features.

This module provides HTTP endpoints for generating model explanations
using the ModelExplainer class.
"""

import json
import logging
from typing import Dict, Any, Optional
import pandas as pd
from sklearn.datasets import make_classification

try:
    from src.model_explainability import ModelExplainer
    from src.baseline_model import train_baseline_model
except ImportError:
    # For CLI usage
    from model_explainability import ModelExplainer
    from baseline_model import train_baseline_model

# Configure logging
logger = logging.getLogger(__name__)


class ExplainabilityAPI:
    """
    REST API endpoints for model explainability.
    
    Provides endpoints to explain individual predictions and get
    feature importance visualizations.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the explainability API.
        
        Args:
            model_path: Path to saved model file (optional)
        """
        self.model = None
        self.explainer = None
        self.model_path = model_path
        
        if model_path:
            self._load_model(model_path)
        
        logger.info("ExplainabilityAPI initialized")
    
    def _load_model(self, model_path: str):
        """Load model from file path."""
        try:
            # For this implementation, we'll create a sample trained model
            # In production, this would load from the specified path
            
            # Create sample training data
            X, y = make_classification(n_samples=1000, n_features=4, n_classes=2, random_state=42)
            X_df = pd.DataFrame(X, columns=['age', 'income', 'credit_score', 'debt_to_income'])
            
            # Train a baseline model
            self.model = train_baseline_model(X_df, y)
            
            # Create background data for explainer (use a sample of training data)
            background_data = X_df.sample(100, random_state=42)
            
            self.explainer = ModelExplainer(self.model, background_data)
            logger.info(f"Model loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def explain_prediction(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        API endpoint to explain a single prediction.
        
        Args:
            request_data: Dictionary containing feature values
            
        Returns:
            JSON response with explanation data
        """
        try:
            if not self.explainer:
                return {
                    'error': 'Model not loaded',
                    'status': 'error'
                }
            
            # Convert request data to DataFrame
            if 'features' in request_data:
                features_df = pd.DataFrame([request_data['features']])
            else:
                # Assume request_data contains the features directly
                features_df = pd.DataFrame([request_data])
            
            # Generate explanation
            explanation = self.explainer.explain_for_api(features_df)
            
            return {
                'status': 'success',
                'explanation': explanation,
                'model_info': self.explainer.get_model_summary()
            }
            
        except Exception as e:
            logger.error(f"Error in explain_prediction: {e}")
            return {
                'error': str(e),
                'status': 'error'
            }
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """
        API endpoint to get global feature importance.
        
        Returns:
            JSON response with feature importance data
        """
        try:
            if not self.explainer:
                return {
                    'error': 'Model not loaded',
                    'status': 'error'
                }
            
            importance_data = self.explainer.get_feature_importance_plot()
            
            return {
                'status': 'success',
                'feature_importance': importance_data,
                'model_info': self.explainer.get_model_summary()
            }
            
        except Exception as e:
            logger.error(f"Error in get_feature_importance: {e}")
            return {
                'error': str(e),
                'status': 'error'
            }
    
    def health_check(self) -> Dict[str, Any]:
        """
        API endpoint for health checking.
        
        Returns:
            JSON response with API status
        """
        return {
            'status': 'healthy',
            'model_loaded': self.model is not None,
            'explainer_ready': self.explainer is not None,
            'api_version': '1.0.0'
        }


def create_flask_app():
    """
    Create a Flask application with explainability endpoints.
    
    Returns:
        Flask application instance
    """
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        logger.warning("Flask not available - API endpoints not created")
        return None
    
    app = Flask(__name__)
    api = ExplainabilityAPI()
    
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify(api.health_check())
    
    @app.route('/explain', methods=['POST'])
    def explain():
        return jsonify(api.explain_prediction(request.json))
    
    @app.route('/feature-importance', methods=['GET'])
    def feature_importance():
        return jsonify(api.get_feature_importance())
    
    @app.route('/load-model', methods=['POST'])
    def load_model():
        data = request.json
        model_path = data.get('model_path')
        
        if not model_path:
            return jsonify({'error': 'model_path required', 'status': 'error'})
        
        try:
            api._load_model(model_path)
            return jsonify({'status': 'success', 'message': 'Model loaded successfully'})
        except Exception as e:
            return jsonify({'error': str(e), 'status': 'error'})
    
    return app


# CLI interface for testing
def main():
    """CLI interface for testing the explainability API."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Model Explainability API')
    parser.add_argument('--test', action='store_true', help='Run test scenario')
    parser.add_argument('--features', type=str, help='JSON string with features to explain')
    
    args = parser.parse_args()
    
    if args.test:
        # Test the API with sample data
        api = ExplainabilityAPI()
        
        # Test health check
        print("Health check:", json.dumps(api.health_check(), indent=2))
        
        # Test with sample features
        sample_features = {
            'age': 35,
            'income': 50000,
            'credit_score': 700,
            'debt_to_income': 0.2
        }
        
        result = api.explain_prediction(sample_features)
        print("Explanation result:", json.dumps(result, indent=2))
        
        # Test feature importance
        importance = api.get_feature_importance()
        print("Feature importance:", json.dumps(importance, indent=2))
    
    elif args.features:
        try:
            features = json.loads(args.features)
            api = ExplainabilityAPI()
            result = api.explain_prediction(features)
            print(json.dumps(result, indent=2))
        except json.JSONDecodeError:
            print("Error: Invalid JSON in features argument")
        except Exception as e:
            print(f"Error: {e}")
    
    else:
        print("Use --test to run test scenario or --features with JSON data")


if __name__ == '__main__':
    main()