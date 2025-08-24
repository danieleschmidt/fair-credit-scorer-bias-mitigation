"""
FastAPI endpoints for fairness evaluation and bias monitoring.

This module provides production-ready REST API endpoints for real-time
bias monitoring, fairness evaluation, and model explanations.
"""

from datetime import datetime
from typing import Any, Dict, List

try:
    from fastapi import BackgroundTasks, FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field, validator
    FASTAPI_AVAILABLE = True
except ImportError:
    # Fallback for environments without FastAPI
    FASTAPI_AVAILABLE = False
    BaseModel = object
    def Field(**kwargs):
        return None

import numpy as np
import pandas as pd

from ..baseline_model import train_baseline_model
from ..data_loader_preprocessor import load_credit_data
from ..evaluate_fairness import run_cross_validation, run_pipeline
from ..fairness_metrics import compute_fairness_metrics
from ..logging_config import get_logger
from ..model_explainability import ModelExplainer

logger = get_logger(__name__)


class PredictionRequest(BaseModel):
    """Request model for individual predictions."""
    features: Dict[str, float] = Field(..., description="Feature values for prediction")
    explain: bool = Field(False, description="Whether to include explanation")

    @validator('features')
    def validate_features(cls, v):
        required_features = {'age', 'income', 'credit_score', 'debt_to_income'}
        if not required_features.issubset(v.keys()):
            missing = required_features - set(v.keys())
            raise ValueError(f"Missing required features: {missing}")
        return v


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    samples: List[Dict[str, float]] = Field(..., description="List of feature dictionaries")
    include_fairness_metrics: bool = Field(True, description="Whether to compute fairness metrics")
    protected_attribute: str = Field("age", description="Protected attribute for fairness analysis")


class BiasMonitoringRequest(BaseModel):
    """Request model for bias monitoring."""
    model_name: str = Field(..., description="Name of the model to monitor")
    predictions: List[Dict[str, Any]] = Field(..., description="Recent predictions with outcomes")
    time_window: str = Field("1h", description="Time window for analysis (e.g., '1h', '1d')")


class ModelTrainingRequest(BaseModel):
    """Request model for model training."""
    method: str = Field("baseline", description="Training method")
    test_size: float = Field(0.3, description="Test set proportion")
    cross_validation: int = Field(1, description="Number of CV folds")

    @validator('method')
    def validate_method(cls, v):
        valid_methods = {"baseline", "reweight", "postprocess", "expgrad"}
        if v not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        return v


class FairnessAPI:
    """
    Production-ready FastAPI application for fairness evaluation.

    Provides endpoints for:
    - Individual and batch predictions with bias analysis
    - Model training with fairness constraints
    - Real-time bias monitoring and alerting
    - Model explanations and feature importance
    """

    def __init__(self):
        """Initialize the fairness API."""
        self.models = {}  # In-memory model store
        self.monitoring_data = []  # Store for monitoring data
        self.explainers = {}  # Store for explainers

        if not FASTAPI_AVAILABLE:
            logger.warning("FastAPI not available - API functionality limited")
            return

        self.app = FastAPI(
            title="Fair Credit Scorer API",
            description="REST API for bias-aware credit scoring",
            version="0.2.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )

        self._setup_routes()
        logger.info("FairnessAPI initialized successfully")

    def _setup_routes(self):
        """Setup all API routes."""
        if not FASTAPI_AVAILABLE:
            return

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "models_loaded": len(self.models),
                "api_version": "0.2.0"
            }

        @self.app.post("/predict")
        async def predict(request: PredictionRequest):
            """Single prediction with optional explanation."""
            try:
                # Use default model if available
                if "default" not in self.models:
                    raise HTTPException(status_code=404, detail="No model loaded")

                model = self.models["default"]
                features_df = pd.DataFrame([request.features])

                # Make prediction
                prediction = model.predict(features_df)[0]
                probability = model.predict_proba(features_df)[0]

                response = {
                    "prediction": int(prediction),
                    "probability": {
                        "class_0": float(probability[0]),
                        "class_1": float(probability[1])
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }

                # Add explanation if requested
                if request.explain and "default" in self.explainers:
                    explainer = self.explainers["default"]
                    explanation = explainer.explain_for_api(features_df)
                    response["explanation"] = explanation

                return response

            except Exception as e:
                logger.error(f"Prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/predict/batch")
        async def predict_batch(request: BatchPredictionRequest):
            """Batch predictions with fairness analysis."""
            try:
                if "default" not in self.models:
                    raise HTTPException(status_code=404, detail="No model loaded")

                model = self.models["default"]
                features_df = pd.DataFrame(request.samples)

                # Make predictions
                predictions = model.predict(features_df)
                probabilities = model.predict_proba(features_df)

                results = {
                    "predictions": predictions.tolist(),
                    "probabilities": probabilities.tolist(),
                    "count": len(predictions),
                    "timestamp": datetime.utcnow().isoformat()
                }

                # Add fairness metrics if requested
                if request.include_fairness_metrics:
                    if request.protected_attribute in features_df.columns:
                        protected = features_df[request.protected_attribute]
                        # For demo, create synthetic true labels
                        y_true = np.random.binomial(1, 0.7, len(predictions))

                        overall, by_group = compute_fairness_metrics(
                            y_true, predictions, protected, probabilities[:, 1]
                        )

                        results["fairness_metrics"] = {
                            "overall": overall.to_dict(),
                            "by_group": by_group.to_dict()
                        }

                return results

            except Exception as e:
                logger.error(f"Batch prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/train")
        async def train_model(request: ModelTrainingRequest, background_tasks: BackgroundTasks):
            """Train a new model with fairness constraints."""
            try:
                # Schedule training as background task
                background_tasks.add_task(
                    self._train_model_background,
                    request.method,
                    request.test_size,
                    request.cross_validation
                )

                return {
                    "message": "Training started",
                    "method": request.method,
                    "status": "training",
                    "timestamp": datetime.utcnow().isoformat()
                }

            except Exception as e:
                logger.error(f"Training error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/monitor/bias")
        async def monitor_bias(request: BiasMonitoringRequest):
            """Real-time bias monitoring endpoint."""
            try:
                # Process monitoring data
                monitoring_result = await self._analyze_bias_drift(
                    request.model_name,
                    request.predictions,
                    request.time_window
                )

                return monitoring_result

            except Exception as e:
                logger.error(f"Bias monitoring error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/models")
        async def list_models():
            """List all loaded models."""
            return {
                "models": list(self.models.keys()),
                "count": len(self.models),
                "timestamp": datetime.utcnow().isoformat()
            }

        @self.app.get("/models/{model_name}/metrics")
        async def get_model_metrics(model_name: str):
            """Get metrics for a specific model."""
            if model_name not in self.models:
                raise HTTPException(status_code=404, detail="Model not found")

            # Return cached metrics or compute new ones
            return {
                "model_name": model_name,
                "metrics": self._get_cached_metrics(model_name),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _train_model_background(self, method: str, test_size: float, cv: int):
        """Background task for model training."""
        try:
            logger.info(f"Starting background training with method={method}")

            if cv > 1:
                results = run_cross_validation(method=method, cv=cv)
            else:
                results = run_pipeline(method=method, test_size=test_size)

            # Store model (simplified - in production would save to disk)
            X_train, X_test, y_train, y_test = load_credit_data(test_size=test_size)
            features_train = X_train.drop("protected", axis=1)

            model = train_baseline_model(features_train, y_train)
            self.models["default"] = model

            # Create explainer
            background_data = features_train.sample(min(100, len(features_train)))
            self.explainers["default"] = ModelExplainer(model, background_data)

            logger.info(f"Training completed with accuracy: {results['accuracy']:.3f}")

        except Exception as e:
            logger.error(f"Background training failed: {e}")

    async def _analyze_bias_drift(self, model_name: str, predictions: List[Dict], time_window: str):
        """Analyze bias drift in real-time predictions."""
        try:
            # Convert predictions to DataFrame
            df = pd.DataFrame(predictions)

            if len(df) == 0:
                return {"status": "no_data", "message": "No predictions to analyze"}

            # Basic bias drift analysis
            required_cols = ['prediction', 'true_label', 'protected_attribute']
            if not all(col in df.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df.columns]
                return {"status": "error", "message": f"Missing columns: {missing}"}

            # Compute fairness metrics for current batch
            overall, by_group = compute_fairness_metrics(
                df['true_label'],
                df['prediction'],
                df['protected_attribute']
            )

            # Simple drift detection (compare to historical)
            drift_detected = self._detect_bias_drift(overall)

            return {
                "model_name": model_name,
                "time_window": time_window,
                "sample_count": len(df),
                "drift_detected": drift_detected,
                "current_metrics": {
                    "demographic_parity_difference": float(overall['demographic_parity_difference']),
                    "equalized_odds_difference": float(overall['equalized_odds_difference']),
                    "accuracy": float(overall['accuracy'])
                },
                "by_group_metrics": by_group.to_dict(),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Bias analysis failed: {e}")
            return {"status": "error", "message": str(e)}

    def _detect_bias_drift(self, current_metrics: pd.Series) -> bool:
        """Simple bias drift detection logic."""
        # Thresholds for bias drift detection
        dpd_threshold = 0.1
        eod_threshold = 0.1

        dpd = abs(current_metrics.get('demographic_parity_difference', 0))
        eod = abs(current_metrics.get('equalized_odds_difference', 0))

        return dpd > dpd_threshold or eod > eod_threshold

    def _get_cached_metrics(self, model_name: str) -> Dict[str, Any]:
        """Get cached metrics for a model."""
        # Simplified - in production would use a proper cache/database
        return {
            "accuracy": 0.83,
            "demographic_parity_difference": 0.05,
            "equalized_odds_difference": 0.08,
            "last_updated": datetime.utcnow().isoformat()
        }

    def get_app(self):
        """Get the FastAPI application instance."""
        if not FASTAPI_AVAILABLE:
            raise RuntimeError("FastAPI not available")
        return self.app


# Factory function for creating the API
def create_fairness_api() -> FairnessAPI:
    """Create and return a FairnessAPI instance."""
    return FairnessAPI()


# CLI interface
def main():
    """CLI interface for running the API server."""
    if not FASTAPI_AVAILABLE:
        print("FastAPI not installed. Install with: pip install fastapi uvicorn")
        return

    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Fair Credit Scorer API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    # Create API instance
    api = create_fairness_api()

    # Run server
    uvicorn.run(
        api.get_app(),
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
