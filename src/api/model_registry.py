"""
Model Registry API for version management and deployment.

This module provides a production-ready model registry with versioning,
metadata tracking, and deployment management capabilities.
"""

import hashlib
import json
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator

from ..logging_config import get_logger
from ..fairness_metrics import compute_fairness_metrics

logger = get_logger(__name__)


class ModelMetadata:
    """Model metadata container."""
    
    def __init__(
        self,
        name: str,
        version: str,
        algorithm: str,
        training_method: str,
        metrics: Dict[str, float],
        fairness_metrics: Dict[str, float],
        feature_names: List[str],
        created_at: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        self.name = name
        self.version = version
        self.algorithm = algorithm
        self.training_method = training_method
        self.metrics = metrics
        self.fairness_metrics = fairness_metrics
        self.feature_names = feature_names
        self.created_at = created_at or datetime.utcnow()
        self.tags = tags or {}
        self.model_hash = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "algorithm": self.algorithm,
            "training_method": self.training_method,
            "metrics": self.metrics,
            "fairness_metrics": self.fairness_metrics,
            "feature_names": self.feature_names,
            "created_at": self.created_at.isoformat(),
            "tags": self.tags,
            "model_hash": self.model_hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create metadata from dictionary."""
        metadata = cls(
            name=data["name"],
            version=data["version"],
            algorithm=data["algorithm"],
            training_method=data["training_method"],
            metrics=data["metrics"],
            fairness_metrics=data["fairness_metrics"],
            feature_names=data["feature_names"],
            created_at=datetime.fromisoformat(data["created_at"]),
            tags=data.get("tags", {})
        )
        metadata.model_hash = data.get("model_hash")
        return metadata


class ModelRegistry:
    """
    Production model registry with versioning and metadata tracking.
    
    Features:
    - Model versioning with semantic versioning
    - Metadata tracking (metrics, fairness, features)
    - Model comparison and promotion
    - Automated testing and validation
    - Deployment management
    """
    
    def __init__(self, registry_path: Union[str, Path] = "models"):
        """
        Initialize model registry.
        
        Args:
            registry_path: Base path for storing models and metadata
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.registry_path / "models").mkdir(exist_ok=True)
        (self.registry_path / "metadata").mkdir(exist_ok=True)
        (self.registry_path / "experiments").mkdir(exist_ok=True)
        
        logger.info(f"ModelRegistry initialized at {self.registry_path}")
    
    def register_model(
        self,
        model: BaseEstimator,
        name: str,
        version: str,
        algorithm: str,
        training_method: str,
        test_data: Optional[tuple] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Register a new model version.
        
        Args:
            model: Trained model to register
            name: Model name
            version: Model version (semantic versioning)
            algorithm: Algorithm used (e.g., "LogisticRegression")
            training_method: Training method (e.g., "baseline", "reweight")
            test_data: Tuple of (X_test, y_test, protected) for evaluation
            tags: Additional metadata tags
            
        Returns:
            Model registration ID
        """
        try:
            # Generate model hash for integrity checking
            model_hash = self._compute_model_hash(model)
            
            # Evaluate model if test data provided
            metrics = {}
            fairness_metrics = {}
            feature_names = []
            
            if test_data is not None:
                X_test, y_test, protected = test_data
                feature_names = list(X_test.columns) if hasattr(X_test, 'columns') else []
                
                # Compute performance metrics
                y_pred = model.predict(X_test)
                y_scores = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                metrics = {
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "precision": float(precision_score(y_test, y_pred, average='binary')),
                    "recall": float(recall_score(y_test, y_pred, average='binary')),
                    "f1": float(f1_score(y_test, y_pred, average='binary'))
                }
                
                # Compute fairness metrics
                overall, _ = compute_fairness_metrics(y_test, y_pred, protected, y_scores)
                fairness_metrics = {
                    "demographic_parity_difference": float(overall["demographic_parity_difference"]),
                    "equalized_odds_difference": float(overall["equalized_odds_difference"]),
                    "accuracy_difference": float(overall["accuracy_difference"])
                }
            
            # Create metadata
            metadata = ModelMetadata(
                name=name,
                version=version,
                algorithm=algorithm,
                training_method=training_method,
                metrics=metrics,
                fairness_metrics=fairness_metrics,
                feature_names=feature_names,
                tags=tags
            )
            metadata.model_hash = model_hash
            
            # Save model and metadata
            model_id = f"{name}_v{version}"
            self._save_model(model, model_id)
            self._save_metadata(metadata, model_id)
            
            logger.info(f"Model registered: {model_id}")
            return model_id
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise
    
    def load_model(self, name: str, version: Optional[str] = None) -> tuple[BaseEstimator, ModelMetadata]:
        """
        Load a model and its metadata.
        
        Args:
            name: Model name
            version: Model version (latest if None)
            
        Returns:
            Tuple of (model, metadata)
        """
        try:
            if version is None:
                version = self.get_latest_version(name)
            
            model_id = f"{name}_v{version}"
            
            # Load model
            model_path = self.registry_path / "models" / f"{model_id}.pkl"
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_id}")
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Load metadata
            metadata = self._load_metadata(model_id)
            
            # Verify model integrity
            current_hash = self._compute_model_hash(model)
            if metadata.model_hash != current_hash:
                logger.warning(f"Model hash mismatch for {model_id}")
            
            logger.info(f"Model loaded: {model_id}")
            return model, metadata
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def list_models(self, name_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all registered models.
        
        Args:
            name_filter: Optional filter by model name
            
        Returns:
            List of model metadata dictionaries
        """
        try:
            models = []
            metadata_dir = self.registry_path / "metadata"
            
            for metadata_file in metadata_dir.glob("*.json"):
                model_id = metadata_file.stem
                metadata = self._load_metadata(model_id)
                
                if name_filter is None or name_filter in metadata.name:
                    models.append(metadata.to_dict())
            
            # Sort by creation date (newest first)
            models.sort(key=lambda x: x["created_at"], reverse=True)
            
            return models
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def compare_models(self, model_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple models across metrics.
        
        Args:
            model_ids: List of model IDs to compare
            
        Returns:
            Comparison results
        """
        try:
            comparison = {
                "models": [],
                "metrics_comparison": {},
                "fairness_comparison": {},
                "recommendation": None
            }
            
            for model_id in model_ids:
                metadata = self._load_metadata(model_id)
                comparison["models"].append({
                    "id": model_id,
                    "name": metadata.name,
                    "version": metadata.version,
                    "training_method": metadata.training_method,
                    "created_at": metadata.created_at.isoformat()
                })
            
            # Aggregate metrics for comparison
            metrics_summary = {}
            fairness_summary = {}
            
            for model_id in model_ids:
                metadata = self._load_metadata(model_id)
                
                for metric, value in metadata.metrics.items():
                    if metric not in metrics_summary:
                        metrics_summary[metric] = {}
                    metrics_summary[metric][model_id] = value
                
                for metric, value in metadata.fairness_metrics.items():
                    if metric not in fairness_summary:
                        fairness_summary[metric] = {}
                    fairness_summary[metric][model_id] = value
            
            comparison["metrics_comparison"] = metrics_summary
            comparison["fairness_comparison"] = fairness_summary
            
            # Simple recommendation logic
            comparison["recommendation"] = self._recommend_best_model(model_ids)
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare models: {e}")
            return {}
    
    def promote_model(self, model_id: str, stage: str = "production") -> bool:
        """
        Promote a model to a specific stage.
        
        Args:
            model_id: Model ID to promote
            stage: Target stage (e.g., "staging", "production")
            
        Returns:
            Success status
        """
        try:
            metadata = self._load_metadata(model_id)
            metadata.tags[f"stage"] = stage
            metadata.tags[f"promoted_at"] = datetime.utcnow().isoformat()
            
            self._save_metadata(metadata, model_id)
            
            # Create stage-specific symlink or copy
            stage_path = self.registry_path / f"{stage}_model"
            model_path = self.registry_path / "models" / f"{model_id}.pkl"
            
            if stage_path.exists():
                stage_path.unlink()
            
            shutil.copy2(model_path, stage_path)
            
            logger.info(f"Model {model_id} promoted to {stage}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            return False
    
    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model and its metadata.
        
        Args:
            model_id: Model ID to delete
            
        Returns:
            Success status
        """
        try:
            # Delete model file
            model_path = self.registry_path / "models" / f"{model_id}.pkl"
            if model_path.exists():
                model_path.unlink()
            
            # Delete metadata file
            metadata_path = self.registry_path / "metadata" / f"{model_id}.json"
            if metadata_path.exists():
                metadata_path.unlink()
            
            logger.info(f"Model deleted: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model: {e}")
            return False
    
    def get_latest_version(self, name: str) -> str:
        """Get the latest version of a model."""
        models = self.list_models(name_filter=name)
        if not models:
            raise ValueError(f"No models found with name: {name}")
        
        # Extract versions and find the latest
        versions = [m["version"] for m in models if m["name"] == name]
        if not versions:
            raise ValueError(f"No models found with name: {name}")
        
        # Simple version sorting (assumes semantic versioning)
        versions.sort(key=lambda v: tuple(map(int, v.split('.'))), reverse=True)
        return versions[0]
    
    def _save_model(self, model: BaseEstimator, model_id: str):
        """Save model to disk."""
        model_path = self.registry_path / "models" / f"{model_id}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    
    def _save_metadata(self, metadata: ModelMetadata, model_id: str):
        """Save metadata to disk."""
        metadata_path = self.registry_path / "metadata" / f"{model_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
    
    def _load_metadata(self, model_id: str) -> ModelMetadata:
        """Load metadata from disk."""
        metadata_path = self.registry_path / "metadata" / f"{model_id}.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {model_id}")
        
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        
        return ModelMetadata.from_dict(data)
    
    def _compute_model_hash(self, model: BaseEstimator) -> str:
        """Compute hash of model for integrity checking."""
        model_bytes = pickle.dumps(model)
        return hashlib.sha256(model_bytes).hexdigest()
    
    def _recommend_best_model(self, model_ids: List[str]) -> Optional[str]:
        """Recommend best model based on fairness-accuracy tradeoff."""
        if not model_ids:
            return None
        
        best_model = None
        best_score = -1
        
        for model_id in model_ids:
            try:
                metadata = self._load_metadata(model_id)
                
                # Simple scoring: balance accuracy and fairness
                accuracy = metadata.metrics.get("accuracy", 0)
                dpd = abs(metadata.fairness_metrics.get("demographic_parity_difference", 1))
                eod = abs(metadata.fairness_metrics.get("equalized_odds_difference", 1))
                
                # Score: high accuracy, low bias
                fairness_penalty = (dpd + eod) / 2
                score = accuracy - fairness_penalty
                
                if score > best_score:
                    best_score = score
                    best_model = model_id
                    
            except Exception as e:
                logger.warning(f"Could not evaluate model {model_id}: {e}")
        
        return best_model


# CLI interface
def main():
    """CLI interface for model registry operations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Registry CLI")
    parser.add_argument("command", choices=["list", "compare", "promote", "delete"])
    parser.add_argument("--name", help="Model name filter")
    parser.add_argument("--model-ids", nargs="+", help="Model IDs for comparison")
    parser.add_argument("--model-id", help="Model ID for promotion/deletion")
    parser.add_argument("--stage", default="production", help="Promotion stage")
    parser.add_argument("--registry-path", default="models", help="Registry path")
    
    args = parser.parse_args()
    
    registry = ModelRegistry(args.registry_path)
    
    if args.command == "list":
        models = registry.list_models(args.name)
        print(f"Found {len(models)} models:")
        for model in models:
            print(f"  {model['name']} v{model['version']} ({model['training_method']})")
    
    elif args.command == "compare":
        if not args.model_ids:
            print("Error: --model-ids required for comparison")
            return
        
        comparison = registry.compare_models(args.model_ids)
        print("Model Comparison:")
        print(json.dumps(comparison, indent=2, default=str))
    
    elif args.command == "promote":
        if not args.model_id:
            print("Error: --model-id required for promotion")
            return
        
        success = registry.promote_model(args.model_id, args.stage)
        if success:
            print(f"Model {args.model_id} promoted to {args.stage}")
        else:
            print("Promotion failed")
    
    elif args.command == "delete":
        if not args.model_id:
            print("Error: --model-id required for deletion")
            return
        
        success = registry.delete_model(args.model_id)
        if success:
            print(f"Model {args.model_id} deleted")
        else:
            print("Deletion failed")


if __name__ == "__main__":
    main()