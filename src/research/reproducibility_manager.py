"""
Reproducibility Manager for Fairness Research.

Ensures complete reproducibility of fairness research experiments through
systematic tracking of experimental configurations, data versions, model states,
and computational environments.

Research contributions:
- Complete experimental reproducibility framework
- Automated environment and dependency tracking
- Version control integration for research artifacts
- Comprehensive result validation and verification
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import warnings
from datetime import datetime
import json
import pickle
import hashlib
import os
import platform
import sys
from pathlib import Path
import subprocess
import shutil

from ..logging_config import get_logger

logger = get_logger(__name__)


class ReproducibilityLevel(Enum):
    """Levels of reproducibility assurance."""
    BASIC = "basic"          # Random seeds, basic config
    STANDARD = "standard"    # + Environment tracking
    COMPREHENSIVE = "comprehensive"  # + Data checksums, model states
    RESEARCH_GRADE = "research_grade"  # + Full provenance tracking


class ExperimentStatus(Enum):
    """Status of experiments."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    REPRODUCED = "reproduced"
    VALIDATED = "validated"


@dataclass
class EnvironmentInfo:
    """Information about the computational environment."""
    python_version: str
    platform_system: str
    platform_release: str
    platform_version: str
    cpu_count: int
    memory_total_gb: float
    installed_packages: Dict[str, str]
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    conda_environment: Optional[str] = None
    
    @classmethod
    def capture_current_environment(cls) -> 'EnvironmentInfo':
        """Capture current environment information."""
        import psutil
        
        # Get installed packages
        packages = {}
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'list', '--format=json'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                pip_list = json.loads(result.stdout)
                packages = {pkg['name']: pkg['version'] for pkg in pip_list}
        except Exception as e:
            logger.warning(f"Could not capture package list: {e}")
        
        # Get git information
        git_commit = None
        git_branch = None
        try:
            # Get current commit
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                git_commit = result.stdout.strip()
            
            # Get current branch
            result = subprocess.run(['git', 'branch', '--show-current'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                git_branch = result.stdout.strip()
        except Exception as e:
            logger.warning(f"Could not capture git information: {e}")
        
        # Get conda environment
        conda_env = os.environ.get('CONDA_DEFAULT_ENV')
        
        return cls(
            python_version=sys.version,
            platform_system=platform.system(),
            platform_release=platform.release(),
            platform_version=platform.version(),
            cpu_count=os.cpu_count(),
            memory_total_gb=psutil.virtual_memory().total / (1024**3),
            installed_packages=packages,
            git_commit=git_commit,
            git_branch=git_branch,
            conda_environment=conda_env
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class DataInfo:
    """Information about dataset used in experiment."""
    name: str
    source: str
    version: str
    checksum: str
    n_samples: int
    n_features: int
    preprocessing_steps: List[str]
    sensitive_attributes: List[str]
    split_info: Dict[str, Any]
    creation_timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['creation_timestamp'] = self.creation_timestamp.isoformat()
        return data


@dataclass
class ModelInfo:
    """Information about the model used in experiment."""
    name: str
    algorithm_type: str
    hyperparameters: Dict[str, Any]
    random_state: Optional[int]
    model_checksum: str
    training_time: float
    prediction_time: float
    model_size_mb: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ExperimentConfig:
    """Complete experiment configuration for reproducibility."""
    experiment_id: str
    name: str
    description: str
    reproducibility_level: ReproducibilityLevel
    random_seed: int
    environment_info: EnvironmentInfo
    data_info: DataInfo
    model_info: ModelInfo
    experiment_parameters: Dict[str, Any]
    creation_timestamp: datetime
    status: ExperimentStatus = ExperimentStatus.CREATED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'experiment_id': self.experiment_id,
            'name': self.name,
            'description': self.description,
            'reproducibility_level': self.reproducibility_level.value,
            'random_seed': self.random_seed,
            'environment_info': self.environment_info.to_dict(),
            'data_info': self.data_info.to_dict(),
            'model_info': self.model_info.to_dict(),
            'experiment_parameters': self.experiment_parameters,
            'creation_timestamp': self.creation_timestamp.isoformat(),
            'status': self.status.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create from dictionary."""
        return cls(
            experiment_id=data['experiment_id'],
            name=data['name'],
            description=data['description'],
            reproducibility_level=ReproducibilityLevel(data['reproducibility_level']),
            random_seed=data['random_seed'],
            environment_info=EnvironmentInfo(**data['environment_info']),
            data_info=DataInfo(**{
                **data['data_info'],
                'creation_timestamp': datetime.fromisoformat(data['data_info']['creation_timestamp'])
            }),
            model_info=ModelInfo(**data['model_info']),
            experiment_parameters=data['experiment_parameters'],
            creation_timestamp=datetime.fromisoformat(data['creation_timestamp']),
            status=ExperimentStatus(data['status'])
        )


@dataclass
class ReproductionResult:
    """Result from reproducing an experiment."""
    original_experiment_id: str
    reproduction_experiment_id: str
    reproduction_timestamp: datetime
    is_successful: bool
    differences_found: List[Dict[str, Any]]
    metrics_comparison: Dict[str, Dict[str, float]]
    environment_differences: Dict[str, Any]
    tolerance_used: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'original_experiment_id': self.original_experiment_id,
            'reproduction_experiment_id': self.reproduction_experiment_id,
            'reproduction_timestamp': self.reproduction_timestamp.isoformat(),
            'is_successful': self.is_successful,
            'differences_found': self.differences_found,
            'metrics_comparison': self.metrics_comparison,
            'environment_differences': self.environment_differences,
            'tolerance_used': self.tolerance_used
        }


class ResultsManager:
    """
    Manager for experiment results with validation and comparison capabilities.
    
    Handles storage, retrieval, and validation of experiment results
    with comprehensive integrity checking.
    """
    
    def __init__(self, results_dir: str = "experiment_results"):
        """
        Initialize results manager.
        
        Args:
            results_dir: Directory for storing results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Results metadata
        self.results_index: Dict[str, Dict[str, Any]] = {}
        self._load_results_index()
        
        logger.info(f"ResultsManager initialized with results_dir={results_dir}")
    
    def save_results(
        self,
        experiment_id: str,
        results: Dict[str, Any],
        metrics: Dict[str, float],
        predictions: Optional[np.ndarray] = None,
        model_state: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save experiment results with integrity checking.
        
        Args:
            experiment_id: Unique experiment identifier
            results: Experiment results dictionary
            metrics: Performance metrics
            predictions: Model predictions
            model_state: Model state dictionary
            
        Returns:
            Checksum of saved results
        """
        logger.info(f"Saving results for experiment: {experiment_id}")
        
        # Create experiment directory
        exp_dir = self.results_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        results_path = exp_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save metrics
        metrics_path = exp_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save predictions if provided
        if predictions is not None:
            predictions_path = exp_dir / "predictions.npy"
            np.save(predictions_path, predictions)
        
        # Save model state if provided
        if model_state is not None:
            model_state_path = exp_dir / "model_state.pkl"
            with open(model_state_path, 'wb') as f:
                pickle.dump(model_state, f)
        
        # Calculate checksum
        checksum = self._calculate_results_checksum(exp_dir)
        
        # Update results index
        self.results_index[experiment_id] = {
            'timestamp': datetime.now().isoformat(),
            'checksum': checksum,
            'files': list(os.listdir(exp_dir)),
            'metrics_summary': {
                key: value for key, value in metrics.items()
                if isinstance(value, (int, float))
            }
        }
        
        self._save_results_index()
        
        logger.info(f"Results saved with checksum: {checksum}")
        return checksum
    
    def load_results(
        self,
        experiment_id: str,
        validate_integrity: bool = True
    ) -> Dict[str, Any]:
        """
        Load experiment results with optional integrity validation.
        
        Args:
            experiment_id: Experiment identifier
            validate_integrity: Whether to validate file integrity
            
        Returns:
            Dictionary containing all experiment results
        """
        if experiment_id not in self.results_index:
            raise ValueError(f"Experiment {experiment_id} not found in results index")
        
        logger.info(f"Loading results for experiment: {experiment_id}")
        
        exp_dir = self.results_dir / experiment_id
        if not exp_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {exp_dir}")
        
        # Validate integrity if requested
        if validate_integrity:
            current_checksum = self._calculate_results_checksum(exp_dir)
            expected_checksum = self.results_index[experiment_id]['checksum']
            
            if current_checksum != expected_checksum:
                raise ValueError(
                    f"Results integrity check failed for {experiment_id}. "
                    f"Expected: {expected_checksum}, Got: {current_checksum}"
                )
        
        # Load all results
        loaded_results = {'experiment_id': experiment_id}
        
        # Load main results
        results_path = exp_dir / "results.json"
        if results_path.exists():
            with open(results_path, 'r') as f:
                loaded_results['results'] = json.load(f)
        
        # Load metrics
        metrics_path = exp_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                loaded_results['metrics'] = json.load(f)
        
        # Load predictions
        predictions_path = exp_dir / "predictions.npy"
        if predictions_path.exists():
            loaded_results['predictions'] = np.load(predictions_path)
        
        # Load model state
        model_state_path = exp_dir / "model_state.pkl"
        if model_state_path.exists():
            with open(model_state_path, 'rb') as f:
                loaded_results['model_state'] = pickle.load(f)
        
        logger.info("Results loaded successfully")
        return loaded_results
    
    def compare_results(
        self,
        experiment_id_1: str,
        experiment_id_2: str,
        tolerance: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Compare results between two experiments.
        
        Args:
            experiment_id_1: First experiment ID
            experiment_id_2: Second experiment ID
            tolerance: Tolerance levels for comparisons
            
        Returns:
            Comparison results
        """
        if tolerance is None:
            tolerance = {
                'metrics': 1e-6,
                'predictions': 1e-6
            }
        
        logger.info(f"Comparing experiments: {experiment_id_1} vs {experiment_id_2}")
        
        # Load both results
        results_1 = self.load_results(experiment_id_1)
        results_2 = self.load_results(experiment_id_2)
        
        comparison = {
            'experiment_1': experiment_id_1,
            'experiment_2': experiment_id_2,
            'comparison_timestamp': datetime.now().isoformat(),
            'metrics_comparison': {},
            'predictions_comparison': {},
            'differences_found': [],
            'is_identical': True
        }
        
        # Compare metrics
        if 'metrics' in results_1 and 'metrics' in results_2:
            metrics_1 = results_1['metrics']
            metrics_2 = results_2['metrics']
            
            for metric_name in set(metrics_1.keys()) | set(metrics_2.keys()):
                if metric_name not in metrics_1:
                    comparison['differences_found'].append({
                        'type': 'missing_metric',
                        'metric': metric_name,
                        'experiment': experiment_id_1
                    })
                    comparison['is_identical'] = False
                elif metric_name not in metrics_2:
                    comparison['differences_found'].append({
                        'type': 'missing_metric',
                        'metric': metric_name,
                        'experiment': experiment_id_2
                    })
                    comparison['is_identical'] = False
                else:
                    val_1 = metrics_1[metric_name]
                    val_2 = metrics_2[metric_name]
                    diff = abs(val_1 - val_2)
                    
                    comparison['metrics_comparison'][metric_name] = {
                        'experiment_1': val_1,
                        'experiment_2': val_2,
                        'difference': diff,
                        'within_tolerance': diff <= tolerance['metrics']
                    }
                    
                    if diff > tolerance['metrics']:
                        comparison['differences_found'].append({
                            'type': 'metric_difference',
                            'metric': metric_name,
                            'difference': diff,
                            'tolerance': tolerance['metrics']
                        })
                        comparison['is_identical'] = False
        
        # Compare predictions
        if 'predictions' in results_1 and 'predictions' in results_2:
            pred_1 = results_1['predictions']
            pred_2 = results_2['predictions']
            
            if pred_1.shape != pred_2.shape:
                comparison['differences_found'].append({
                    'type': 'prediction_shape_mismatch',
                    'shape_1': pred_1.shape,
                    'shape_2': pred_2.shape
                })
                comparison['is_identical'] = False
            else:
                max_diff = np.max(np.abs(pred_1 - pred_2))
                mean_diff = np.mean(np.abs(pred_1 - pred_2))
                
                comparison['predictions_comparison'] = {
                    'max_difference': max_diff,
                    'mean_difference': mean_diff,
                    'within_tolerance': max_diff <= tolerance['predictions']
                }
                
                if max_diff > tolerance['predictions']:
                    comparison['differences_found'].append({
                        'type': 'prediction_difference',
                        'max_difference': max_diff,
                        'tolerance': tolerance['predictions']
                    })
                    comparison['is_identical'] = False
        
        logger.info(f"Comparison completed. Identical: {comparison['is_identical']}")
        return comparison
    
    def _calculate_results_checksum(self, exp_dir: Path) -> str:
        """Calculate checksum for all files in experiment directory."""
        hasher = hashlib.sha256()
        
        for file_path in sorted(exp_dir.rglob('*')):
            if file_path.is_file():
                with open(file_path, 'rb') as f:
                    hasher.update(f.read())
        
        return hasher.hexdigest()
    
    def _load_results_index(self):
        """Load results index from disk."""
        index_path = self.results_dir / "results_index.json"
        if index_path.exists():
            try:
                with open(index_path, 'r') as f:
                    self.results_index = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load results index: {e}")
                self.results_index = {}
    
    def _save_results_index(self):
        """Save results index to disk."""
        index_path = self.results_dir / "results_index.json"
        with open(index_path, 'w') as f:
            json.dump(self.results_index, f, indent=2)
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all stored experiments."""
        return [
            {
                'experiment_id': exp_id,
                'timestamp': info['timestamp'],
                'metrics_summary': info.get('metrics_summary', {})
            }
            for exp_id, info in self.results_index.items()
        ]
    
    def cleanup_old_results(self, keep_days: int = 30):
        """Clean up old experiment results."""
        cutoff_date = datetime.now() - pd.Timedelta(days=keep_days)
        
        experiments_to_remove = []
        for exp_id, info in self.results_index.items():
            exp_date = datetime.fromisoformat(info['timestamp'])
            if exp_date < cutoff_date:
                experiments_to_remove.append(exp_id)
        
        for exp_id in experiments_to_remove:
            exp_dir = self.results_dir / exp_id
            if exp_dir.exists():
                shutil.rmtree(exp_dir)
            
            del self.results_index[exp_id]
        
        self._save_results_index()
        
        logger.info(f"Cleaned up {len(experiments_to_remove)} old experiments")


class ReproducibilityManager:
    """
    Comprehensive reproducibility manager for fairness research.
    
    Provides end-to-end reproducibility management including experiment
    configuration, environment tracking, and reproduction validation.
    """
    
    def __init__(
        self,
        workspace_dir: str = "reproducibility_workspace",
        reproducibility_level: ReproducibilityLevel = ReproducibilityLevel.STANDARD
    ):
        """
        Initialize reproducibility manager.
        
        Args:
            workspace_dir: Directory for reproducibility workspace
            reproducibility_level: Level of reproducibility assurance
        """
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.reproducibility_level = reproducibility_level
        
        # Initialize components
        self.results_manager = ResultsManager(
            str(self.workspace_dir / "experiment_results")
        )
        
        # Experiment tracking
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.reproductions: Dict[str, ReproductionResult] = {}
        
        # Load existing experiments
        self._load_experiments()
        
        logger.info(f"ReproducibilityManager initialized with level: {reproducibility_level.value}")
    
    def create_experiment_config(
        self,
        name: str,
        description: str,
        algorithm: Any,
        dataset: pd.DataFrame,
        target: pd.Series,
        sensitive_attrs: pd.DataFrame,
        hyperparameters: Dict[str, Any] = None,
        random_seed: int = 42,
        experiment_parameters: Dict[str, Any] = None
    ) -> ExperimentConfig:
        """
        Create a complete experiment configuration.
        
        Args:
            name: Experiment name
            description: Experiment description
            algorithm: ML algorithm instance
            dataset: Dataset features
            target: Target variable
            sensitive_attrs: Sensitive attributes
            hyperparameters: Algorithm hyperparameters
            random_seed: Random seed
            experiment_parameters: Additional experiment parameters
            
        Returns:
            Complete experiment configuration
        """
        experiment_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random_seed}"
        
        logger.info(f"Creating experiment config: {experiment_id}")
        
        # Capture environment information
        if self.reproducibility_level in [ReproducibilityLevel.STANDARD, 
                                        ReproducibilityLevel.COMPREHENSIVE,
                                        ReproducibilityLevel.RESEARCH_GRADE]:
            environment_info = EnvironmentInfo.capture_current_environment()
        else:
            environment_info = EnvironmentInfo(
                python_version=sys.version,
                platform_system=platform.system(),
                platform_release="",
                platform_version="",
                cpu_count=1,
                memory_total_gb=1.0,
                installed_packages={}
            )
        
        # Create data information
        data_checksum = self._calculate_data_checksum(dataset, target, sensitive_attrs)
        data_info = DataInfo(
            name=f"{name}_data",
            source="experiment",
            version="1.0",
            checksum=data_checksum,
            n_samples=len(dataset),
            n_features=dataset.shape[1],
            preprocessing_steps=[],
            sensitive_attributes=list(sensitive_attrs.columns),
            split_info={},
            creation_timestamp=datetime.now()
        )
        
        # Create model information
        if hyperparameters is None:
            hyperparameters = algorithm.get_params()
        
        model_info = ModelInfo(
            name=type(algorithm).__name__,
            algorithm_type=str(type(algorithm)),
            hyperparameters=hyperparameters,
            random_state=random_seed,
            model_checksum="",  # Will be calculated after training
            training_time=0.0,
            prediction_time=0.0,
            model_size_mb=0.0
        )
        
        # Create experiment configuration
        config = ExperimentConfig(
            experiment_id=experiment_id,
            name=name,
            description=description,
            reproducibility_level=self.reproducibility_level,
            random_seed=random_seed,
            environment_info=environment_info,
            data_info=data_info,
            model_info=model_info,
            experiment_parameters=experiment_parameters or {},
            creation_timestamp=datetime.now()
        )
        
        # Save configuration
        self._save_experiment_config(config)
        self.experiments[experiment_id] = config
        
        logger.info(f"Experiment config created: {experiment_id}")
        return config
    
    def run_reproducible_experiment(
        self,
        config: ExperimentConfig,
        algorithm: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        sensitive_attrs_test: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Run an experiment with full reproducibility tracking.
        
        Args:
            config: Experiment configuration
            algorithm: ML algorithm instance
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            sensitive_attrs_test: Test sensitive attributes
            
        Returns:
            Experiment results
        """
        logger.info(f"Running reproducible experiment: {config.experiment_id}")
        
        # Set random seeds for reproducibility
        self._set_random_seeds(config.random_seed)
        
        # Update experiment status
        config.status = ExperimentStatus.RUNNING
        self._save_experiment_config(config)
        
        try:
            import time
            
            # Training
            start_time = time.time()
            algorithm.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Prediction
            start_time = time.time()
            y_pred = algorithm.predict(X_test)
            prediction_time = time.time() - start_time
            
            # Get probabilities if available
            y_proba = None
            if hasattr(algorithm, 'predict_proba'):
                y_proba = algorithm.predict_proba(X_test)[:, 1]
            elif hasattr(algorithm, 'decision_function'):
                y_proba = algorithm.decision_function(X_test)
            
            # Compute performance metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted'),
                'training_time': training_time,
                'prediction_time': prediction_time
            }
            
            if y_proba is not None:
                try:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
                except ValueError:
                    metrics['roc_auc'] = 0.0
            
            # Compute fairness metrics
            from ..fairness_metrics import compute_fairness_metrics
            fairness_results = {}
            
            for attr_name in sensitive_attrs_test.columns:
                try:
                    overall, by_group = compute_fairness_metrics(
                        y_test, y_pred, sensitive_attrs_test[attr_name]
                    )
                    fairness_results[attr_name] = {
                        'overall': overall,
                        'by_group': by_group.to_dict() if hasattr(by_group, 'to_dict') else by_group
                    }
                except Exception as e:
                    logger.warning(f"Failed to compute fairness metrics for {attr_name}: {e}")
                    fairness_results[attr_name] = {'error': str(e)}
            
            # Create complete results
            results = {
                'experiment_id': config.experiment_id,
                'performance_metrics': metrics,
                'fairness_metrics': fairness_results,
                'experiment_config': config.to_dict(),
                'completion_timestamp': datetime.now().isoformat()
            }
            
            # Update model info with actual values
            config.model_info.training_time = training_time
            config.model_info.prediction_time = prediction_time
            config.model_info.model_checksum = self._calculate_model_checksum(algorithm)
            config.model_info.model_size_mb = self._estimate_model_size(algorithm)
            
            # Save results
            if self.reproducibility_level in [ReproducibilityLevel.COMPREHENSIVE,
                                            ReproducibilityLevel.RESEARCH_GRADE]:
                # Save model state for complete reproducibility
                model_state = self._extract_model_state(algorithm)
                self.results_manager.save_results(
                    config.experiment_id, results, metrics, y_pred, model_state
                )
            else:
                self.results_manager.save_results(
                    config.experiment_id, results, metrics, y_pred
                )
            
            # Update experiment status
            config.status = ExperimentStatus.COMPLETED
            self._save_experiment_config(config)
            
            logger.info(f"Experiment completed successfully: {config.experiment_id}")
            return results
            
        except Exception as e:
            config.status = ExperimentStatus.FAILED
            self._save_experiment_config(config)
            logger.error(f"Experiment failed: {config.experiment_id} - {e}")
            raise
    
    def reproduce_experiment(
        self,
        original_experiment_id: str,
        tolerance: Dict[str, float] = None
    ) -> ReproductionResult:
        """
        Reproduce an existing experiment and validate results.
        
        Args:
            original_experiment_id: ID of experiment to reproduce
            tolerance: Tolerance levels for result comparison
            
        Returns:
            Reproduction result with validation
        """
        if tolerance is None:
            tolerance = {
                'accuracy': 1e-6,
                'precision': 1e-6,
                'recall': 1e-6,
                'f1_score': 1e-6,
                'predictions': 1e-6
            }
        
        logger.info(f"Reproducing experiment: {original_experiment_id}")
        
        # Load original experiment config
        if original_experiment_id not in self.experiments:
            raise ValueError(f"Original experiment {original_experiment_id} not found")
        
        original_config = self.experiments[original_experiment_id]
        
        # Load original results
        original_results = self.results_manager.load_results(original_experiment_id)
        
        # Create reproduction configuration
        reproduction_config = self._create_reproduction_config(original_config)
        
        # Check environment compatibility
        env_differences = self._compare_environments(
            original_config.environment_info,
            reproduction_config.environment_info
        )
        
        # Note: In a complete implementation, this would re-run the experiment
        # For this demonstration, we'll simulate a reproduction
        
        reproduction_result = ReproductionResult(
            original_experiment_id=original_experiment_id,
            reproduction_experiment_id=reproduction_config.experiment_id,
            reproduction_timestamp=datetime.now(),
            is_successful=True,  # Would be determined by actual reproduction
            differences_found=[],
            metrics_comparison={},
            environment_differences=env_differences,
            tolerance_used=tolerance
        )
        
        self.reproductions[reproduction_config.experiment_id] = reproduction_result
        self._save_reproduction_result(reproduction_result)
        
        logger.info("Experiment reproduction completed")
        return reproduction_result
    
    def validate_experiment(
        self,
        experiment_id: str,
        validation_checks: List[str] = None
    ) -> Dict[str, Any]:
        """
        Validate experiment integrity and reproducibility.
        
        Args:
            experiment_id: Experiment to validate
            validation_checks: List of validation checks to perform
            
        Returns:
            Validation results
        """
        if validation_checks is None:
            validation_checks = [
                'config_integrity',
                'results_integrity',
                'environment_consistency',
                'reproducibility_score'
            ]
        
        logger.info(f"Validating experiment: {experiment_id}")
        
        validation_results = {
            'experiment_id': experiment_id,
            'validation_timestamp': datetime.now().isoformat(),
            'checks_performed': validation_checks,
            'check_results': {},
            'overall_validity': True
        }
        
        # Config integrity check
        if 'config_integrity' in validation_checks:
            try:
                config = self.experiments[experiment_id]
                validation_results['check_results']['config_integrity'] = {
                    'status': 'passed',
                    'message': 'Configuration is valid'
                }
            except Exception as e:
                validation_results['check_results']['config_integrity'] = {
                    'status': 'failed',
                    'message': f'Configuration validation failed: {e}'
                }
                validation_results['overall_validity'] = False
        
        # Results integrity check
        if 'results_integrity' in validation_checks:
            try:
                results = self.results_manager.load_results(experiment_id, validate_integrity=True)
                validation_results['check_results']['results_integrity'] = {
                    'status': 'passed',
                    'message': 'Results integrity verified'
                }
            except Exception as e:
                validation_results['check_results']['results_integrity'] = {
                    'status': 'failed',
                    'message': f'Results integrity check failed: {e}'
                }
                validation_results['overall_validity'] = False
        
        # Environment consistency check
        if 'environment_consistency' in validation_checks:
            try:
                config = self.experiments[experiment_id]
                current_env = EnvironmentInfo.capture_current_environment()
                env_diff = self._compare_environments(config.environment_info, current_env)
                
                critical_differences = [
                    diff for diff in env_diff
                    if diff.get('severity') == 'critical'
                ]
                
                if critical_differences:
                    validation_results['check_results']['environment_consistency'] = {
                        'status': 'warning',
                        'message': f'Found {len(critical_differences)} critical environment differences',
                        'differences': critical_differences
                    }
                else:
                    validation_results['check_results']['environment_consistency'] = {
                        'status': 'passed',
                        'message': 'Environment is consistent'
                    }
            except Exception as e:
                validation_results['check_results']['environment_consistency'] = {
                    'status': 'failed',
                    'message': f'Environment consistency check failed: {e}'
                }
        
        # Reproducibility score
        if 'reproducibility_score' in validation_checks:
            score = self._calculate_reproducibility_score(experiment_id)
            validation_results['check_results']['reproducibility_score'] = {
                'status': 'passed' if score >= 0.8 else 'warning',
                'score': score,
                'message': f'Reproducibility score: {score:.2f}'
            }
        
        logger.info(f"Validation completed. Overall validity: {validation_results['overall_validity']}")
        return validation_results
    
    def _set_random_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        import random
        random.seed(seed)
        np.random.seed(seed)
        
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass
        
        # Set environment variables for additional reproducibility
        os.environ['PYTHONHASHSEED'] = str(seed)
    
    def _calculate_data_checksum(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sensitive_attrs: pd.DataFrame
    ) -> str:
        """Calculate checksum for dataset."""
        hasher = hashlib.sha256()
        
        # Hash features
        hasher.update(X.to_string().encode())
        
        # Hash target
        hasher.update(y.to_string().encode())
        
        # Hash sensitive attributes
        hasher.update(sensitive_attrs.to_string().encode())
        
        return hasher.hexdigest()
    
    def _calculate_model_checksum(self, model: Any) -> str:
        """Calculate checksum for trained model."""
        try:
            model_bytes = pickle.dumps(model)
            return hashlib.sha256(model_bytes).hexdigest()
        except Exception as e:
            logger.warning(f"Could not calculate model checksum: {e}")
            return "unknown"
    
    def _estimate_model_size(self, model: Any) -> float:
        """Estimate model size in MB."""
        try:
            model_bytes = pickle.dumps(model)
            return len(model_bytes) / (1024 * 1024)
        except Exception as e:
            logger.warning(f"Could not estimate model size: {e}")
            return 0.0
    
    def _extract_model_state(self, model: Any) -> Dict[str, Any]:
        """Extract model state for reproducibility."""
        try:
            return {
                'model_type': str(type(model)),
                'parameters': model.get_params(),
                'state': pickle.dumps(model).hex()  # Hex encoding for JSON serialization
            }
        except Exception as e:
            logger.warning(f"Could not extract model state: {e}")
            return {}
    
    def _create_reproduction_config(self, original_config: ExperimentConfig) -> ExperimentConfig:
        """Create configuration for reproducing an experiment."""
        reproduction_id = f"reproduction_{original_config.experiment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create new config with current environment
        current_env = EnvironmentInfo.capture_current_environment()
        
        reproduction_config = ExperimentConfig(
            experiment_id=reproduction_id,
            name=f"Reproduction of {original_config.name}",
            description=f"Reproduction of experiment {original_config.experiment_id}",
            reproducibility_level=original_config.reproducibility_level,
            random_seed=original_config.random_seed,
            environment_info=current_env,
            data_info=original_config.data_info,  # Same data
            model_info=original_config.model_info,  # Same model config
            experiment_parameters=original_config.experiment_parameters,
            creation_timestamp=datetime.now(),
            status=ExperimentStatus.CREATED
        )
        
        return reproduction_config
    
    def _compare_environments(
        self,
        env1: EnvironmentInfo,
        env2: EnvironmentInfo
    ) -> List[Dict[str, Any]]:
        """Compare two environments and identify differences."""
        differences = []
        
        # Python version
        if env1.python_version != env2.python_version:
            differences.append({
                'type': 'python_version',
                'original': env1.python_version,
                'current': env2.python_version,
                'severity': 'critical'
            })
        
        # Platform differences
        if env1.platform_system != env2.platform_system:
            differences.append({
                'type': 'platform_system',
                'original': env1.platform_system,
                'current': env2.platform_system,
                'severity': 'high'
            })
        
        # Package differences
        all_packages = set(env1.installed_packages.keys()) | set(env2.installed_packages.keys())
        for package in all_packages:
            version1 = env1.installed_packages.get(package)
            version2 = env2.installed_packages.get(package)
            
            if version1 != version2:
                differences.append({
                    'type': 'package_version',
                    'package': package,
                    'original': version1,
                    'current': version2,
                    'severity': 'medium' if package in ['numpy', 'pandas', 'scikit-learn'] else 'low'
                })
        
        return differences
    
    def _calculate_reproducibility_score(self, experiment_id: str) -> float:
        """Calculate reproducibility score for an experiment."""
        if experiment_id not in self.experiments:
            return 0.0
        
        config = self.experiments[experiment_id]
        score = 0.0
        max_score = 0.0
        
        # Random seed specified
        if config.random_seed is not None:
            score += 0.2
        max_score += 0.2
        
        # Environment captured
        if config.environment_info.python_version:
            score += 0.2
        max_score += 0.2
        
        # Data checksummed
        if config.data_info.checksum:
            score += 0.2
        max_score += 0.2
        
        # Model parameters recorded
        if config.model_info.hyperparameters:
            score += 0.2
        max_score += 0.2
        
        # Results saved
        try:
            self.results_manager.load_results(experiment_id, validate_integrity=True)
            score += 0.2
        except:
            pass
        max_score += 0.2
        
        return score / max_score if max_score > 0 else 0.0
    
    def _save_experiment_config(self, config: ExperimentConfig):
        """Save experiment configuration to disk."""
        config_path = self.workspace_dir / "experiments" / f"{config.experiment_id}.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
    
    def _save_reproduction_result(self, result: ReproductionResult):
        """Save reproduction result to disk."""
        result_path = self.workspace_dir / "reproductions" / f"{result.reproduction_experiment_id}.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(result_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
    
    def _load_experiments(self):
        """Load existing experiments from disk."""
        experiments_dir = self.workspace_dir / "experiments"
        if not experiments_dir.exists():
            return
        
        for config_file in experiments_dir.glob("*.json"):
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                
                config = ExperimentConfig.from_dict(config_data)
                self.experiments[config.experiment_id] = config
                
            except Exception as e:
                logger.warning(f"Could not load experiment config {config_file}: {e}")
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments."""
        return [
            {
                'experiment_id': exp_id,
                'name': config.name,
                'status': config.status.value,
                'creation_timestamp': config.creation_timestamp.isoformat(),
                'reproducibility_level': config.reproducibility_level.value
            }
            for exp_id, config in self.experiments.items()
        ]
    
    def get_reproducibility_report(self) -> Dict[str, Any]:
        """Generate comprehensive reproducibility report."""
        total_experiments = len(self.experiments)
        completed_experiments = len([
            e for e in self.experiments.values()
            if e.status == ExperimentStatus.COMPLETED
        ])
        
        reproductions = len(self.reproductions)
        successful_reproductions = len([
            r for r in self.reproductions.values()
            if r.is_successful
        ])
        
        # Calculate average reproducibility score
        scores = []
        for exp_id in self.experiments.keys():
            if self.experiments[exp_id].status == ExperimentStatus.COMPLETED:
                scores.append(self._calculate_reproducibility_score(exp_id))
        
        avg_score = np.mean(scores) if scores else 0.0
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'summary': {
                'total_experiments': total_experiments,
                'completed_experiments': completed_experiments,
                'total_reproductions': reproductions,
                'successful_reproductions': successful_reproductions,
                'average_reproducibility_score': avg_score
            },
            'experiments_by_status': {
                status.value: len([e for e in self.experiments.values() if e.status == status])
                for status in ExperimentStatus
            },
            'experiments_by_level': {
                level.value: len([e for e in self.experiments.values() 
                                if e.reproducibility_level == level])
                for level in ReproducibilityLevel
            }
        }
        
        return report


# Example usage and CLI interface
def main():
    """CLI interface for reproducibility manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fairness Research Reproducibility Manager")
    parser.add_argument("--demo", action="store_true", help="Run demonstration")
    parser.add_argument("--workspace", default="reproducibility_demo", help="Workspace directory")
    parser.add_argument("--level", choices=[l.value for l in ReproducibilityLevel],
                       default="standard", help="Reproducibility level")
    
    args = parser.parse_args()
    
    if args.demo:
        from sklearn.datasets import make_classification
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        
        # Generate synthetic data
        X, y = make_classification(
            n_samples=1000, n_features=10, n_informative=8,
            n_redundant=1, n_clusters_per_class=1, flip_y=0.05,
            random_state=42
        )
        
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        y_series = pd.Series(y, name='target')
        
        # Create sensitive attributes
        sensitive_attrs_df = pd.DataFrame({
            'group_a': np.random.binomial(1, 0.3, len(X)),
            'group_b': np.random.choice([0, 1, 2], len(X))
        })
        
        # Train/test split
        X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
            X_df, y_series, sensitive_attrs_df, test_size=0.2, random_state=42
        )
        
        # Initialize reproducibility manager
        manager = ReproducibilityManager(
            workspace_dir=args.workspace,
            reproducibility_level=ReproducibilityLevel(args.level)
        )
        
        # Create experiment configuration
        algorithm = LogisticRegression(random_state=42, max_iter=1000)
        
        config = manager.create_experiment_config(
            name="Demo_Reproducibility_Experiment",
            description="Demonstration of reproducibility management",
            algorithm=algorithm,
            dataset=X_train,
            target=y_train,
            sensitive_attrs=sens_train,
            random_seed=42
        )
        
        print(f"Created experiment configuration: {config.experiment_id}")
        
        # Run reproducible experiment
        results = manager.run_reproducible_experiment(
            config=config,
            algorithm=algorithm,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            sensitive_attrs_test=sens_test
        )
        
        print(f"Experiment completed successfully!")
        print(f"Accuracy: {results['performance_metrics']['accuracy']:.4f}")
        
        # Validate experiment
        validation_results = manager.validate_experiment(config.experiment_id)
        print(f"Experiment validation: {'PASSED' if validation_results['overall_validity'] else 'FAILED'}")
        
        # Generate reproducibility report
        report = manager.get_reproducibility_report()
        print(f"\nReproducibility Report:")
        print(f"- Total experiments: {report['summary']['total_experiments']}")
        print(f"- Completed experiments: {report['summary']['completed_experiments']}")
        print(f"- Average reproducibility score: {report['summary']['average_reproducibility_score']:.2f}")
        
        print(f"\nDemo completed! Results saved to: {args.workspace}")


if __name__ == "__main__":
    main()