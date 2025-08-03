"""
Automated fairness and hyperparameter optimization.

Advanced optimization algorithms for finding optimal fairness-accuracy
tradeoffs using multi-objective optimization and automated hyperparameter tuning.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import itertools
import warnings

from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score

try:
    from scipy.optimize import minimize, differential_evolution
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from ..fairness_metrics import compute_fairness_metrics
from ..bias_mitigator import reweight_samples, postprocess_equalized_odds, expgrad_demographic_parity
from ..baseline_model import train_baseline_model
from ..logging_config import get_logger

logger = get_logger(__name__)


class OptimizationObjective(Enum):
    """Optimization objectives."""
    ACCURACY = "accuracy"
    FAIRNESS = "fairness"
    BALANCED = "balanced"
    CUSTOM = "custom"


@dataclass
class OptimizationResult:
    """Result of optimization process."""
    best_params: Dict[str, Any]
    best_score: float
    best_model: BaseEstimator
    optimization_history: List[Dict[str, Any]]
    fairness_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]
    pareto_front: Optional[List[Tuple[float, float]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'fairness_metrics': self.fairness_metrics,
            'performance_metrics': self.performance_metrics,
            'n_iterations': len(self.optimization_history),
            'pareto_front_size': len(self.pareto_front) if self.pareto_front else 0
        }


class AutoFairnessOptimizer:
    """
    Automated fairness optimization using multi-objective optimization.
    
    Finds optimal fairness-accuracy tradeoffs using evolutionary algorithms
    and Pareto optimization.
    """
    
    def __init__(
        self,
        protected_attributes: List[str],
        fairness_constraints: Optional[Dict[str, float]] = None,
        optimization_objective: OptimizationObjective = OptimizationObjective.BALANCED,
        max_iterations: int = 100,
        population_size: int = 50,
        random_state: int = 42
    ):
        """
        Initialize fairness optimizer.
        
        Args:
            protected_attributes: List of protected attribute names
            fairness_constraints: Constraints on fairness metrics
            optimization_objective: Optimization objective
            max_iterations: Maximum optimization iterations
            population_size: Population size for evolutionary algorithm
            random_state: Random seed
        """
        self.protected_attributes = protected_attributes
        self.fairness_constraints = fairness_constraints or {
            'demographic_parity_difference': 0.1,
            'equalized_odds_difference': 0.1
        }
        self.optimization_objective = optimization_objective
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.random_state = random_state
        
        # Optimization state
        self.optimization_history: List[Dict[str, Any]] = []
        self.pareto_front: List[Tuple[float, float]] = []
        
        logger.info("AutoFairnessOptimizer initialized")
    
    def optimize(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        base_model: BaseEstimator,
        mitigation_methods: Optional[List[str]] = None
    ) -> OptimizationResult:
        """
        Optimize fairness-accuracy tradeoff.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            base_model: Base model to optimize
            mitigation_methods: List of mitigation methods to try
            
        Returns:
            OptimizationResult with best configuration
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for optimization")
        
        logger.info("Starting fairness optimization")
        
        mitigation_methods = mitigation_methods or ['baseline', 'reweight', 'postprocess', 'expgrad']
        
        # Define search space
        search_space = self._define_search_space(mitigation_methods)
        
        # Multi-objective optimization
        if self.optimization_objective == OptimizationObjective.BALANCED:
            result = self._multi_objective_optimization(
                X_train, y_train, X_val, y_val, base_model, search_space
            )
        else:
            result = self._single_objective_optimization(
                X_train, y_train, X_val, y_val, base_model, search_space
            )
        
        logger.info(f"Optimization completed with score: {result.best_score:.4f}")
        return result
    
    def _define_search_space(self, mitigation_methods: List[str]) -> Dict[str, Any]:
        """Define hyperparameter search space."""
        search_space = {
            'mitigation_method': mitigation_methods,
            'model_params': {
                'C': (0.001, 100.0),  # Regularization for LogisticRegression
                'max_iter': [100, 500, 1000],
                'solver': ['liblinear', 'lbfgs']
            },
            'preprocessing': {
                'scale_features': [True, False],
                'handle_imbalance': [True, False]
            }
        }
        
        # Add method-specific parameters
        if 'reweight' in mitigation_methods:
            search_space['reweight_params'] = {
                'smoothing': (0.0, 1.0)
            }
        
        if 'postprocess' in mitigation_methods:
            search_space['postprocess_params'] = {
                'constraint': ['equalized_odds', 'demographic_parity']
            }
        
        return search_space
    
    def _multi_objective_optimization(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        base_model: BaseEstimator,
        search_space: Dict[str, Any]
    ) -> OptimizationResult:
        """Perform multi-objective optimization using NSGA-II inspired approach."""
        try:
            # Initialize population
            population = self._initialize_population(search_space)
            
            best_individual = None
            best_score = -np.inf
            
            for iteration in range(self.max_iterations):
                # Evaluate population
                evaluated_population = []
                
                for individual in population:
                    try:
                        # Train and evaluate model
                        model, fairness_score, accuracy_score = self._evaluate_individual(
                            individual, X_train, y_train, X_val, y_val, base_model
                        )
                        
                        # Multi-objective score
                        if self.optimization_objective == OptimizationObjective.BALANCED:
                            # Balance accuracy and fairness
                            combined_score = 0.6 * accuracy_score + 0.4 * (1 - fairness_score)
                        else:
                            combined_score = accuracy_score
                        
                        evaluated_individual = {
                            'params': individual,
                            'model': model,
                            'accuracy': accuracy_score,
                            'fairness_violation': fairness_score,
                            'combined_score': combined_score
                        }
                        
                        evaluated_population.append(evaluated_individual)
                        self.optimization_history.append(evaluated_individual)
                        
                        # Update Pareto front
                        self.pareto_front.append((accuracy_score, fairness_score))
                        
                        # Track best individual
                        if combined_score > best_score:
                            best_score = combined_score
                            best_individual = evaluated_individual
                            
                    except Exception as e:
                        logger.warning(f"Failed to evaluate individual: {e}")
                        continue
                
                # Selection and mutation for next generation
                population = self._evolve_population(evaluated_population, search_space)
                
                if iteration % 10 == 0:
                    logger.info(f"Iteration {iteration}: best score = {best_score:.4f}")
            
            # Compute final metrics
            if best_individual:
                fairness_metrics, performance_metrics = self._compute_final_metrics(
                    best_individual['model'], X_val, y_val
                )
                
                return OptimizationResult(
                    best_params=best_individual['params'],
                    best_score=best_score,
                    best_model=best_individual['model'],
                    optimization_history=self.optimization_history,
                    fairness_metrics=fairness_metrics,
                    performance_metrics=performance_metrics,
                    pareto_front=self.pareto_front
                )
            else:
                raise ValueError("Optimization failed to find valid solution")
                
        except Exception as e:
            logger.error(f"Multi-objective optimization failed: {e}")
            raise
    
    def _single_objective_optimization(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        base_model: BaseEstimator,
        search_space: Dict[str, Any]
    ) -> OptimizationResult:
        """Perform single-objective optimization using grid search."""
        best_score = -np.inf
        best_params = None
        best_model = None
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(search_space)
        
        for i, params in enumerate(param_combinations):
            if i >= self.max_iterations:
                break
            
            try:
                model, fairness_score, accuracy_score = self._evaluate_individual(
                    params, X_train, y_train, X_val, y_val, base_model
                )
                
                # Single objective score
                if self.optimization_objective == OptimizationObjective.ACCURACY:
                    score = accuracy_score
                elif self.optimization_objective == OptimizationObjective.FAIRNESS:
                    score = 1 - fairness_score
                else:
                    score = 0.7 * accuracy_score + 0.3 * (1 - fairness_score)
                
                evaluation = {
                    'params': params,
                    'model': model,
                    'accuracy': accuracy_score,
                    'fairness_violation': fairness_score,
                    'score': score
                }
                
                self.optimization_history.append(evaluation)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_model = model
                    
            except Exception as e:
                logger.warning(f"Failed to evaluate parameters: {e}")
                continue
        
        if best_model:
            fairness_metrics, performance_metrics = self._compute_final_metrics(
                best_model, X_val, y_val
            )
            
            return OptimizationResult(
                best_params=best_params,
                best_score=best_score,
                best_model=best_model,
                optimization_history=self.optimization_history,
                fairness_metrics=fairness_metrics,
                performance_metrics=performance_metrics
            )
        else:
            raise ValueError("Optimization failed to find valid solution")
    
    def _initialize_population(self, search_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Initialize random population for evolutionary algorithm."""
        population = []
        np.random.seed(self.random_state)
        
        for _ in range(self.population_size):
            individual = self._sample_random_params(search_space)
            population.append(individual)
        
        return population
    
    def _sample_random_params(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Sample random parameters from search space."""
        params = {}
        
        # Sample mitigation method
        params['mitigation_method'] = np.random.choice(search_space['mitigation_method'])
        
        # Sample model parameters
        model_params = {}
        for param, values in search_space['model_params'].items():
            if isinstance(values, tuple):
                # Continuous parameter
                model_params[param] = np.random.uniform(values[0], values[1])
            else:
                # Discrete parameter
                model_params[param] = np.random.choice(values)
        params['model_params'] = model_params
        
        # Sample preprocessing parameters
        preprocessing = {}
        for param, values in search_space['preprocessing'].items():
            preprocessing[param] = np.random.choice(values)
        params['preprocessing'] = preprocessing
        
        return params
    
    def _evaluate_individual(
        self,
        params: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        base_model: BaseEstimator
    ) -> Tuple[BaseEstimator, float, float]:
        """Evaluate individual parameter configuration."""
        try:
            # Prepare data
            X_train_processed = X_train.copy()
            X_val_processed = X_val.copy()
            
            if params['preprocessing']['scale_features']:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_train_processed = pd.DataFrame(
                    scaler.fit_transform(X_train_processed),
                    columns=X_train_processed.columns,
                    index=X_train_processed.index
                )
                X_val_processed = pd.DataFrame(
                    scaler.transform(X_val_processed),
                    columns=X_val_processed.columns,
                    index=X_val_processed.index
                )
            
            # Create model with parameters
            model = clone(base_model)
            if hasattr(model, 'set_params'):
                model.set_params(**params['model_params'])
            
            # Apply mitigation method
            method = params['mitigation_method']
            
            if method == 'baseline':
                model.fit(X_train_processed, y_train)
            
            elif method == 'reweight':
                # Assume first protected attribute for simplicity
                protected_attr = self.protected_attributes[0]
                if protected_attr in X_train.columns:
                    weights = reweight_samples(y_train, X_train[protected_attr])
                    model.fit(X_train_processed, y_train, sample_weight=weights)
                else:
                    model.fit(X_train_processed, y_train)
            
            elif method == 'postprocess':
                model.fit(X_train_processed, y_train)
                protected_attr = self.protected_attributes[0]
                if protected_attr in X_train.columns:
                    model = postprocess_equalized_odds(
                        model, X_train_processed, y_train, X_train[protected_attr]
                    )
            
            elif method == 'expgrad':
                protected_attr = self.protected_attributes[0]
                if protected_attr in X_train.columns:
                    model = expgrad_demographic_parity(
                        X_train_processed, y_train, X_train[protected_attr]
                    )
                else:
                    model.fit(X_train_processed, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_val_processed)
            accuracy = accuracy_score(y_val, y_pred)
            
            # Compute fairness violations
            fairness_violation = 0.0
            
            for protected_attr in self.protected_attributes:
                if protected_attr in X_val.columns:
                    try:
                        overall, _ = compute_fairness_metrics(
                            y_val, y_pred, X_val[protected_attr]
                        )
                        
                        # Aggregate fairness violations
                        for constraint, threshold in self.fairness_constraints.items():
                            if constraint in overall:
                                violation = max(0, abs(overall[constraint]) - threshold)
                                fairness_violation += violation
                    except Exception as e:
                        logger.warning(f"Fairness computation failed: {e}")
                        fairness_violation += 1.0  # Penalty for failure
            
            return model, fairness_violation, accuracy
            
        except Exception as e:
            logger.error(f"Individual evaluation failed: {e}")
            # Return dummy values with high penalty
            return base_model, 1.0, 0.0
    
    def _evolve_population(
        self,
        evaluated_population: List[Dict[str, Any]],
        search_space: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Evolve population for next generation."""
        # Sort by combined score
        evaluated_population.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Select top 50% as parents
        n_parents = len(evaluated_population) // 2
        parents = evaluated_population[:n_parents]
        
        # Generate new population
        new_population = []
        
        # Keep best individuals (elitism)
        n_elite = min(10, n_parents)
        for i in range(n_elite):
            new_population.append(parents[i]['params'])
        
        # Generate offspring through mutation
        while len(new_population) < self.population_size:
            parent = np.random.choice(parents)
            offspring = self._mutate_individual(parent['params'], search_space)
            new_population.append(offspring)
        
        return new_population
    
    def _mutate_individual(
        self,
        params: Dict[str, Any],
        search_space: Dict[str, Any],
        mutation_rate: float = 0.3
    ) -> Dict[str, Any]:
        """Mutate individual parameters."""
        mutated = params.copy()
        
        # Mutate with probability
        if np.random.random() < mutation_rate:
            # Randomly select parameter to mutate
            if np.random.random() < 0.5:
                # Mutate mitigation method
                mutated['mitigation_method'] = np.random.choice(search_space['mitigation_method'])
            else:
                # Mutate model parameters
                param_name = np.random.choice(list(search_space['model_params'].keys()))
                values = search_space['model_params'][param_name]
                
                if isinstance(values, tuple):
                    # Add Gaussian noise to continuous parameters
                    current_value = mutated['model_params'][param_name]
                    noise = np.random.normal(0, 0.1 * (values[1] - values[0]))
                    new_value = np.clip(current_value + noise, values[0], values[1])
                    mutated['model_params'][param_name] = new_value
                else:
                    # Random choice for discrete parameters
                    mutated['model_params'][param_name] = np.random.choice(values)
        
        return mutated
    
    def _generate_param_combinations(self, search_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for grid search."""
        # Simplified grid search for discrete parameters
        combinations = []
        
        for method in search_space['mitigation_method']:
            # Sample model parameters
            model_param_grid = []
            for param, values in search_space['model_params'].items():
                if isinstance(values, tuple):
                    # Sample a few values for continuous parameters
                    param_values = np.linspace(values[0], values[1], 3)
                else:
                    param_values = values
                model_param_grid.append(param_values)
            
            # Generate combinations
            for model_params in itertools.product(*model_param_grid):
                model_param_dict = dict(zip(search_space['model_params'].keys(), model_params))
                
                for preprocessing_combo in itertools.product(*search_space['preprocessing'].values()):
                    preprocessing_dict = dict(zip(search_space['preprocessing'].keys(), preprocessing_combo))
                    
                    combination = {
                        'mitigation_method': method,
                        'model_params': model_param_dict,
                        'preprocessing': preprocessing_dict
                    }
                    
                    combinations.append(combination)
        
        return combinations
    
    def _compute_final_metrics(
        self,
        model: BaseEstimator,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Compute final fairness and performance metrics."""
        try:
            y_pred = model.predict(X_val)
            
            # Performance metrics
            performance_metrics = {
                'accuracy': float(accuracy_score(y_val, y_pred)),
                'precision': float(precision_score(y_val, y_pred, average='binary')),
                'recall': float(recall_score(y_val, y_pred, average='binary'))
            }
            
            # Fairness metrics
            fairness_metrics = {}
            
            for protected_attr in self.protected_attributes:
                if protected_attr in X_val.columns:
                    try:
                        overall, _ = compute_fairness_metrics(
                            y_val, y_pred, X_val[protected_attr]
                        )
                        
                        fairness_metrics.update({
                            f"{protected_attr}_demographic_parity_difference": float(overall["demographic_parity_difference"]),
                            f"{protected_attr}_equalized_odds_difference": float(overall["equalized_odds_difference"])
                        })
                    except Exception as e:
                        logger.warning(f"Fairness computation failed for {protected_attr}: {e}")
            
            return fairness_metrics, performance_metrics
            
        except Exception as e:
            logger.error(f"Final metrics computation failed: {e}")
            return {}, {}


class HyperparameterTuner:
    """
    Advanced hyperparameter tuning with fairness considerations.
    
    Extends traditional hyperparameter tuning to include fairness
    metrics in the optimization objective.
    """
    
    def __init__(
        self,
        param_space: Dict[str, Any],
        scoring_function: Optional[Callable] = None,
        cv_folds: int = 5,
        n_iterations: int = 50,
        random_state: int = 42
    ):
        """
        Initialize hyperparameter tuner.
        
        Args:
            param_space: Parameter space definition
            scoring_function: Custom scoring function
            cv_folds: Number of cross-validation folds
            n_iterations: Number of optimization iterations
            random_state: Random seed
        """
        self.param_space = param_space
        self.scoring_function = scoring_function
        self.cv_folds = cv_folds
        self.n_iterations = n_iterations
        self.random_state = random_state
        
        # Tuning history
        self.tuning_history: List[Dict[str, Any]] = []
        
        logger.info("HyperparameterTuner initialized")
    
    def tune(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        protected_attributes: List[str]
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters with fairness considerations.
        
        Args:
            model: Model to tune
            X: Feature matrix
            y: Target variable
            protected_attributes: Protected attribute names
            
        Returns:
            Tuning results with best parameters
        """
        logger.info("Starting hyperparameter tuning")
        
        best_score = -np.inf
        best_params = None
        
        # Cross-validation setup
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        for iteration in range(self.n_iterations):
            # Sample parameters
            params = self._sample_parameters()
            
            try:
                # Cross-validation evaluation
                scores = []
                
                for train_idx, val_idx in cv.split(X, y):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # Train model with sampled parameters
                    model_clone = clone(model)
                    model_clone.set_params(**params)
                    model_clone.fit(X_train, y_train)
                    
                    # Evaluate
                    if self.scoring_function:
                        score = self.scoring_function(model_clone, X_val, y_val, protected_attributes)
                    else:
                        score = self._default_scoring(model_clone, X_val, y_val, protected_attributes)
                    
                    scores.append(score)
                
                # Average score across folds
                avg_score = np.mean(scores)
                
                # Track history
                result = {
                    'iteration': iteration,
                    'params': params,
                    'score': avg_score,
                    'std': np.std(scores)
                }
                self.tuning_history.append(result)
                
                # Update best
                if avg_score > best_score:
                    best_score = avg_score
                    best_params = params
                
                if iteration % 10 == 0:
                    logger.info(f"Iteration {iteration}: best score = {best_score:.4f}")
                    
            except Exception as e:
                logger.warning(f"Parameter evaluation failed: {e}")
                continue
        
        logger.info(f"Hyperparameter tuning completed: best score = {best_score:.4f}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'tuning_history': self.tuning_history,
            'n_iterations': len(self.tuning_history)
        }
    
    def _sample_parameters(self) -> Dict[str, Any]:
        """Sample parameters from parameter space."""
        params = {}
        
        for param_name, param_config in self.param_space.items():
            if isinstance(param_config, dict):
                if 'type' in param_config:
                    if param_config['type'] == 'uniform':
                        params[param_name] = np.random.uniform(
                            param_config['low'], param_config['high']
                        )
                    elif param_config['type'] == 'loguniform':
                        params[param_name] = np.random.lognormal(
                            np.log(param_config['low']),
                            np.log(param_config['high'])
                        )
                    elif param_config['type'] == 'choice':
                        params[param_name] = np.random.choice(param_config['choices'])
                else:
                    # Assume it's a range
                    if 'low' in param_config and 'high' in param_config:
                        params[param_name] = np.random.uniform(
                            param_config['low'], param_config['high']
                        )
            elif isinstance(param_config, list):
                params[param_name] = np.random.choice(param_config)
            elif isinstance(param_config, tuple):
                params[param_name] = np.random.uniform(param_config[0], param_config[1])
        
        return params
    
    def _default_scoring(
        self,
        model: BaseEstimator,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        protected_attributes: List[str]
    ) -> float:
        """Default scoring function balancing accuracy and fairness."""
        try:
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            
            # Compute fairness penalty
            fairness_penalty = 0.0
            
            for protected_attr in protected_attributes:
                if protected_attr in X_val.columns:
                    try:
                        overall, _ = compute_fairness_metrics(
                            y_val, y_pred, X_val[protected_attr]
                        )
                        
                        # Penalize high bias
                        dp_penalty = abs(overall.get('demographic_parity_difference', 0))
                        eo_penalty = abs(overall.get('equalized_odds_difference', 0))
                        fairness_penalty += (dp_penalty + eo_penalty) / 2
                        
                    except Exception:
                        fairness_penalty += 0.5  # Penalty for computation failure
            
            # Balance accuracy and fairness (70% accuracy, 30% fairness)
            score = 0.7 * accuracy - 0.3 * fairness_penalty
            
            return score
            
        except Exception as e:
            logger.error(f"Scoring function failed: {e}")
            return -1.0  # Low score for failures


# CLI interface
def main():
    """CLI interface for optimization testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fairness Optimization CLI")
    parser.add_argument("command", choices=["optimize", "tune"])
    parser.add_argument("--data", required=True, help="Data file path")
    parser.add_argument("--target", default="target", help="Target column name")
    parser.add_argument("--protected", nargs="+", default=["protected"], help="Protected attribute names")
    parser.add_argument("--iterations", type=int, default=50, help="Number of iterations")
    
    args = parser.parse_args()
    
    # Load data
    data = pd.read_csv(args.data)
    X = data.drop(columns=[args.target])
    y = data[args.target]
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    
    if args.command == "optimize":
        # Test fairness optimization
        from sklearn.linear_model import LogisticRegression
        
        optimizer = AutoFairnessOptimizer(
            protected_attributes=args.protected,
            max_iterations=args.iterations
        )
        
        base_model = LogisticRegression(random_state=42)
        
        result = optimizer.optimize(X_train, y_train, X_val, y_val, base_model)
        
        print("Optimization Results:")
        print(f"  Best score: {result.best_score:.4f}")
        print(f"  Best params: {result.best_params}")
        print(f"  Fairness metrics: {result.fairness_metrics}")
        print(f"  Performance metrics: {result.performance_metrics}")
    
    elif args.command == "tune":
        # Test hyperparameter tuning
        param_space = {
            'C': {'type': 'loguniform', 'low': 0.001, 'high': 100},
            'max_iter': {'type': 'choice', 'choices': [100, 500, 1000]},
            'solver': {'type': 'choice', 'choices': ['liblinear', 'lbfgs']}
        }
        
        tuner = HyperparameterTuner(
            param_space=param_space,
            n_iterations=args.iterations
        )
        
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42)
        
        result = tuner.tune(model, X, y, args.protected)
        
        print("Hyperparameter Tuning Results:")
        print(f"  Best score: {result['best_score']:.4f}")
        print(f"  Best params: {result['best_params']}")
        print(f"  Iterations: {result['n_iterations']}")


if __name__ == "__main__":
    main()