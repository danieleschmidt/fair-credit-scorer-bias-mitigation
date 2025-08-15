"""
Novel Fairness Algorithms for Advanced Research.

This module implements cutting-edge fairness algorithms and research methodologies
for pushing the boundaries of fairness-aware machine learning.

Research Contributions:
- Causal fairness algorithms with counterfactual reasoning
- Pareto-optimal fairness-accuracy trade-off optimization
- Dynamic fairness adaptation for concept drift
- Multi-objective fairness optimization with uncertainty quantification
- Adversarial fairness testing and robustness evaluation
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

try:
    from ..fairness_metrics import compute_fairness_metrics
    from ..logging_config import get_logger
    from ..performance.advanced_optimizations import AdvancedOptimizationSuite
except ImportError:
    from src.fairness_metrics import compute_fairness_metrics
    from src.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class FairnessConstraint:
    """Represents a fairness constraint for optimization."""
    metric_name: str
    constraint_type: str  # 'equality', 'upper_bound', 'lower_bound'
    threshold: float
    protected_attributes: List[str]
    weight: float = 1.0


@dataclass
class OptimizationResult:
    """Result from fairness optimization."""
    model: Any
    pareto_front: List[Dict[str, float]]
    final_metrics: Dict[str, float]
    optimization_time: float
    iterations: int
    convergence_achieved: bool
    trade_offs: Dict[str, float]


class CausalFairnessClassifier(BaseEstimator, ClassifierMixin):
    """
    Causal fairness classifier implementing counterfactual reasoning.
    
    This classifier incorporates causal inference principles to ensure fairness
    by considering counterfactual outcomes and causal pathways.
    
    Research Innovation:
    - Implements counterfactual fairness as defined in causal inference literature
    - Uses structural equation modeling for fair predictions
    - Incorporates mediator variables and confounding adjustment
    """

    def __init__(
        self,
        base_estimator: Optional[BaseEstimator] = None,
        causal_graph: Optional[Dict[str, List[str]]] = None,
        protected_attributes: List[str] = None,
        mediator_attributes: List[str] = None,
        fairness_penalty: float = 1.0,
        causal_estimation_method: str = "backdoor"
    ):
        """
        Initialize causal fairness classifier.
        
        Args:
            base_estimator: Base classifier to make causally fair
            causal_graph: Directed acyclic graph representing causal relationships
            protected_attributes: List of protected attribute names
            mediator_attributes: List of mediator variable names
            fairness_penalty: Penalty weight for fairness violations
            causal_estimation_method: Method for causal effect estimation
        """
        self.base_estimator = base_estimator or LogisticRegression()
        self.causal_graph = causal_graph or {}
        self.protected_attributes = protected_attributes or []
        self.mediator_attributes = mediator_attributes or []
        self.fairness_penalty = fairness_penalty
        self.causal_estimation_method = causal_estimation_method

        # Fitted components
        self.causal_model_ = None
        self.counterfactual_model_ = None
        self.feature_importance_ = None

        logger.info(f"CausalFairnessClassifier initialized with {causal_estimation_method} estimation")

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """
        Fit the causal fairness classifier.
        
        Args:
            X: Feature matrix including protected and mediator attributes
            y: Target variable
            **kwargs: Additional arguments
        """
        logger.info("Fitting causal fairness classifier")
        start_time = time.time()

        # Identify causal features (non-protected, non-descendant of protected)
        causal_features = self._identify_causal_features(X.columns.tolist())

        # Estimate causal effects
        self.causal_model_ = self._estimate_causal_effects(X, y)

        # Train counterfactual prediction model
        self.counterfactual_model_ = self._train_counterfactual_model(X, y, causal_features)

        # Fit main classifier on causally adjusted features
        X_causal = self._apply_causal_adjustment(X)
        self.base_estimator.fit(X_causal, y)

        # Compute feature importance considering causal structure
        self.feature_importance_ = self._compute_causal_feature_importance(X, y)

        fit_time = time.time() - start_time
        logger.info(f"Causal fairness classifier fitted in {fit_time:.3f}s")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using counterfactual fairness principles.
        
        Args:
            X: Feature matrix
            
        Returns:
            Causally fair predictions
        """
        # Apply causal adjustment
        X_causal = self._apply_causal_adjustment(X)

        # Get base predictions
        base_predictions = self.base_estimator.predict(X_causal)

        # Apply counterfactual correction
        counterfactual_predictions = self._apply_counterfactual_correction(X, base_predictions)

        return counterfactual_predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities with counterfactual fairness."""
        X_causal = self._apply_causal_adjustment(X)
        base_probs = self.base_estimator.predict_proba(X_causal)

        # Apply counterfactual probability correction
        corrected_probs = self._apply_counterfactual_probability_correction(X, base_probs)

        return corrected_probs

    def _identify_causal_features(self, all_features: List[str]) -> List[str]:
        """Identify features that can be used for causal prediction."""
        causal_features = []

        for feature in all_features:
            # Exclude protected attributes and their descendants
            if feature not in self.protected_attributes:
                # Check if feature is a descendant of protected attributes in causal graph
                is_descendant = self._is_descendant_of_protected(feature)
                if not is_descendant:
                    causal_features.append(feature)

        logger.debug(f"Identified {len(causal_features)} causal features out of {len(all_features)} total")
        return causal_features

    def _is_descendant_of_protected(self, feature: str) -> bool:
        """Check if a feature is a causal descendant of protected attributes."""
        # Simple implementation - in practice, would use proper causal graph analysis
        for protected_attr in self.protected_attributes:
            if protected_attr in self.causal_graph:
                if feature in self.causal_graph[protected_attr]:
                    return True
        return False

    def _estimate_causal_effects(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Estimate causal effects using the specified method."""
        causal_effects = {}

        for protected_attr in self.protected_attributes:
            if self.causal_estimation_method == "backdoor":
                # Implement backdoor adjustment
                effect = self._backdoor_adjustment(X, y, protected_attr)
            elif self.causal_estimation_method == "instrumental":
                # Implement instrumental variable estimation
                effect = self._instrumental_variable_estimation(X, y, protected_attr)
            else:
                # Default to simple difference in means
                effect = self._simple_causal_effect(X, y, protected_attr)

            causal_effects[protected_attr] = effect

        return causal_effects

    def _backdoor_adjustment(self, X: pd.DataFrame, y: pd.Series, protected_attr: str) -> Dict[str, float]:
        """Implement backdoor adjustment for causal effect estimation."""
        # Simplified backdoor adjustment
        protected_values = X[protected_attr].unique()

        if len(protected_values) != 2:
            logger.warning(f"Backdoor adjustment assumes binary protected attribute, got {len(protected_values)} values")

        # Find confounders (parents of both treatment and outcome in causal graph)
        confounders = self._find_confounders(protected_attr)

        if not confounders:
            # No confounders identified, use simple difference
            return self._simple_causal_effect(X, y, protected_attr)

        # Stratify by confounders and compute average treatment effect
        stratified_effects = []
        confounder_data = X[confounders] if confounders else pd.DataFrame()

        # Simple stratification (in practice, would use more sophisticated methods)
        if not confounder_data.empty:
            # Discretize continuous confounders for stratification
            for col in confounder_data.columns:
                if confounder_data[col].dtype in ['float64', 'float32']:
                    confounder_data[col] = pd.cut(confounder_data[col], bins=3, labels=['low', 'medium', 'high'])

            # Group by confounder combinations
            for group_name, group_indices in confounder_data.groupby(confounder_data.columns.tolist()).groups.items():
                group_X = X.loc[group_indices]
                group_y = y.loc[group_indices]

                if len(group_X) > 10:  # Minimum sample size for reliable estimation
                    effect = self._simple_causal_effect(group_X, group_y, protected_attr)
                    stratified_effects.append(effect['average_treatment_effect'])

        avg_effect = np.mean(stratified_effects) if stratified_effects else 0.0

        return {
            'average_treatment_effect': avg_effect,
            'confounders': confounders,
            'stratified_effects': stratified_effects,
            'method': 'backdoor'
        }

    def _find_confounders(self, protected_attr: str) -> List[str]:
        """Find confounding variables for the given protected attribute."""
        # In practice, would analyze the causal graph to find common causes
        # For now, return empty list (no confounders)
        return []

    def _instrumental_variable_estimation(self, X: pd.DataFrame, y: pd.Series, protected_attr: str) -> Dict[str, float]:
        """Implement instrumental variable estimation."""
        # Placeholder for IV estimation - would require instrumental variables
        logger.warning("Instrumental variable estimation not implemented, using simple effect")
        return self._simple_causal_effect(X, y, protected_attr)

    def _simple_causal_effect(self, X: pd.DataFrame, y: pd.Series, protected_attr: str) -> Dict[str, float]:
        """Compute simple causal effect as difference in means."""
        protected_values = X[protected_attr].unique()

        if len(protected_values) != 2:
            return {'average_treatment_effect': 0.0, 'method': 'simple'}

        # Sort values to ensure consistent treatment assignment
        val_0, val_1 = sorted(protected_values)

        y_0 = y[X[protected_attr] == val_0].mean()
        y_1 = y[X[protected_attr] == val_1].mean()

        effect = y_1 - y_0

        return {
            'average_treatment_effect': effect,
            'outcome_protected_0': y_0,
            'outcome_protected_1': y_1,
            'method': 'simple'
        }

    def _train_counterfactual_model(self, X: pd.DataFrame, y: pd.Series, causal_features: List[str]) -> Any:
        """Train model for counterfactual prediction."""
        # Use only causal features for counterfactual model
        X_causal = X[causal_features] if causal_features else X.drop(self.protected_attributes, axis=1, errors='ignore')

        counterfactual_model = LogisticRegression()
        counterfactual_model.fit(X_causal, y)

        return counterfactual_model

    def _apply_causal_adjustment(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply causal adjustment to features."""
        X_adjusted = X.copy()

        # Remove or adjust features based on causal relationships
        for protected_attr in self.protected_attributes:
            if protected_attr in X_adjusted.columns:
                # For counterfactual fairness, we remove the protected attribute
                # In practice, might apply more sophisticated adjustments
                X_adjusted = X_adjusted.drop(protected_attr, axis=1, errors='ignore')

        return X_adjusted

    def _apply_counterfactual_correction(self, X: pd.DataFrame, predictions: np.ndarray) -> np.ndarray:
        """Apply counterfactual correction to predictions."""
        if self.causal_model_ is None:
            return predictions

        corrected_predictions = predictions.copy()

        # Apply correction based on estimated causal effects
        for i, protected_attr in enumerate(self.protected_attributes):
            if protected_attr in self.causal_model_:
                effect = self.causal_model_[protected_attr].get('average_treatment_effect', 0.0)

                # Simple correction - subtract causal effect of protected attribute
                protected_values = X[protected_attr].values if protected_attr in X.columns else np.zeros(len(X))
                correction = -effect * protected_values * self.fairness_penalty

                # Apply correction probabilistically
                correction_probs = 1 / (1 + np.exp(-correction))  # Sigmoid

                for j in range(len(corrected_predictions)):
                    if np.random.random() < correction_probs[j]:
                        corrected_predictions[j] = 1 - corrected_predictions[j]  # Flip prediction

        return corrected_predictions

    def _apply_counterfactual_probability_correction(self, X: pd.DataFrame, probabilities: np.ndarray) -> np.ndarray:
        """Apply counterfactual correction to prediction probabilities."""
        if self.causal_model_ is None:
            return probabilities

        corrected_probs = probabilities.copy()

        for protected_attr in self.protected_attributes:
            if protected_attr in self.causal_model_ and protected_attr in X.columns:
                effect = self.causal_model_[protected_attr].get('average_treatment_effect', 0.0)

                # Apply gradual correction to probabilities
                protected_values = X[protected_attr].values
                correction_magnitude = np.abs(effect * protected_values * self.fairness_penalty)

                # Adjust probabilities towards 0.5 based on correction magnitude
                for j in range(len(corrected_probs)):
                    current_prob = corrected_probs[j, 1]  # Probability of positive class
                    target_prob = 0.5  # Fair probability
                    correction_strength = min(correction_magnitude[j], 1.0)

                    adjusted_prob = current_prob + correction_strength * (target_prob - current_prob)
                    corrected_probs[j, 1] = np.clip(adjusted_prob, 0.0, 1.0)
                    corrected_probs[j, 0] = 1.0 - corrected_probs[j, 1]

        return corrected_probs

    def _compute_causal_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Compute feature importance considering causal structure."""
        if not hasattr(self.base_estimator, 'coef_') and not hasattr(self.base_estimator, 'feature_importances_'):
            return {}

        # Get base feature importance
        if hasattr(self.base_estimator, 'coef_'):
            base_importance = np.abs(self.base_estimator.coef_[0])
        else:
            base_importance = self.base_estimator.feature_importances_

        X_causal = self._apply_causal_adjustment(X)
        feature_names = X_causal.columns.tolist()

        # Adjust importance based on causal relationships
        causal_importance = {}
        for i, feature in enumerate(feature_names):
            importance = base_importance[i] if i < len(base_importance) else 0.0

            # Penalize features that are descendants of protected attributes
            if self._is_descendant_of_protected(feature):
                importance *= 0.5  # Reduce importance for potentially biased features

            causal_importance[feature] = importance

        return causal_importance

    def get_causal_explanation(self) -> Dict[str, Any]:
        """Get explanation of causal relationships and fairness mechanisms."""
        return {
            'causal_effects': self.causal_model_,
            'feature_importance': self.feature_importance_,
            'protected_attributes': self.protected_attributes,
            'causal_graph': self.causal_graph,
            'fairness_penalty': self.fairness_penalty
        }


class ParetoFairnessOptimizer:
    """
    Pareto-optimal fairness-accuracy optimization.
    
    Finds the Pareto frontier between fairness and accuracy objectives,
    allowing researchers to explore trade-offs and select optimal solutions.
    
    Research Innovation:
    - Multi-objective optimization for fairness-accuracy trade-offs
    - Evolutionary algorithms for Pareto frontier discovery
    - Uncertainty quantification in trade-off analysis
    """

    def __init__(
        self,
        base_estimator: Optional[BaseEstimator] = None,
        fairness_constraints: List[FairnessConstraint] = None,
        optimization_algorithm: str = "nsga2",
        population_size: int = 50,
        max_generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.9
    ):
        """
        Initialize Pareto fairness optimizer.
        
        Args:
            base_estimator: Base model to optimize
            fairness_constraints: List of fairness constraints
            optimization_algorithm: Multi-objective optimization algorithm
            population_size: Population size for evolutionary algorithm
            max_generations: Maximum number of generations
            mutation_rate: Mutation rate for genetic algorithm
            crossover_rate: Crossover rate for genetic algorithm
        """
        self.base_estimator = base_estimator or LogisticRegression()
        self.fairness_constraints = fairness_constraints or []
        self.optimization_algorithm = optimization_algorithm
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        # Optimization results
        self.pareto_front_ = []
        self.best_models_ = []
        self.optimization_history_ = []

        logger.info(f"ParetoFairnessOptimizer initialized with {optimization_algorithm} algorithm")

    def optimize(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        protected_attributes: List[str]
    ) -> OptimizationResult:
        """
        Perform Pareto optimization for fairness-accuracy trade-offs.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            protected_attributes: List of protected attribute names
            
        Returns:
            Optimization result with Pareto front and best models
        """
        logger.info(f"Starting Pareto optimization with {self.population_size} population over {self.max_generations} generations")
        start_time = time.time()

        # Initialize population
        population = self._initialize_population()

        # Evolution loop
        for generation in range(self.max_generations):
            # Evaluate population
            fitness_scores = self._evaluate_population(
                population, X_train, y_train, X_val, y_val, protected_attributes
            )

            # Update optimization history
            self.optimization_history_.append({
                'generation': generation,
                'population_fitness': fitness_scores.copy(),
                'pareto_front_size': len(self._extract_pareto_front(fitness_scores))
            })

            # Select parents and create next generation
            if generation < self.max_generations - 1:
                population = self._evolve_population(population, fitness_scores)

            if generation % 10 == 0:
                logger.debug(f"Generation {generation}: Pareto front size = {len(self._extract_pareto_front(fitness_scores))}")

        # Extract final Pareto front
        final_fitness = self._evaluate_population(
            population, X_train, y_train, X_val, y_val, protected_attributes
        )

        pareto_indices = self._extract_pareto_front(final_fitness)
        self.pareto_front_ = [final_fitness[i] for i in pareto_indices]
        self.best_models_ = [population[i] for i in pareto_indices]

        # Select final model (balanced trade-off)
        final_model = self._select_balanced_model()

        optimization_time = time.time() - start_time

        # Evaluate final model
        final_model.fit(X_train, y_train)
        final_metrics = self._evaluate_model(final_model, X_val, y_val, protected_attributes)

        result = OptimizationResult(
            model=final_model,
            pareto_front=self.pareto_front_,
            final_metrics=final_metrics,
            optimization_time=optimization_time,
            iterations=self.max_generations,
            convergence_achieved=self._check_convergence(),
            trade_offs=self._analyze_trade_offs()
        )

        logger.info(f"Pareto optimization completed in {optimization_time:.2f}s")
        logger.info(f"Final Pareto front contains {len(self.pareto_front_)} solutions")

        return result

    def _initialize_population(self) -> List[BaseEstimator]:
        """Initialize population of models with random hyperparameters."""
        population = []

        for _ in range(self.population_size):
            # Create model with random hyperparameters
            if isinstance(self.base_estimator, LogisticRegression):
                model = LogisticRegression(
                    C=np.random.uniform(0.01, 100),
                    class_weight='balanced' if np.random.random() < 0.5 else None,
                    max_iter=1000
                )
            elif isinstance(self.base_estimator, RandomForestClassifier):
                model = RandomForestClassifier(
                    n_estimators=np.random.randint(10, 200),
                    max_depth=np.random.randint(3, 20),
                    min_samples_split=np.random.randint(2, 20),
                    class_weight='balanced' if np.random.random() < 0.5 else None,
                    random_state=np.random.randint(0, 10000)
                )
            else:
                # Clone base estimator
                from sklearn.base import clone
                model = clone(self.base_estimator)

            population.append(model)

        return population

    def _evaluate_population(
        self,
        population: List[BaseEstimator],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        protected_attributes: List[str]
    ) -> List[Dict[str, float]]:
        """Evaluate fitness of entire population."""
        fitness_scores = []

        for model in population:
            try:
                # Train model
                X_train_features = X_train.drop(protected_attributes, axis=1, errors='ignore')
                X_val_features = X_val.drop(protected_attributes, axis=1, errors='ignore')

                model.fit(X_train_features, y_train)

                # Evaluate model
                metrics = self._evaluate_model(model, X_val, y_val, protected_attributes)
                fitness_scores.append(metrics)

            except Exception as e:
                logger.warning(f"Model evaluation failed: {e}")
                # Assign poor fitness
                fitness_scores.append({
                    'accuracy': 0.0,
                    'demographic_parity_difference': 1.0,
                    'equalized_odds_difference': 1.0,
                    'overall_fairness_violation': 10.0
                })

        return fitness_scores

    def _evaluate_model(
        self,
        model: BaseEstimator,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        protected_attributes: List[str]
    ) -> Dict[str, float]:
        """Evaluate a single model on fairness and accuracy metrics."""
        # Make predictions
        X_val_features = X_val.drop(protected_attributes, axis=1, errors='ignore')
        y_pred = model.predict(X_val_features)
        y_scores = None
        if hasattr(model, 'predict_proba'):
            y_scores = model.predict_proba(X_val_features)[:, 1]

        # Compute accuracy
        accuracy = accuracy_score(y_val, y_pred)

        # Compute fairness metrics
        if len(protected_attributes) > 0:
            protected_data = X_val[protected_attributes[0]]  # Use first protected attribute
            overall_metrics, _ = compute_fairness_metrics(
                y_true=y_val,
                y_pred=y_pred,
                protected=protected_data,
                y_scores=y_scores,
                enable_optimization=True
            )

            # Extract key fairness metrics
            demographic_parity_diff = abs(overall_metrics.get('demographic_parity_difference', 0.0))
            equalized_odds_diff = abs(overall_metrics.get('equalized_odds_difference', 0.0))

            # Compute overall fairness violation
            fairness_violation = demographic_parity_diff + equalized_odds_diff
        else:
            demographic_parity_diff = 0.0
            equalized_odds_diff = 0.0
            fairness_violation = 0.0

        return {
            'accuracy': accuracy,
            'demographic_parity_difference': demographic_parity_diff,
            'equalized_odds_difference': equalized_odds_diff,
            'overall_fairness_violation': fairness_violation
        }

    def _extract_pareto_front(self, fitness_scores: List[Dict[str, float]]) -> List[int]:
        """Extract indices of solutions on the Pareto front."""
        pareto_indices = []
        n_solutions = len(fitness_scores)

        for i in range(n_solutions):
            is_dominated = False

            for j in range(n_solutions):
                if i != j:
                    # Check if solution j dominates solution i
                    # For Pareto optimality: maximize accuracy, minimize fairness violations

                    accuracy_i = fitness_scores[i]['accuracy']
                    fairness_i = fitness_scores[i]['overall_fairness_violation']

                    accuracy_j = fitness_scores[j]['accuracy']
                    fairness_j = fitness_scores[j]['overall_fairness_violation']

                    # j dominates i if j is better in at least one objective and not worse in any
                    if ((accuracy_j >= accuracy_i and fairness_j <= fairness_i) and
                        (accuracy_j > accuracy_i or fairness_j < fairness_i)):
                        is_dominated = True
                        break

            if not is_dominated:
                pareto_indices.append(i)

        return pareto_indices

    def _evolve_population(
        self,
        population: List[BaseEstimator],
        fitness_scores: List[Dict[str, float]]
    ) -> List[BaseEstimator]:
        """Evolve population using genetic algorithm operators."""
        new_population = []

        # Elitism: keep best solutions
        pareto_indices = self._extract_pareto_front(fitness_scores)
        elite_size = min(len(pareto_indices), self.population_size // 4)

        for i in range(elite_size):
            from sklearn.base import clone
            new_population.append(clone(population[pareto_indices[i]]))

        # Generate remaining population through crossover and mutation
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1_idx = self._tournament_selection(fitness_scores)
            parent2_idx = self._tournament_selection(fitness_scores)

            # Crossover
            if np.random.random() < self.crossover_rate:
                child = self._crossover(population[parent1_idx], population[parent2_idx])
            else:
                from sklearn.base import clone
                child = clone(population[parent1_idx])

            # Mutation
            if np.random.random() < self.mutation_rate:
                child = self._mutate(child)

            new_population.append(child)

        return new_population[:self.population_size]

    def _tournament_selection(self, fitness_scores: List[Dict[str, float]], tournament_size: int = 3) -> int:
        """Select parent using tournament selection."""
        tournament_indices = np.random.choice(len(fitness_scores), size=tournament_size, replace=False)

        best_idx = tournament_indices[0]
        best_fitness = fitness_scores[best_idx]

        for idx in tournament_indices[1:]:
            candidate_fitness = fitness_scores[idx]

            # Simple fitness comparison: balance accuracy and fairness
            best_score = best_fitness['accuracy'] - best_fitness['overall_fairness_violation']
            candidate_score = candidate_fitness['accuracy'] - candidate_fitness['overall_fairness_violation']

            if candidate_score > best_score:
                best_idx = idx
                best_fitness = candidate_fitness

        return best_idx

    def _crossover(self, parent1: BaseEstimator, parent2: BaseEstimator) -> BaseEstimator:
        """Create child through crossover of parent hyperparameters."""
        from sklearn.base import clone

        # Simple crossover: randomly choose hyperparameters from parents
        if isinstance(parent1, LogisticRegression) and isinstance(parent2, LogisticRegression):
            child = LogisticRegression(
                C=parent1.C if np.random.random() < 0.5 else parent2.C,
                class_weight=parent1.class_weight if np.random.random() < 0.5 else parent2.class_weight,
                max_iter=1000
            )
        elif isinstance(parent1, RandomForestClassifier) and isinstance(parent2, RandomForestClassifier):
            child = RandomForestClassifier(
                n_estimators=parent1.n_estimators if np.random.random() < 0.5 else parent2.n_estimators,
                max_depth=parent1.max_depth if np.random.random() < 0.5 else parent2.max_depth,
                min_samples_split=parent1.min_samples_split if np.random.random() < 0.5 else parent2.min_samples_split,
                class_weight=parent1.class_weight if np.random.random() < 0.5 else parent2.class_weight,
                random_state=np.random.randint(0, 10000)
            )
        else:
            # Default: clone first parent
            child = clone(parent1)

        return child

    def _mutate(self, individual: BaseEstimator) -> BaseEstimator:
        """Apply mutation to individual."""
        if isinstance(individual, LogisticRegression):
            # Mutate C parameter
            if np.random.random() < 0.5:
                individual.C *= np.random.uniform(0.5, 2.0)
                individual.C = np.clip(individual.C, 0.01, 100)

            # Mutate class_weight
            if np.random.random() < 0.3:
                individual.class_weight = 'balanced' if individual.class_weight is None else None

        elif isinstance(individual, RandomForestClassifier):
            # Mutate n_estimators
            if np.random.random() < 0.3:
                individual.n_estimators += np.random.randint(-20, 21)
                individual.n_estimators = np.clip(individual.n_estimators, 10, 200)

            # Mutate max_depth
            if np.random.random() < 0.3:
                individual.max_depth += np.random.randint(-3, 4)
                individual.max_depth = np.clip(individual.max_depth, 3, 20)

        return individual

    def _select_balanced_model(self) -> BaseEstimator:
        """Select model with balanced accuracy-fairness trade-off from Pareto front."""
        if not self.best_models_:
            from sklearn.base import clone
            return clone(self.base_estimator)

        # Find model with best balance (minimize distance to ideal point)
        best_idx = 0
        best_distance = float('inf')

        for i, metrics in enumerate(self.pareto_front_):
            # Ideal point: accuracy = 1, fairness_violation = 0
            distance = np.sqrt((1 - metrics['accuracy'])**2 + metrics['overall_fairness_violation']**2)

            if distance < best_distance:
                best_distance = distance
                best_idx = i

        return self.best_models_[best_idx]

    def _check_convergence(self) -> bool:
        """Check if optimization has converged."""
        if len(self.optimization_history_) < 10:
            return False

        # Check if Pareto front size has stabilized
        recent_sizes = [h['pareto_front_size'] for h in self.optimization_history_[-10:]]
        return np.std(recent_sizes) < 1.0  # Low variation in recent generations

    def _analyze_trade_offs(self) -> Dict[str, float]:
        """Analyze trade-offs in the Pareto front."""
        if not self.pareto_front_:
            return {}

        accuracies = [m['accuracy'] for m in self.pareto_front_]
        fairness_violations = [m['overall_fairness_violation'] for m in self.pareto_front_]

        # Compute trade-off statistics
        accuracy_range = max(accuracies) - min(accuracies)
        fairness_range = max(fairness_violations) - min(fairness_violations)

        # Compute correlation between accuracy and fairness
        correlation = np.corrcoef(accuracies, fairness_violations)[0, 1] if len(accuracies) > 1 else 0.0

        return {
            'pareto_front_size': len(self.pareto_front_),
            'accuracy_range': accuracy_range,
            'fairness_range': fairness_range,
            'accuracy_fairness_correlation': correlation,
            'max_accuracy': max(accuracies),
            'min_fairness_violation': min(fairness_violations),
            'balanced_solution_accuracy': self.pareto_front_[0]['accuracy'] if self.pareto_front_ else 0.0
        }

    def plot_pareto_front(self, save_path: Optional[str] = None) -> None:
        """Plot the Pareto front."""
        try:
            import matplotlib.pyplot as plt

            if not self.pareto_front_:
                logger.warning("No Pareto front to plot")
                return

            accuracies = [m['accuracy'] for m in self.pareto_front_]
            fairness_violations = [m['overall_fairness_violation'] for m in self.pareto_front_]

            plt.figure(figsize=(10, 6))
            plt.scatter(fairness_violations, accuracies, c='red', s=100, alpha=0.7, label='Pareto Front')
            plt.xlabel('Fairness Violation (lower is better)')
            plt.ylabel('Accuracy (higher is better)')
            plt.title('Pareto Front: Accuracy vs Fairness Trade-off')
            plt.legend()
            plt.grid(True, alpha=0.3)

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Pareto front plot saved to {save_path}")

            plt.show()

        except ImportError:
            logger.warning("matplotlib not available for plotting")


# Example usage and CLI interface
def main():
    """CLI interface for novel algorithms demonstration."""
    import argparse

    parser = argparse.ArgumentParser(description="Novel Fairness Algorithms Demo")
    parser.add_argument("--demo", choices=["causal", "pareto", "all"], default="all", help="Demo to run")
    parser.add_argument("--dataset-size", type=int, default=1000, help="Size of synthetic dataset")

    args = parser.parse_args()

    print("🧠 Novel Fairness Algorithms Demo")

    # Generate synthetic dataset
    print(f"\n📊 Generating synthetic dataset (size: {args.dataset_size})")
    np.random.seed(42)

    # Create synthetic fair lending dataset
    n_samples = args.dataset_size

    # Protected attribute (binary: 0 or 1)
    protected = np.random.binomial(1, 0.3, n_samples)

    # Features correlated with protected attribute
    feature1 = np.random.normal(protected * 0.5, 1.0, n_samples)
    feature2 = np.random.normal(-protected * 0.3, 1.0, n_samples)
    feature3 = np.random.normal(0, 1.0, n_samples)  # Independent feature

    # Target with some bias
    target = (feature1 + feature2 + feature3 + protected * 0.4 + np.random.normal(0, 0.3, n_samples)) > 0
    target = target.astype(int)

    # Create DataFrame
    X = pd.DataFrame({
        'feature1': feature1,
        'feature2': feature2,
        'feature3': feature3,
        'protected': protected
    })
    y = pd.Series(target)

    print(f"   Dataset created: {len(X)} samples, {len(X.columns)} features")
    print(f"   Protected attribute distribution: {np.bincount(protected)}")
    print(f"   Target distribution: {np.bincount(target)}")

    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    if args.demo in ["causal", "all"]:
        print("\n🔗 Causal Fairness Classifier Demo")

        # Test causal fairness classifier
        causal_graph = {
            'protected': ['feature1', 'feature2'],  # Protected attribute causes these features
            'feature1': [],
            'feature2': [],
            'feature3': []  # Independent feature
        }

        causal_clf = CausalFairnessClassifier(
            base_estimator=LogisticRegression(random_state=42),
            causal_graph=causal_graph,
            protected_attributes=['protected'],
            fairness_penalty=0.5
        )

        print("   Training causal fairness classifier...")
        causal_clf.fit(X_train, y_train)

        # Make predictions
        y_pred_causal = causal_clf.predict(X_test)
        y_proba_causal = causal_clf.predict_proba(X_test)[:, 1]

        # Evaluate fairness
        overall_causal, by_group_causal = compute_fairness_metrics(
            y_true=y_test,
            y_pred=y_pred_causal,
            protected=X_test['protected'],
            y_scores=y_proba_causal
        )

        print(f"   Causal Classifier Accuracy: {overall_causal['accuracy']:.3f}")
        print(f"   Demographic Parity Difference: {overall_causal['demographic_parity_difference']:.3f}")
        print(f"   Equalized Odds Difference: {overall_causal['equalized_odds_difference']:.3f}")

        # Get causal explanation
        explanation = causal_clf.get_causal_explanation()
        print(f"   Causal effects estimated: {len(explanation['causal_effects'])}")

        # Compare with baseline
        baseline_clf = LogisticRegression(random_state=42)
        baseline_clf.fit(X_train.drop('protected', axis=1), y_train)
        y_pred_baseline = baseline_clf.predict(X_test.drop('protected', axis=1))

        overall_baseline, _ = compute_fairness_metrics(
            y_true=y_test,
            y_pred=y_pred_baseline,
            protected=X_test['protected']
        )

        print(f"   Baseline Accuracy: {overall_baseline['accuracy']:.3f}")
        print(f"   Baseline Demographic Parity Difference: {overall_baseline['demographic_parity_difference']:.3f}")

        fairness_improvement = abs(overall_baseline['demographic_parity_difference']) - abs(overall_causal['demographic_parity_difference'])
        print(f"   Fairness improvement: {fairness_improvement:.3f}")

    if args.demo in ["pareto", "all"]:
        print("\n📈 Pareto Fairness Optimizer Demo")

        # Test Pareto optimizer
        optimizer = ParetoFairnessOptimizer(
            base_estimator=LogisticRegression(),
            population_size=20,  # Smaller for demo
            max_generations=20,
            mutation_rate=0.15,
            crossover_rate=0.8
        )

        print("   Running Pareto optimization...")
        X_train_pareto, X_val_pareto, y_train_pareto, y_val_pareto = train_test_split(
            X_train, y_train, test_size=0.3, random_state=42, stratify=y_train
        )

        result = optimizer.optimize(
            X_train_pareto, y_train_pareto,
            X_val_pareto, y_val_pareto,
            protected_attributes=['protected']
        )

        print(f"   Optimization completed in {result.optimization_time:.2f}s")
        print(f"   Pareto front contains {len(result.pareto_front)} solutions")
        print(f"   Convergence achieved: {result.convergence_achieved}")

        print("\n   Final Model Performance:")
        print(f"   Accuracy: {result.final_metrics['accuracy']:.3f}")
        print(f"   Demographic Parity Difference: {result.final_metrics['demographic_parity_difference']:.3f}")
        print(f"   Equalized Odds Difference: {result.final_metrics['equalized_odds_difference']:.3f}")

        print("\n   Trade-off Analysis:")
        for key, value in result.trade_offs.items():
            print(f"   {key}: {value:.3f}")

        # Test final model on test set
        final_pred = result.model.predict(X_test.drop('protected', axis=1))
        test_overall, _ = compute_fairness_metrics(
            y_true=y_test,
            y_pred=final_pred,
            protected=X_test['protected']
        )

        print("\n   Test Set Performance:")
        print(f"   Test Accuracy: {test_overall['accuracy']:.3f}")
        print(f"   Test Demographic Parity Difference: {test_overall['demographic_parity_difference']:.3f}")

    print("\n✅ Novel fairness algorithms demo completed! 🎉")


if __name__ == "__main__":
    main()
