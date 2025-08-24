"""
Quantum-Inspired Fairness Algorithms - Cutting-Edge Research Implementation.

This module implements revolutionary quantum-inspired approaches to fairness-aware
machine learning, representing the next generation of bias mitigation techniques.

Research Contributions:
- Quantum superposition-inspired ensemble fairness optimization
- Entanglement-based feature correlation analysis for bias detection
- Quantum annealing-inspired hyperparameter optimization for fairness
- Quantum circuit-inspired neural network architectures for fair prediction
- Uncertainty quantification using quantum measurement principles
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

try:
    from ..fairness_metrics import compute_fairness_metrics
    from ..logging_config import get_logger
except ImportError:
    from fairness_metrics import compute_fairness_metrics
    from logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class QuantumFairnessResult:
    """Result from quantum fairness optimization."""
    optimized_model: BaseEstimator
    fairness_metrics: Dict[str, float]
    quantum_parameters: Dict[str, Any]
    optimization_history: List[Dict[str, float]]
    entanglement_scores: Dict[Tuple[str, str], float]
    uncertainty_quantification: Dict[str, float]
    convergence_achieved: bool
    optimization_time: float


class QuantumSuperpositionEnsemble(BaseEstimator, ClassifierMixin):
    """
    Quantum superposition-inspired ensemble for fairness optimization.

    This classifier uses principles from quantum superposition to maintain
    multiple model states simultaneously, allowing for optimal fairness-accuracy
    trade-offs through probabilistic model selection.

    Research Innovation:
    - Implements quantum superposition analogy for model ensembling
    - Dynamic amplitude adjustment based on fairness constraints
    - Quantum measurement-inspired prediction aggregation
    - Entanglement-based feature interaction analysis
    """

    def __init__(
        self,
        base_estimators: List[BaseEstimator] = None,
        protected_attributes: List[str] = None,
        fairness_weight: float = 0.5,
        quantum_temperature: float = 1.0,
        measurement_shots: int = 1000,
        entanglement_threshold: float = 0.1,
        enable_quantum_annealing: bool = True
    ):
        """
        Initialize quantum superposition ensemble.

        Args:
            base_estimators: List of base models for the superposition
            protected_attributes: List of protected attribute names
            fairness_weight: Weight for fairness vs accuracy optimization
            quantum_temperature: Temperature parameter for quantum state evolution
            measurement_shots: Number of measurement shots for prediction aggregation
            entanglement_threshold: Threshold for detecting feature entanglement
            enable_quantum_annealing: Enable quantum annealing optimization
        """
        self.base_estimators = base_estimators or [
            LogisticRegression(max_iter=1000),
            LogisticRegression(C=0.1, max_iter=1000),
            LogisticRegression(C=10, max_iter=1000)
        ]
        self.protected_attributes = protected_attributes or []
        self.fairness_weight = fairness_weight
        self.quantum_temperature = quantum_temperature
        self.measurement_shots = measurement_shots
        self.entanglement_threshold = entanglement_threshold
        self.enable_quantum_annealing = enable_quantum_annealing

        # Quantum state variables
        self.amplitude_weights_ = None
        self.quantum_states_ = []
        self.entanglement_matrix_ = None
        self.fitted_estimators_ = []
        self.quantum_circuit_depth_ = 0

        logger.info("QuantumSuperpositionEnsemble initialized with quantum-inspired fairness optimization")

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """
        Fit the quantum superposition ensemble.

        Implements quantum-inspired optimization where each base estimator
        represents a quantum state with associated amplitude weights.
        """
        logger.info("Fitting quantum superposition ensemble with fairness optimization")
        start_time = time.time()

        # Initialize quantum states
        n_estimators = len(self.base_estimators)
        self.amplitude_weights_ = np.ones(n_estimators) / np.sqrt(n_estimators)

        # Analyze feature entanglement
        self.entanglement_matrix_ = self._analyze_feature_entanglement(X)

        # Fit base estimators (quantum states)
        self.fitted_estimators_ = []
        quantum_state_performance = []

        # Prepare features for training (excluding protected attributes for base models)
        X_features = X.drop(self.protected_attributes, axis=1, errors='ignore')

        for i, estimator in enumerate(self.base_estimators):
            logger.debug(f"Fitting quantum state {i+1}/{n_estimators}")

            # Clone and fit estimator
            fitted_estimator = clone(estimator)
            fitted_estimator.fit(X_features, y)
            self.fitted_estimators_.append(fitted_estimator)

            # Evaluate quantum state performance
            predictions = fitted_estimator.predict(X_features)
            proba = fitted_estimator.predict_proba(X_features)[:, 1] if hasattr(fitted_estimator, 'predict_proba') else None

            # Compute fairness metrics for this quantum state
            if len(self.protected_attributes) > 0 and self.protected_attributes[0] in X.columns:
                overall_metrics, _ = compute_fairness_metrics(
                    y_true=y,
                    y_pred=predictions,
                    protected=X[self.protected_attributes[0]],
                    y_scores=proba,
                    enable_optimization=True
                )

                # Quantum state performance combines accuracy and fairness
                accuracy = overall_metrics.get('accuracy', 0.0)
                demographic_parity_diff = abs(overall_metrics.get('demographic_parity_difference', 0.0))

                # Quantum fitness function
                quantum_fitness = (1 - self.fairness_weight) * accuracy - self.fairness_weight * demographic_parity_diff
            else:
                accuracy = accuracy_score(y, predictions)
                quantum_fitness = accuracy

            quantum_state_performance.append({
                'accuracy': accuracy,
                'fairness_violation': demographic_parity_diff if len(self.protected_attributes) > 0 else 0.0,
                'quantum_fitness': quantum_fitness
            })

        # Optimize amplitude weights using quantum annealing principles
        if self.enable_quantum_annealing:
            self.amplitude_weights_ = self._quantum_annealing_optimization(quantum_state_performance)
        else:
            # Simple softmax-based weight assignment
            fitness_scores = [state['quantum_fitness'] for state in quantum_state_performance]
            self.amplitude_weights_ = self._softmax(np.array(fitness_scores) / self.quantum_temperature)

        # Normalize amplitude weights (quantum normalization constraint)
        self.amplitude_weights_ = self.amplitude_weights_ / np.sqrt(np.sum(self.amplitude_weights_**2))

        self.quantum_circuit_depth_ = len(self.fitted_estimators_)
        fit_time = time.time() - start_time

        logger.info(f"Quantum superposition ensemble fitted in {fit_time:.3f}s with {self.quantum_circuit_depth_} quantum states")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using quantum measurement principles.

        Implements quantum measurement where the final prediction is obtained
        through probabilistic sampling based on amplitude weights.
        """
        X_features = X.drop(self.protected_attributes, axis=1, errors='ignore')

        # Get predictions from all quantum states
        state_predictions = []
        for estimator in self.fitted_estimators_:
            predictions = estimator.predict(X_features)
            state_predictions.append(predictions)

        state_predictions = np.array(state_predictions).T  # Shape: (n_samples, n_estimators)

        # Quantum measurement-inspired prediction aggregation
        final_predictions = np.zeros(len(X))

        # Compute measurement probabilities (squared amplitudes)
        measurement_probabilities = self.amplitude_weights_**2
        measurement_probabilities = measurement_probabilities / np.sum(measurement_probabilities)

        for i in range(len(X)):
            # Quantum measurement simulation
            measurement_counts = np.zeros(2)  # For binary classification

            for _shot in range(self.measurement_shots):
                # Sample quantum state based on measurement probabilities
                chosen_state = np.random.choice(len(self.fitted_estimators_), p=measurement_probabilities)
                prediction = state_predictions[i, chosen_state]
                measurement_counts[int(prediction)] += 1

            # Final prediction based on measurement statistics
            final_predictions[i] = 1 if measurement_counts[1] > measurement_counts[0] else 0

        return final_predictions.astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities using quantum superposition."""
        X_features = X.drop(self.protected_attributes, axis=1, errors='ignore')

        # Get probabilities from all quantum states
        state_probabilities = []
        for estimator in self.fitted_estimators_:
            if hasattr(estimator, 'predict_proba'):
                proba = estimator.predict_proba(X_features)
            else:
                # Convert predictions to probabilities
                predictions = estimator.predict(X_features)
                proba = np.column_stack([1 - predictions, predictions])
            state_probabilities.append(proba)

        state_probabilities = np.array(state_probabilities)  # Shape: (n_estimators, n_samples, n_classes)

        # Quantum superposition of probability amplitudes
        measurement_probabilities = self.amplitude_weights_**2
        measurement_probabilities = measurement_probabilities / np.sum(measurement_probabilities)

        # Weighted average based on quantum measurement probabilities
        final_probabilities = np.zeros((len(X), 2))
        for i, prob_weight in enumerate(measurement_probabilities):
            final_probabilities += prob_weight * state_probabilities[i]

        return final_probabilities

    def _analyze_feature_entanglement(self, X: pd.DataFrame) -> np.ndarray:
        """
        Analyze feature entanglement using quantum-inspired correlation analysis.

        Computes a quantum-inspired entanglement matrix representing the degree
        of correlation between features, analogous to quantum entanglement.
        """
        logger.debug("Analyzing feature entanglement patterns")

        # Compute correlation matrix
        correlation_matrix = X.corr().abs().values

        # Apply quantum-inspired transformation
        # High correlations represent strong "entanglement"
        entanglement_matrix = correlation_matrix.copy()

        # Apply von Neumann entropy-inspired transformation
        for i in range(entanglement_matrix.shape[0]):
            for j in range(entanglement_matrix.shape[1]):
                if i != j:
                    correlation = correlation_matrix[i, j]
                    # Quantum entanglement measure (simplified)
                    if correlation > self.entanglement_threshold:
                        entanglement_matrix[i, j] = -correlation * np.log(correlation + 1e-8)
                    else:
                        entanglement_matrix[i, j] = 0.0

        return entanglement_matrix

    def _quantum_annealing_optimization(self, quantum_state_performance: List[Dict[str, float]]) -> np.ndarray:
        """
        Optimize amplitude weights using quantum annealing-inspired algorithm.

        Implements a simplified quantum annealing process to find optimal
        amplitude weights that balance fairness and accuracy.
        """
        logger.debug("Optimizing amplitude weights using quantum annealing")

        n_states = len(quantum_state_performance)

        # Initialize random amplitudes
        amplitudes = np.random.random(n_states)
        amplitudes = amplitudes / np.sqrt(np.sum(amplitudes**2))

        # Quantum annealing parameters
        initial_temperature = 1.0
        cooling_rate = 0.95
        max_iterations = 100

        current_temperature = initial_temperature
        best_amplitudes = amplitudes.copy()
        best_energy = self._compute_quantum_energy(amplitudes, quantum_state_performance)

        for iteration in range(max_iterations):
            # Generate neighbor solution
            new_amplitudes = amplitudes + np.random.normal(0, 0.1, n_states)
            new_amplitudes = np.abs(new_amplitudes)  # Ensure positive amplitudes
            new_amplitudes = new_amplitudes / np.sqrt(np.sum(new_amplitudes**2))  # Normalize

            # Compute energy difference
            new_energy = self._compute_quantum_energy(new_amplitudes, quantum_state_performance)
            energy_diff = new_energy - best_energy

            # Quantum acceptance probability (Boltzmann distribution)
            if energy_diff < 0 or np.random.random() < np.exp(-energy_diff / current_temperature):
                amplitudes = new_amplitudes
                if new_energy < best_energy:
                    best_amplitudes = new_amplitudes.copy()
                    best_energy = new_energy

            # Cool down temperature
            current_temperature *= cooling_rate

            if iteration % 20 == 0:
                logger.debug(f"Quantum annealing iteration {iteration}: energy = {best_energy:.4f}")

        logger.debug(f"Quantum annealing completed: final energy = {best_energy:.4f}")
        return best_amplitudes

    def _compute_quantum_energy(self, amplitudes: np.ndarray, performance_data: List[Dict[str, float]]) -> float:
        """
        Compute quantum energy function for optimization.

        The energy function represents the cost we want to minimize,
        combining fairness violations and accuracy losses.
        """
        # Ensure normalization constraint
        if abs(np.sum(amplitudes**2) - 1.0) > 1e-6:
            return float('inf')  # Violates quantum normalization

        total_energy = 0.0

        for _i, (amplitude, perf) in enumerate(zip(amplitudes, performance_data)):
            # Energy contribution from this quantum state
            state_energy = (
                self.fairness_weight * perf['fairness_violation'] +
                (1 - self.fairness_weight) * (1 - perf['accuracy'])
            )

            # Weight by squared amplitude (measurement probability)
            total_energy += (amplitude**2) * state_energy

        return total_energy

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax function for amplitude weight assignment."""
        exp_x = np.exp(x - np.max(x))  # Numerical stability
        return exp_x / np.sum(exp_x)

    def get_quantum_state_info(self) -> Dict[str, Any]:
        """Get information about the quantum state of the ensemble."""
        return {
            'amplitude_weights': self.amplitude_weights_.tolist() if self.amplitude_weights_ is not None else None,
            'quantum_circuit_depth': self.quantum_circuit_depth_,
            'measurement_shots': self.measurement_shots,
            'quantum_temperature': self.quantum_temperature,
            'entanglement_threshold': self.entanglement_threshold,
            'n_quantum_states': len(self.fitted_estimators_) if self.fitted_estimators_ else 0
        }


class QuantumFairnessOptimizer:
    """
    Quantum-inspired optimizer for fairness-aware machine learning.

    Uses quantum computing principles to optimize both fairness and accuracy
    simultaneously through quantum superposition and measurement techniques.
    """

    def __init__(
        self,
        base_estimator: Optional[BaseEstimator] = None,
        quantum_iterations: int = 50,
        superposition_depth: int = 5,
        entanglement_analysis: bool = True,
        uncertainty_quantification: bool = True
    ):
        """
        Initialize quantum fairness optimizer.

        Args:
            base_estimator: Base model to optimize
            quantum_iterations: Number of quantum optimization iterations
            superposition_depth: Number of models in quantum superposition
            entanglement_analysis: Enable feature entanglement analysis
            uncertainty_quantification: Enable quantum uncertainty quantification
        """
        self.base_estimator = base_estimator or LogisticRegression()
        self.quantum_iterations = quantum_iterations
        self.superposition_depth = superposition_depth
        self.entanglement_analysis = entanglement_analysis
        self.uncertainty_quantification = uncertainty_quantification

        logger.info("QuantumFairnessOptimizer initialized")

    def optimize(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        protected_attributes: List[str],
        fairness_constraints: List[str] = None
    ) -> QuantumFairnessResult:
        """
        Perform quantum-inspired fairness optimization.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            protected_attributes: List of protected attribute names
            fairness_constraints: List of fairness constraints to optimize

        Returns:
            Quantum fairness optimization results
        """
        logger.info(f"Starting quantum fairness optimization with {self.quantum_iterations} iterations")
        start_time = time.time()

        if fairness_constraints is None:
            fairness_constraints = ['demographic_parity', 'equalized_odds']

        # Create quantum superposition ensemble
        base_estimators = []
        for i in range(self.superposition_depth):
            # Create variations of base estimator for quantum superposition
            if hasattr(self.base_estimator, 'C'):
                # Logistic regression variants
                C_values = np.logspace(-3, 3, self.superposition_depth)
                estimator = clone(self.base_estimator)
                estimator.set_params(C=C_values[i])
            else:
                estimator = clone(self.base_estimator)

            base_estimators.append(estimator)

        # Initialize quantum ensemble
        quantum_ensemble = QuantumSuperpositionEnsemble(
            base_estimators=base_estimators,
            protected_attributes=protected_attributes,
            fairness_weight=0.5,
            quantum_temperature=1.0,
            measurement_shots=1000,
            enable_quantum_annealing=True
        )

        # Optimization history tracking
        optimization_history = []

        # Iterative quantum optimization
        best_model = None
        best_score = -float('inf')

        for iteration in range(self.quantum_iterations):
            # Adjust quantum parameters based on iteration
            temperature = 1.0 * (0.95 ** iteration)  # Simulated annealing
            quantum_ensemble.quantum_temperature = temperature

            # Fit quantum ensemble
            quantum_ensemble.fit(X_train, y_train)

            # Evaluate on validation set
            predictions = quantum_ensemble.predict(X_val)
            probabilities = quantum_ensemble.predict_proba(X_val)[:, 1]

            # Compute fairness metrics
            overall_metrics, by_group_metrics = compute_fairness_metrics(
                y_true=y_val,
                y_pred=predictions,
                protected=X_val[protected_attributes[0]] if protected_attributes else None,
                y_scores=probabilities,
                enable_optimization=True
            )

            # Quantum scoring function
            accuracy = overall_metrics.get('accuracy', 0.0)
            demographic_parity_diff = abs(overall_metrics.get('demographic_parity_difference', 0.0))
            equalized_odds_diff = abs(overall_metrics.get('equalized_odds_difference', 0.0))

            # Multi-objective quantum score
            quantum_score = (
                0.4 * accuracy -
                0.3 * demographic_parity_diff -
                0.3 * equalized_odds_diff
            )

            # Update best model
            if quantum_score > best_score:
                best_score = quantum_score
                best_model = clone(quantum_ensemble)

            # Record optimization history
            optimization_history.append({
                'iteration': iteration,
                'accuracy': accuracy,
                'demographic_parity_difference': demographic_parity_diff,
                'equalized_odds_difference': equalized_odds_diff,
                'quantum_score': quantum_score,
                'quantum_temperature': temperature
            })

            if iteration % 10 == 0:
                logger.debug(f"Quantum iteration {iteration}: score = {quantum_score:.4f}")

        # Analyze feature entanglement
        entanglement_scores = {}
        if self.entanglement_analysis and hasattr(best_model, 'entanglement_matrix_'):
            feature_names = X_train.drop(protected_attributes, axis=1, errors='ignore').columns
            for i, feat1 in enumerate(feature_names):
                for j, feat2 in enumerate(feature_names):
                    if i < j:
                        score = best_model.entanglement_matrix_[i, j] if best_model.entanglement_matrix_ is not None else 0.0
                        entanglement_scores[(feat1, feat2)] = score

        # Quantum uncertainty quantification
        uncertainty_metrics = {}
        if self.uncertainty_quantification:
            # Compute prediction uncertainties using quantum measurement variance
            val_probabilities = best_model.predict_proba(X_val)
            uncertainty_metrics = {
                'prediction_entropy': -np.mean(np.sum(val_probabilities * np.log(val_probabilities + 1e-8), axis=1)),
                'quantum_coherence': np.mean(best_model.amplitude_weights_**2) if hasattr(best_model, 'amplitude_weights_') else 0.0,
                'measurement_variance': np.var(val_probabilities[:, 1])
            }

        optimization_time = time.time() - start_time

        # Final validation metrics
        final_predictions = best_model.predict(X_val)
        final_probabilities = best_model.predict_proba(X_val)[:, 1]
        final_metrics, _ = compute_fairness_metrics(
            y_true=y_val,
            y_pred=final_predictions,
            protected=X_val[protected_attributes[0]] if protected_attributes else None,
            y_scores=final_probabilities
        )

        result = QuantumFairnessResult(
            optimized_model=best_model,
            fairness_metrics=dict(final_metrics),
            quantum_parameters=best_model.get_quantum_state_info(),
            optimization_history=optimization_history,
            entanglement_scores=entanglement_scores,
            uncertainty_quantification=uncertainty_metrics,
            convergence_achieved=self._check_convergence(optimization_history),
            optimization_time=optimization_time
        )

        logger.info(f"Quantum fairness optimization completed in {optimization_time:.2f}s")
        logger.info(f"Final quantum score: {best_score:.4f}")

        return result

    def _check_convergence(self, history: List[Dict[str, float]]) -> bool:
        """Check if quantum optimization has converged."""
        if len(history) < 10:
            return False

        # Check if quantum scores have stabilized
        recent_scores = [h['quantum_score'] for h in history[-10:]]
        score_variance = np.var(recent_scores)

        return score_variance < 1e-4  # Convergence threshold


def demonstrate_quantum_fairness():
    """Demonstration of quantum fairness algorithms."""
    print("🌌 Quantum-Inspired Fairness Algorithms Demo")

    # Generate synthetic dataset with bias
    np.random.seed(42)
    n_samples = 1000

    # Protected attribute
    protected = np.random.binomial(1, 0.3, n_samples)

    # Features with entanglement
    feature1 = np.random.normal(protected * 0.8, 1.0, n_samples)
    feature2 = np.random.normal(-protected * 0.6, 1.0, n_samples)
    feature3 = np.random.normal(0, 1.0, n_samples)
    feature4 = feature1 * 0.5 + np.random.normal(0, 0.3, n_samples)  # Entangled with feature1

    # Target with bias
    target_prob = 1 / (1 + np.exp(-(feature1 + feature2 + feature3 + protected * 0.7)))
    target = np.random.binomial(1, target_prob, n_samples)

    # Create DataFrame
    X = pd.DataFrame({
        'feature1': feature1,
        'feature2': feature2,
        'feature3': feature3,
        'feature4': feature4,
        'protected': protected
    })
    y = pd.Series(target)

    print(f"📊 Dataset: {len(X)} samples, {len(X.columns)} features")
    print(f"   Protected attribute distribution: {np.bincount(protected)}")
    print(f"   Target distribution: {np.bincount(target)}")

    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print("\n🚀 Testing Quantum Superposition Ensemble")

    # Test quantum superposition ensemble
    quantum_ensemble = QuantumSuperpositionEnsemble(
        protected_attributes=['protected'],
        fairness_weight=0.6,
        quantum_temperature=0.5,
        measurement_shots=500,
        enable_quantum_annealing=True
    )

    quantum_ensemble.fit(X_train, y_train)
    predictions = quantum_ensemble.predict(X_test)
    probabilities = quantum_ensemble.predict_proba(X_test)[:, 1]

    # Evaluate fairness
    overall_metrics, by_group_metrics = compute_fairness_metrics(
        y_true=y_test,
        y_pred=predictions,
        protected=X_test['protected'],
        y_scores=probabilities
    )

    print(f"   Quantum Ensemble Accuracy: {overall_metrics['accuracy']:.3f}")
    print(f"   Demographic Parity Difference: {overall_metrics['demographic_parity_difference']:.3f}")
    print(f"   Equalized Odds Difference: {overall_metrics['equalized_odds_difference']:.3f}")

    # Show quantum state information
    quantum_info = quantum_ensemble.get_quantum_state_info()
    print(f"   Quantum Circuit Depth: {quantum_info['quantum_circuit_depth']}")
    print(f"   Amplitude Weights: {quantum_info['amplitude_weights'][:3]}...")  # Show first 3

    print("\n⚡ Testing Quantum Fairness Optimizer")

    # Test quantum fairness optimizer
    optimizer = QuantumFairnessOptimizer(
        quantum_iterations=30,
        superposition_depth=4,
        entanglement_analysis=True,
        uncertainty_quantification=True
    )

    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.3, random_state=42, stratify=y_train
    )

    quantum_result = optimizer.optimize(
        X_train_split, y_train_split,
        X_val_split, y_val_split,
        protected_attributes=['protected']
    )

    print(f"   Optimization completed in {quantum_result.optimization_time:.2f}s")
    print(f"   Convergence achieved: {quantum_result.convergence_achieved}")
    print(f"   Final accuracy: {quantum_result.fairness_metrics['accuracy']:.3f}")
    print(f"   Final demographic parity difference: {abs(quantum_result.fairness_metrics['demographic_parity_difference']):.3f}")

    # Show entanglement analysis
    if quantum_result.entanglement_scores:
        print(f"   Feature entanglements detected: {len(quantum_result.entanglement_scores)}")
        top_entanglement = max(quantum_result.entanglement_scores.items(), key=lambda x: x[1])
        print(f"   Strongest entanglement: {top_entanglement[0]} (score: {top_entanglement[1]:.3f})")

    # Show uncertainty quantification
    if quantum_result.uncertainty_quantification:
        print(f"   Prediction entropy: {quantum_result.uncertainty_quantification['prediction_entropy']:.3f}")
        print(f"   Quantum coherence: {quantum_result.uncertainty_quantification['quantum_coherence']:.3f}")

    # Test final model on test set
    final_predictions = quantum_result.optimized_model.predict(X_test)
    test_overall, _ = compute_fairness_metrics(
        y_true=y_test,
        y_pred=final_predictions,
        protected=X_test['protected']
    )

    print("\n🧪 Test Set Performance:")
    print(f"   Test Accuracy: {test_overall['accuracy']:.3f}")
    print(f"   Test Demographic Parity Difference: {test_overall['demographic_parity_difference']:.3f}")

    print("\n✅ Quantum fairness algorithms demonstration completed! 🌟")


if __name__ == "__main__":
    demonstrate_quantum_fairness()
