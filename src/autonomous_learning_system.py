"""
Autonomous Learning System v3.0
Advanced self-learning and adaptation engine for continuous system optimization.

This module implements sophisticated autonomous learning capabilities including
reinforcement learning, meta-learning, neural architecture search, and
evolutionary optimization for continuous system improvement.
"""

import asyncio
import logging
import pickle
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


class LearningStrategy(Enum):
    """Learning strategies for autonomous adaptation."""
    REINFORCEMENT_LEARNING = "reinforcement"
    META_LEARNING = "meta_learning"
    EVOLUTIONARY_OPTIMIZATION = "evolutionary"
    NEURAL_ARCHITECTURE_SEARCH = "nas"
    MULTI_ARMED_BANDIT = "bandit"
    GRADIENT_BASED_META_LEARNING = "gradient_meta"


class AdaptationTrigger(Enum):
    """Triggers for autonomous adaptation."""
    PERFORMANCE_DEGRADATION = "performance_drop"
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PERIODIC = "periodic"
    MANUAL = "manual"
    ANOMALY_DETECTION = "anomaly"


@dataclass
class LearningConfiguration:
    """Configuration for autonomous learning system."""
    strategy: LearningStrategy = LearningStrategy.REINFORCEMENT_LEARNING
    adaptation_triggers: List[AdaptationTrigger] = field(default_factory=lambda: [AdaptationTrigger.PERFORMANCE_DEGRADATION])
    learning_rate: float = 0.01
    exploration_rate: float = 0.1
    memory_size: int = 10000
    batch_size: int = 32
    target_performance: float = 0.85
    performance_threshold: float = 0.05
    adaptation_interval: int = 100
    max_adaptations: int = 50
    convergence_patience: int = 10


@dataclass
class PerformanceMetrics:
    """Performance metrics for tracking system behavior."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_time: float
    inference_time: float
    fairness_score: Optional[float] = None
    resource_usage: Optional[Dict[str, float]] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class AdaptationAction:
    """Represents an adaptation action and its parameters."""
    action_type: str
    parameters: Dict[str, Any]
    expected_improvement: float
    confidence: float
    resource_cost: float


class ExperienceBuffer:
    """Experience buffer for storing learning experiences."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.experiences = []
        self.priorities = []

    def add_experience(self, state: Dict[str, Any], action: AdaptationAction,
                      reward: float, next_state: Dict[str, Any], done: bool, priority: float = 1.0):
        """Add experience to buffer with priority."""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'timestamp': time.time()
        }

        if len(self.experiences) >= self.max_size:
            # Remove oldest experience
            self.experiences.pop(0)
            self.priorities.pop(0)

        self.experiences.append(experience)
        self.priorities.append(priority)

    def sample_batch(self, batch_size: int, prioritized: bool = True) -> List[Dict[str, Any]]:
        """Sample batch of experiences."""
        if not self.experiences:
            return []

        if prioritized and len(self.priorities) > 0:
            # Prioritized sampling
            priorities = np.array(self.priorities)
            probabilities = priorities / np.sum(priorities)
            indices = np.random.choice(len(self.experiences),
                                     size=min(batch_size, len(self.experiences)),
                                     p=probabilities, replace=False)
        else:
            # Random sampling
            indices = np.random.choice(len(self.experiences),
                                     size=min(batch_size, len(self.experiences)),
                                     replace=False)

        return [self.experiences[i] for i in indices]

    def get_recent_experiences(self, n: int = 100) -> List[Dict[str, Any]]:
        """Get most recent experiences."""
        return self.experiences[-n:]


class ReinforcementLearningAgent:
    """Reinforcement learning agent for autonomous system optimization."""

    def __init__(self, action_space: List[str], state_space_dims: int, learning_rate: float = 0.01):
        self.action_space = action_space
        self.state_space_dims = state_space_dims
        self.learning_rate = learning_rate

        # Q-table for simple RL (can be replaced with neural networks)
        self.q_table = {}
        self.epsilon = 0.1  # Exploration rate
        self.gamma = 0.95   # Discount factor

        self.total_rewards = []
        self.learning_history = []

    def _state_to_key(self, state: Dict[str, Any]) -> str:
        """Convert state dictionary to string key."""
        # Simplified state representation
        key_components = []
        for key in sorted(state.keys()):
            if isinstance(state[key], (int, float)):
                key_components.append(f"{key}:{state[key]:.3f}")
            else:
                key_components.append(f"{key}:{str(state[key])}")
        return "|".join(key_components)

    def get_action(self, state: Dict[str, Any], exploration: bool = True) -> AdaptationAction:
        """Get action using epsilon-greedy policy."""
        state_key = self._state_to_key(state)

        # Initialize Q-values for new states
        if state_key not in self.q_table:
            self.q_table[state_key] = dict.fromkeys(self.action_space, 0.0)

        # Epsilon-greedy action selection
        if exploration and np.random.random() < self.epsilon:
            # Explore: random action
            action_type = np.random.choice(self.action_space)
        else:
            # Exploit: best known action
            q_values = self.q_table[state_key]
            action_type = max(q_values, key=q_values.get)

        # Generate action parameters based on action type
        parameters = self._generate_action_parameters(action_type, state)

        return AdaptationAction(
            action_type=action_type,
            parameters=parameters,
            expected_improvement=self.q_table[state_key].get(action_type, 0.0),
            confidence=1.0 - self.epsilon,
            resource_cost=self._estimate_resource_cost(action_type, parameters)
        )

    def _generate_action_parameters(self, action_type: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate parameters for action based on current state."""
        base_params = {
            "hyperparameter_tuning": {
                "learning_rate_factor": np.random.uniform(0.5, 2.0),
                "regularization_factor": np.random.uniform(0.1, 10.0),
                "batch_size_factor": np.random.choice([0.5, 1.0, 2.0])
            },
            "model_architecture": {
                "layer_modification": np.random.choice(["add", "remove", "modify"]),
                "layer_type": np.random.choice(["dense", "dropout", "batch_norm"]),
                "size_factor": np.random.uniform(0.5, 2.0)
            },
            "data_augmentation": {
                "augmentation_type": np.random.choice(["noise", "synthetic", "sampling"]),
                "intensity": np.random.uniform(0.1, 0.5)
            },
            "feature_engineering": {
                "feature_selection_ratio": np.random.uniform(0.7, 1.0),
                "feature_transformation": np.random.choice(["pca", "scaling", "polynomial"])
            }
        }

        return base_params.get(action_type, {})

    def _estimate_resource_cost(self, action_type: str, parameters: Dict[str, Any]) -> float:
        """Estimate computational cost of action."""
        base_costs = {
            "hyperparameter_tuning": 0.3,
            "model_architecture": 0.5,
            "data_augmentation": 0.2,
            "feature_engineering": 0.4
        }
        return base_costs.get(action_type, 0.1)

    def update_q_value(self, state: Dict[str, Any], action: AdaptationAction,
                      reward: float, next_state: Dict[str, Any]):
        """Update Q-value using Q-learning."""
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)

        # Initialize Q-values if needed
        if state_key not in self.q_table:
            self.q_table[state_key] = dict.fromkeys(self.action_space, 0.0)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = dict.fromkeys(self.action_space, 0.0)

        # Q-learning update
        current_q = self.q_table[state_key][action.action_type]
        max_next_q = max(self.q_table[next_state_key].values())

        new_q = current_q + self.learning_rate * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_key][action.action_type] = new_q

        # Record learning progress
        self.learning_history.append({
            'state': state_key,
            'action': action.action_type,
            'reward': reward,
            'q_value': new_q,
            'timestamp': time.time()
        })


class MetaLearningAgent:
    """Meta-learning agent for rapid adaptation to new tasks."""

    def __init__(self, base_models: List[Any], adaptation_steps: int = 5):
        self.base_models = base_models
        self.adaptation_steps = adaptation_steps
        self.task_history = []
        self.meta_parameters = {}
        self.adaptation_performance = []

    def adapt_to_task(self, X_support: np.ndarray, y_support: np.ndarray,
                     X_query: np.ndarray, y_query: np.ndarray) -> Dict[str, Any]:
        """Adapt to new task using few-shot learning."""
        start_time = time.time()

        best_model = None
        best_performance = 0
        adaptation_results = []

        for model in self.base_models:
            # Clone model for adaptation
            adapted_model = clone(model)

            # Fine-tune on support set
            for step in range(self.adaptation_steps):
                adapted_model.fit(X_support, y_support)

                # Evaluate on query set
                y_pred = adapted_model.predict(X_query)
                performance = accuracy_score(y_query, y_pred)

                adaptation_results.append({
                    'model': type(model).__name__,
                    'step': step,
                    'performance': performance
                })

                if performance > best_performance:
                    best_performance = performance
                    best_model = adapted_model

        adaptation_time = time.time() - start_time

        # Store task experience
        task_info = {
            'support_size': len(X_support),
            'query_size': len(X_query),
            'best_performance': best_performance,
            'adaptation_time': adaptation_time,
            'adaptation_steps': self.adaptation_steps,
            'timestamp': time.time()
        }
        self.task_history.append(task_info)

        return {
            'adapted_model': best_model,
            'performance': best_performance,
            'adaptation_results': adaptation_results,
            'task_info': task_info
        }

    def update_meta_parameters(self, task_results: List[Dict[str, Any]]):
        """Update meta-learning parameters based on task results."""
        if not task_results:
            return

        # Analyze performance patterns
        performances = [result['performance'] for result in task_results]
        avg_performance = np.mean(performances)
        std_performance = np.std(performances)

        # Update adaptation strategy
        if avg_performance < 0.7:  # Poor performance threshold
            self.adaptation_steps = min(self.adaptation_steps + 1, 10)
        elif std_performance < 0.05:  # Low variance
            self.adaptation_steps = max(self.adaptation_steps - 1, 1)

        # Update meta-parameters
        self.meta_parameters.update({
            'avg_performance': avg_performance,
            'performance_std': std_performance,
            'adaptation_steps': self.adaptation_steps,
            'last_update': time.time()
        })


class EvolutionaryOptimizer:
    """Evolutionary optimization for hyperparameter and architecture search."""

    def __init__(self, population_size: int = 20, mutation_rate: float = 0.1, crossover_rate: float = 0.7):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.population = []
        self.fitness_history = []
        self.generation_count = 0

    def initialize_population(self, parameter_bounds: Dict[str, Tuple[float, float]]) -> List[Dict[str, Any]]:
        """Initialize random population."""
        population = []

        for _ in range(self.population_size):
            individual = {}
            for param_name, (low, high) in parameter_bounds.items():
                if isinstance(low, int) and isinstance(high, int):
                    individual[param_name] = np.random.randint(low, high + 1)
                else:
                    individual[param_name] = np.random.uniform(low, high)
            population.append(individual)

        self.population = population
        return population

    def evaluate_fitness(self, individual: Dict[str, Any], fitness_function: Callable) -> float:
        """Evaluate fitness of individual."""
        try:
            fitness = fitness_function(individual)
            return fitness if fitness is not None else -float('inf')
        except Exception as e:
            logger.warning(f"Fitness evaluation failed: {e}")
            return -float('inf')

    def selection(self, population: List[Dict[str, Any]], fitnesses: List[float]) -> List[Dict[str, Any]]:
        """Tournament selection."""
        selected = []
        tournament_size = 3

        for _ in range(len(population)):
            # Tournament selection
            tournament_indices = np.random.choice(len(population), size=tournament_size, replace=False)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
            selected.append(population[winner_idx].copy())

        return selected

    def crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Uniform crossover."""
        child1, child2 = parent1.copy(), parent2.copy()

        for key in parent1.keys():
            if np.random.random() < self.crossover_rate:
                child1[key], child2[key] = parent2[key], parent1[key]

        return child1, child2

    def mutate(self, individual: Dict[str, Any], parameter_bounds: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """Gaussian mutation."""
        mutated = individual.copy()

        for param_name, value in individual.items():
            if np.random.random() < self.mutation_rate:
                if param_name in parameter_bounds:
                    low, high = parameter_bounds[param_name]
                    if isinstance(low, int) and isinstance(high, int):
                        # Integer parameter
                        mutation_strength = max(1, int((high - low) * 0.1))
                        mutated[param_name] = np.clip(
                            value + np.random.randint(-mutation_strength, mutation_strength + 1),
                            low, high
                        )
                    else:
                        # Float parameter
                        mutation_strength = (high - low) * 0.1
                        mutated[param_name] = np.clip(
                            value + np.random.normal(0, mutation_strength),
                            low, high
                        )

        return mutated

    def evolve(self, fitness_function: Callable, parameter_bounds: Dict[str, Tuple[float, float]],
              generations: int = 50) -> Dict[str, Any]:
        """Run evolutionary optimization."""
        if not self.population:
            self.initialize_population(parameter_bounds)

        best_individual = None
        best_fitness = -float('inf')

        for generation in range(generations):
            # Evaluate fitness
            fitnesses = []
            for individual in self.population:
                fitness = self.evaluate_fitness(individual, fitness_function)
                fitnesses.append(fitness)

            # Track best individual
            max_fitness_idx = np.argmax(fitnesses)
            if fitnesses[max_fitness_idx] > best_fitness:
                best_fitness = fitnesses[max_fitness_idx]
                best_individual = self.population[max_fitness_idx].copy()

            # Record generation statistics
            self.fitness_history.append({
                'generation': generation,
                'best_fitness': np.max(fitnesses),
                'avg_fitness': np.mean(fitnesses),
                'std_fitness': np.std(fitnesses)
            })

            # Selection, crossover, and mutation
            selected = self.selection(self.population, fitnesses)

            new_population = []
            for i in range(0, len(selected), 2):
                parent1 = selected[i]
                parent2 = selected[(i + 1) % len(selected)]

                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1, parameter_bounds)
                child2 = self.mutate(child2, parameter_bounds)

                new_population.extend([child1, child2])

            self.population = new_population[:self.population_size]
            self.generation_count += 1

            # Early stopping
            if len(self.fitness_history) > 10:
                recent_best = [gen['best_fitness'] for gen in self.fitness_history[-10:]]
                if np.std(recent_best) < 0.001:  # Convergence check
                    logger.info(f"Evolutionary optimization converged at generation {generation}")
                    break

        return {
            'best_individual': best_individual,
            'best_fitness': best_fitness,
            'generations': self.generation_count,
            'fitness_history': self.fitness_history
        }


class AutonomousLearningSystem:
    """
    Advanced autonomous learning system for continuous optimization.

    Combines multiple learning strategies to automatically improve system
    performance without human intervention.
    """

    def __init__(self, config: LearningConfiguration):
        self.config = config
        self.experience_buffer = ExperienceBuffer(config.memory_size)

        # Initialize learning agents
        self.rl_agent = None
        self.meta_learning_agent = None
        self.evolutionary_optimizer = EvolutionaryOptimizer()

        # Performance tracking
        self.performance_history = []
        self.adaptation_history = []
        self.current_performance = None

        # State tracking
        self.current_state = {}
        self.system_state = "initializing"
        self.adaptation_count = 0

        # Learning statistics
        self.learning_stats = {
            'total_adaptations': 0,
            'successful_adaptations': 0,
            'average_improvement': 0.0,
            'last_adaptation_time': None
        }

    def initialize(self, base_models: List[Any], action_space: List[str]):
        """Initialize learning system components."""
        logger.info("Initializing autonomous learning system")

        # Initialize RL agent
        self.rl_agent = ReinforcementLearningAgent(
            action_space=action_space,
            state_space_dims=10,  # Simplified state space
            learning_rate=self.config.learning_rate
        )

        # Initialize meta-learning agent
        self.meta_learning_agent = MetaLearningAgent(
            base_models=base_models,
            adaptation_steps=5
        )

        self.system_state = "ready"
        logger.info("Autonomous learning system initialized successfully")

    def compute_system_state(self, model: Any, X: np.ndarray, y: np.ndarray,
                           performance_metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Compute current system state for learning algorithms."""
        state = {
            'performance_accuracy': performance_metrics.accuracy,
            'performance_f1': performance_metrics.f1_score,
            'training_time': performance_metrics.training_time,
            'data_size': len(X),
            'feature_count': X.shape[1] if len(X.shape) > 1 else 1,
            'adaptation_count': self.adaptation_count,
            'time_since_last_adaptation': time.time() - (self.learning_stats.get('last_adaptation_time', time.time()) - 300),
            'average_improvement': self.learning_stats.get('average_improvement', 0.0)
        }

        # Add fairness information if available
        if performance_metrics.fairness_score is not None:
            state['fairness_score'] = performance_metrics.fairness_score

        return state

    def should_adapt(self, current_performance: PerformanceMetrics, trigger: AdaptationTrigger) -> bool:
        """Determine if system should adapt based on trigger conditions."""
        if trigger == AdaptationTrigger.PERFORMANCE_DEGRADATION:
            if not self.performance_history:
                return False

            recent_performance = np.mean([p.accuracy for p in self.performance_history[-5:]])
            return current_performance.accuracy < recent_performance - self.config.performance_threshold

        elif trigger == AdaptationTrigger.PERIODIC:
            if not self.learning_stats.get('last_adaptation_time'):
                return True

            time_since_last = time.time() - self.learning_stats['last_adaptation_time']
            return time_since_last > self.config.adaptation_interval

        elif trigger == AdaptationTrigger.DATA_DRIFT:
            # Simplified drift detection
            if len(self.performance_history) < 10:
                return False

            recent_variance = np.var([p.accuracy for p in self.performance_history[-10:]])
            return recent_variance > 0.01  # Threshold for drift

        return False

    def generate_adaptation_action(self, state: Dict[str, Any], strategy: LearningStrategy) -> AdaptationAction:
        """Generate adaptation action using specified strategy."""
        if strategy == LearningStrategy.REINFORCEMENT_LEARNING:
            if self.rl_agent is None:
                raise ValueError("RL agent not initialized")
            return self.rl_agent.get_action(state)

        elif strategy == LearningStrategy.EVOLUTIONARY_OPTIMIZATION:
            # Use evolutionary algorithm to suggest hyperparameters
            parameter_bounds = {
                'learning_rate': (0.001, 0.1),
                'regularization': (0.01, 10.0),
                'batch_size': (16, 128)
            }

            def fitness_func(params):
                # Simplified fitness based on expected improvement
                return np.random.uniform(0.5, 1.0)  # Placeholder

            result = self.evolutionary_optimizer.evolve(fitness_func, parameter_bounds, generations=10)

            return AdaptationAction(
                action_type="hyperparameter_tuning",
                parameters=result['best_individual'],
                expected_improvement=result['best_fitness'],
                confidence=0.8,
                resource_cost=0.3
            )

        else:
            # Default action for unsupported strategies
            return AdaptationAction(
                action_type="hyperparameter_tuning",
                parameters={'learning_rate_factor': 1.1},
                expected_improvement=0.05,
                confidence=0.5,
                resource_cost=0.1
            )

    def apply_adaptation(self, model: Any, action: AdaptationAction, X: np.ndarray, y: np.ndarray) -> Any:
        """Apply adaptation action to model."""
        try:
            adapted_model = clone(model)

            if action.action_type == "hyperparameter_tuning":
                # Apply hyperparameter changes
                current_params = adapted_model.get_params()

                for param, factor in action.parameters.items():
                    if param.endswith('_factor'):
                        base_param = param.replace('_factor', '')
                        if base_param in current_params:
                            old_value = current_params[base_param]
                            new_value = old_value * factor if isinstance(old_value, (int, float)) else old_value
                            adapted_model.set_params(**{base_param: new_value})

            elif action.action_type == "model_architecture":
                # For now, return original model (architecture changes are complex)
                pass

            # Retrain adapted model
            adapted_model.fit(X, y)
            return adapted_model

        except Exception as e:
            logger.warning(f"Failed to apply adaptation: {e}")
            return model

    def evaluate_adaptation(self, original_performance: PerformanceMetrics,
                          new_performance: PerformanceMetrics, action: AdaptationAction) -> float:
        """Evaluate success of adaptation and compute reward."""
        # Performance improvement reward
        accuracy_improvement = new_performance.accuracy - original_performance.accuracy
        f1_improvement = new_performance.f1_score - original_performance.f1_score

        # Time efficiency reward
        time_efficiency = max(0, original_performance.training_time - new_performance.training_time) / original_performance.training_time

        # Resource cost penalty
        resource_penalty = action.resource_cost * 0.1

        # Combined reward
        reward = accuracy_improvement + 0.5 * f1_improvement + 0.2 * time_efficiency - resource_penalty

        # Fairness bonus if available
        if (original_performance.fairness_score is not None and
            new_performance.fairness_score is not None):
            fairness_improvement = new_performance.fairness_score - original_performance.fairness_score
            reward += 0.3 * fairness_improvement

        return reward

    async def autonomous_learning_cycle(self, model: Any, X: np.ndarray, y: np.ndarray,
                                      evaluation_func: Callable = None) -> Dict[str, Any]:
        """Run complete autonomous learning cycle."""
        logger.info("Starting autonomous learning cycle")
        start_time = time.time()

        if evaluation_func is None:
            evaluation_func = self._default_evaluation

        # Evaluate current performance
        current_metrics = evaluation_func(model, X, y)
        self.current_performance = current_metrics
        self.performance_history.append(current_metrics)

        # Compute system state
        current_state = self.compute_system_state(model, X, y, current_metrics)
        self.current_state = current_state

        adaptations_made = 0
        successful_adaptations = 0
        total_improvement = 0.0

        # Check adaptation triggers
        for trigger in self.config.adaptation_triggers:
            if self.should_adapt(current_metrics, trigger):
                logger.info(f"Adaptation triggered by: {trigger.value}")

                if adaptations_made >= self.config.max_adaptations:
                    logger.info("Maximum adaptations reached for this cycle")
                    break

                # Generate adaptation action
                action = self.generate_adaptation_action(current_state, self.config.strategy)

                # Apply adaptation
                adapted_model = self.apply_adaptation(model, action, X, y)

                # Evaluate adapted model
                new_metrics = evaluation_func(adapted_model, X, y)

                # Compute reward
                reward = self.evaluate_adaptation(current_metrics, new_metrics, action)

                # Update learning systems
                if self.rl_agent is not None:
                    next_state = self.compute_system_state(adapted_model, X, y, new_metrics)
                    self.rl_agent.update_q_value(current_state, action, reward, next_state)
                    current_state = next_state

                # Store experience
                self.experience_buffer.add_experience(
                    state=current_state,
                    action=action,
                    reward=reward,
                    next_state=self.compute_system_state(adapted_model, X, y, new_metrics),
                    done=False,
                    priority=abs(reward) + 1.0
                )

                # Track adaptation
                adaptation_info = {
                    'trigger': trigger.value,
                    'action': action,
                    'reward': reward,
                    'improvement': new_metrics.accuracy - current_metrics.accuracy,
                    'timestamp': time.time()
                }
                self.adaptation_history.append(adaptation_info)

                # Update statistics
                adaptations_made += 1
                if reward > 0:
                    successful_adaptations += 1
                    total_improvement += new_metrics.accuracy - current_metrics.accuracy
                    model = adapted_model  # Accept improvement
                    current_metrics = new_metrics

        # Update learning statistics
        total_time = time.time() - start_time
        self.learning_stats.update({
            'total_adaptations': self.learning_stats.get('total_adaptations', 0) + adaptations_made,
            'successful_adaptations': self.learning_stats.get('successful_adaptations', 0) + successful_adaptations,
            'average_improvement': (self.learning_stats.get('average_improvement', 0.0) +
                                  (total_improvement / max(adaptations_made, 1))) / 2,
            'last_adaptation_time': time.time()
        })

        logger.info(f"Autonomous learning cycle completed in {total_time:.2f}s")
        logger.info(f"Adaptations made: {adaptations_made}, Successful: {successful_adaptations}")

        return {
            'adapted_model': model,
            'final_performance': current_metrics,
            'adaptations_made': adaptations_made,
            'successful_adaptations': successful_adaptations,
            'total_improvement': total_improvement,
            'learning_time': total_time,
            'learning_stats': self.learning_stats.copy()
        }

    def _default_evaluation(self, model: Any, X: np.ndarray, y: np.ndarray) -> PerformanceMetrics:
        """Default evaluation function."""
        start_time = time.time()

        # Training time (for fitted models, this is 0)
        training_time = 0.0

        # Make predictions
        y_pred = model.predict(X)

        # Compute metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)

        inference_time = time.time() - start_time

        return PerformanceMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            training_time=training_time,
            inference_time=inference_time
        )

    def get_learning_summary(self) -> Dict[str, Any]:
        """Get comprehensive learning system summary."""
        return {
            'system_state': self.system_state,
            'total_adaptations': self.adaptation_count,
            'learning_stats': self.learning_stats.copy(),
            'performance_trend': [p.accuracy for p in self.performance_history[-10:]],
            'recent_adaptations': len(self.adaptation_history),
            'experience_buffer_size': len(self.experience_buffer.experiences),
            'configuration': {
                'strategy': self.config.strategy.value,
                'triggers': [t.value for t in self.config.adaptation_triggers],
                'learning_rate': self.config.learning_rate,
                'target_performance': self.config.target_performance
            }
        }

    def save_learning_state(self, filepath: str):
        """Save learning system state to file."""
        state = {
            'config': self.config,
            'performance_history': self.performance_history,
            'adaptation_history': self.adaptation_history,
            'learning_stats': self.learning_stats,
            'experience_buffer': self.experience_buffer.experiences,
            'rl_q_table': self.rl_agent.q_table if self.rl_agent else None
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"Learning state saved to {filepath}")

    def load_learning_state(self, filepath: str):
        """Load learning system state from file."""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)

            self.config = state.get('config', self.config)
            self.performance_history = state.get('performance_history', [])
            self.adaptation_history = state.get('adaptation_history', [])
            self.learning_stats = state.get('learning_stats', {})

            # Restore experience buffer
            if 'experience_buffer' in state:
                for exp in state['experience_buffer']:
                    self.experience_buffer.experiences.append(exp)

            # Restore RL agent Q-table
            if self.rl_agent and 'rl_q_table' in state and state['rl_q_table']:
                self.rl_agent.q_table = state['rl_q_table']

            logger.info(f"Learning state loaded from {filepath}")

        except Exception as e:
            logger.warning(f"Failed to load learning state: {e}")


# Factory functions
def create_autonomous_learning_system(
    strategy: str = "reinforcement",
    adaptation_triggers: List[str] = None,
    learning_rate: float = 0.01,
    target_performance: float = 0.85
) -> AutonomousLearningSystem:
    """Factory function to create autonomous learning system."""

    # Parse strategy
    strategy_map = {
        "reinforcement": LearningStrategy.REINFORCEMENT_LEARNING,
        "meta": LearningStrategy.META_LEARNING,
        "evolutionary": LearningStrategy.EVOLUTIONARY_OPTIMIZATION,
        "bandit": LearningStrategy.MULTI_ARMED_BANDIT
    }
    strategy_enum = strategy_map.get(strategy, LearningStrategy.REINFORCEMENT_LEARNING)

    # Parse triggers
    if adaptation_triggers is None:
        adaptation_triggers = ["performance_drop", "periodic"]

    trigger_map = {
        "performance_drop": AdaptationTrigger.PERFORMANCE_DEGRADATION,
        "periodic": AdaptationTrigger.PERIODIC,
        "drift": AdaptationTrigger.DATA_DRIFT,
        "anomaly": AdaptationTrigger.ANOMALY_DETECTION
    }
    triggers = [trigger_map.get(t, AdaptationTrigger.PERIODIC) for t in adaptation_triggers]

    # Create configuration
    config = LearningConfiguration(
        strategy=strategy_enum,
        adaptation_triggers=triggers,
        learning_rate=learning_rate,
        target_performance=target_performance
    )

    return AutonomousLearningSystem(config)


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    # Generate example data
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

    # Create base models
    base_models = [
        RandomForestClassifier(random_state=42),
        LogisticRegression(random_state=42)
    ]

    # Create autonomous learning system
    learning_system = create_autonomous_learning_system(
        strategy="reinforcement",
        adaptation_triggers=["performance_drop", "periodic"],
        learning_rate=0.01,
        target_performance=0.85
    )

    # Initialize system
    action_space = ["hyperparameter_tuning", "model_architecture", "data_augmentation"]
    learning_system.initialize(base_models, action_space)

    # Run autonomous learning cycle
    initial_model = RandomForestClassifier(random_state=42)
    initial_model.fit(X, y)

    async def run_learning():
        result = await learning_system.autonomous_learning_cycle(initial_model, X, y)

        print("🤖 Autonomous Learning Results:")
        print(f"📊 Adaptations made: {result['adaptations_made']}")
        print(f"✅ Successful adaptations: {result['successful_adaptations']}")
        print(f"📈 Total improvement: {result['total_improvement']:.4f}")
        print(f"⏱️ Learning time: {result['learning_time']:.2f}s")

        summary = learning_system.get_learning_summary()
        print(f"🎯 Final system state: {summary['system_state']}")

        return result

    # Run the autonomous learning system
    result = asyncio.run(run_learning())

    logger.info("Autonomous learning system demonstration completed successfully")
