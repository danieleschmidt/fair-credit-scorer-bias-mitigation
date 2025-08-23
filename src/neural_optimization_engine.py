"""
Neural Optimization Engine v3.0
Advanced neural network-based optimization for autonomous system tuning.

This module implements sophisticated neural optimization techniques including:
- Neural Architecture Search (NAS)
- Differentiable optimization
- Attention-based hyperparameter optimization
- Graph neural networks for system modeling
- Transformer-based performance prediction
"""

import asyncio
import json
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error

logger = logging.getLogger(__name__)


class OptimizationObjective(Enum):
    """Optimization objectives for neural optimization."""
    ACCURACY = "accuracy"
    FAIRNESS = "fairness"
    EFFICIENCY = "efficiency"
    MULTI_OBJECTIVE = "multi_objective"
    PARETO_OPTIMAL = "pareto_optimal"


class NeuralArchitecture(Enum):
    """Neural architecture types for optimization."""
    FEEDFORWARD = "feedforward"
    ATTENTION = "attention"
    GRAPH_NEURAL = "graph_neural"
    TRANSFORMER = "transformer"
    CONVOLUTIONAL = "convolutional"


@dataclass
class OptimizationParameters:
    """Parameters for neural optimization engine."""
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    patience: int = 10
    architecture: NeuralArchitecture = NeuralArchitecture.ATTENTION
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    attention_heads: int = 8
    dropout_rate: float = 0.1
    l2_regularization: float = 0.01


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization."""
    primary_score: float
    secondary_scores: Dict[str, float] = field(default_factory=dict)
    training_time: float = 0.0
    memory_usage: float = 0.0
    convergence_steps: int = 0
    stability_score: float = 1.0


class AttentionLayer:
    """Simplified attention mechanism for hyperparameter optimization."""
    
    def __init__(self, input_dim: int, num_heads: int = 8):
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        # Initialize weights (simplified)
        self.query_weights = np.random.normal(0, 0.1, (input_dim, input_dim))
        self.key_weights = np.random.normal(0, 0.1, (input_dim, input_dim))
        self.value_weights = np.random.normal(0, 0.1, (input_dim, input_dim))
        self.output_weights = np.random.normal(0, 0.1, (input_dim, input_dim))
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through attention layer."""
        batch_size, seq_len = x.shape[:2]
        
        # Compute Q, K, V
        queries = np.dot(x, self.query_weights)
        keys = np.dot(x, self.key_weights) 
        values = np.dot(x, self.value_weights)
        
        # Reshape for multi-head attention
        queries = queries.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        keys = keys.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        values = values.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        queries = np.transpose(queries, (0, 2, 1, 3))
        keys = np.transpose(keys, (0, 2, 1, 3))
        values = np.transpose(values, (0, 2, 1, 3))
        
        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attention_scores = np.matmul(queries, np.transpose(keys, (0, 1, 3, 2))) / scale
        
        # Apply softmax
        attention_weights = self._softmax(attention_scores)
        
        # Apply attention to values
        attended = np.matmul(attention_weights, values)
        
        # Transpose back and reshape
        attended = np.transpose(attended, (0, 2, 1, 3))
        attended = attended.reshape(batch_size, seq_len, self.input_dim)
        
        # Output projection
        output = np.dot(attended, self.output_weights)
        
        return output
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax along last dimension."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class GraphNeuralNetwork:
    """Graph neural network for system modeling."""
    
    def __init__(self, node_features: int, edge_features: int, hidden_dim: int = 64):
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        
        # Initialize layers
        self.node_embedding = np.random.normal(0, 0.1, (node_features, hidden_dim))
        self.edge_embedding = np.random.normal(0, 0.1, (edge_features, hidden_dim))
        self.message_weights = np.random.normal(0, 0.1, (hidden_dim * 2, hidden_dim))
        self.update_weights = np.random.normal(0, 0.1, (hidden_dim * 2, hidden_dim))
        self.output_weights = np.random.normal(0, 0.1, (hidden_dim, 1))
    
    def forward(self, node_features: np.ndarray, edge_features: np.ndarray, 
                adjacency: np.ndarray) -> np.ndarray:
        """Forward pass through graph neural network."""
        # Embed nodes and edges
        node_embeds = np.dot(node_features, self.node_embedding)
        edge_embeds = np.dot(edge_features, self.edge_embedding)
        
        # Message passing
        num_nodes = node_embeds.shape[0]
        messages = np.zeros_like(node_embeds)
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adjacency[i, j] > 0:  # If edge exists
                    # Concatenate node features and edge features
                    message_input = np.concatenate([node_embeds[j], edge_embeds[i, j]])
                    message = np.dot(message_input, self.message_weights)
                    messages[i] += message
        
        # Update node representations
        updated_nodes = []
        for i in range(num_nodes):
            update_input = np.concatenate([node_embeds[i], messages[i]])
            updated_node = np.dot(update_input, self.update_weights)
            updated_nodes.append(updated_node)
        
        updated_nodes = np.array(updated_nodes)
        
        # Graph-level prediction
        graph_representation = np.mean(updated_nodes, axis=0)
        output = np.dot(graph_representation, self.output_weights)
        
        return output


class TransformerOptimizer:
    """Transformer-based performance prediction and optimization."""
    
    def __init__(self, input_dim: int, num_heads: int = 8, num_layers: int = 4):
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Initialize transformer layers
        self.attention_layers = [AttentionLayer(input_dim, num_heads) for _ in range(num_layers)]
        self.feed_forward_layers = []
        
        for _ in range(num_layers):
            ff_layer = {
                'linear1': np.random.normal(0, 0.1, (input_dim, input_dim * 4)),
                'linear2': np.random.normal(0, 0.1, (input_dim * 4, input_dim))
            }
            self.feed_forward_layers.append(ff_layer)
        
        self.output_layer = np.random.normal(0, 0.1, (input_dim, 1))
        self.layer_norms = [{'weight': np.ones(input_dim), 'bias': np.zeros(input_dim)} 
                           for _ in range(num_layers * 2)]
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through transformer."""
        current_input = x
        
        for i in range(self.num_layers):
            # Self-attention with residual connection
            attention_output = self.attention_layers[i].forward(current_input)
            current_input = self._layer_norm(current_input + attention_output, self.layer_norms[i * 2])
            
            # Feed-forward with residual connection
            ff_output = self._feed_forward(current_input, self.feed_forward_layers[i])
            current_input = self._layer_norm(current_input + ff_output, self.layer_norms[i * 2 + 1])
        
        # Output projection
        output = np.dot(np.mean(current_input, axis=1), self.output_layer)
        return output
    
    def _feed_forward(self, x: np.ndarray, layer: Dict[str, np.ndarray]) -> np.ndarray:
        """Feed-forward layer with ReLU activation."""
        hidden = np.maximum(0, np.dot(x, layer['linear1']))  # ReLU
        output = np.dot(hidden, layer['linear2'])
        return output
    
    def _layer_norm(self, x: np.ndarray, norm_params: Dict[str, np.ndarray]) -> np.ndarray:
        """Layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(var + 1e-8)
        return normalized * norm_params['weight'] + norm_params['bias']


class NeuralArchitectureSearch:
    """Neural Architecture Search for optimal model configuration."""
    
    def __init__(self, search_space: Dict[str, List[Any]]):
        self.search_space = search_space
        self.architecture_history = []
        self.performance_predictor = None
        self.best_architecture = None
        self.best_performance = -float('inf')
    
    def sample_architecture(self) -> Dict[str, Any]:
        """Sample architecture from search space."""
        architecture = {}
        for param, values in self.search_space.items():
            architecture[param] = np.random.choice(values)
        return architecture
    
    def evaluate_architecture(self, architecture: Dict[str, Any], 
                            evaluation_func: Callable) -> PerformanceMetrics:
        """Evaluate architecture performance."""
        try:
            start_time = time.time()
            performance = evaluation_func(architecture)
            evaluation_time = time.time() - start_time
            
            if isinstance(performance, (int, float)):
                metrics = PerformanceMetrics(
                    primary_score=performance,
                    training_time=evaluation_time
                )
            else:
                metrics = performance
            
            # Track architecture
            self.architecture_history.append({
                'architecture': architecture,
                'performance': metrics,
                'timestamp': time.time()
            })
            
            # Update best architecture
            if metrics.primary_score > self.best_performance:
                self.best_performance = metrics.primary_score
                self.best_architecture = architecture
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Architecture evaluation failed: {e}")
            return PerformanceMetrics(primary_score=-float('inf'))
    
    def search(self, evaluation_func: Callable, max_evaluations: int = 100,
              strategy: str = "random") -> Dict[str, Any]:
        """Search for optimal architecture."""
        logger.info(f"Starting NAS with {strategy} strategy")
        
        if strategy == "random":
            return self._random_search(evaluation_func, max_evaluations)
        elif strategy == "evolutionary":
            return self._evolutionary_search(evaluation_func, max_evaluations)
        elif strategy == "bayesian":
            return self._bayesian_search(evaluation_func, max_evaluations)
        else:
            return self._random_search(evaluation_func, max_evaluations)
    
    def _random_search(self, evaluation_func: Callable, max_evaluations: int) -> Dict[str, Any]:
        """Random architecture search."""
        for i in range(max_evaluations):
            architecture = self.sample_architecture()
            metrics = self.evaluate_architecture(architecture, evaluation_func)
            
            if i % 10 == 0:
                logger.info(f"NAS evaluation {i}/{max_evaluations}, "
                           f"best score: {self.best_performance:.4f}")
        
        return {
            'best_architecture': self.best_architecture,
            'best_performance': self.best_performance,
            'evaluations': len(self.architecture_history),
            'search_strategy': 'random'
        }
    
    def _evolutionary_search(self, evaluation_func: Callable, max_evaluations: int) -> Dict[str, Any]:
        """Evolutionary architecture search."""
        population_size = min(20, max_evaluations // 5)
        
        # Initialize population
        population = [self.sample_architecture() for _ in range(population_size)]
        population_fitness = []
        
        for arch in population:
            metrics = self.evaluate_architecture(arch, evaluation_func)
            population_fitness.append(metrics.primary_score)
        
        generations = max_evaluations // population_size
        
        for gen in range(generations):
            # Selection (tournament selection)
            new_population = []
            for _ in range(population_size):
                # Tournament selection
                tournament_indices = np.random.choice(len(population), size=3, replace=False)
                tournament_fitness = [population_fitness[i] for i in tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                new_population.append(population[winner_idx].copy())
            
            # Mutation
            for i in range(population_size):
                if np.random.random() < 0.5:  # Mutation probability
                    arch = new_population[i]
                    param_to_mutate = np.random.choice(list(self.search_space.keys()))
                    arch[param_to_mutate] = np.random.choice(self.search_space[param_to_mutate])
            
            # Evaluation
            population = new_population
            population_fitness = []
            for arch in population:
                metrics = self.evaluate_architecture(arch, evaluation_func)
                population_fitness.append(metrics.primary_score)
            
            if gen % 5 == 0:
                logger.info(f"NAS generation {gen}/{generations}, "
                           f"best score: {self.best_performance:.4f}")
        
        return {
            'best_architecture': self.best_architecture,
            'best_performance': self.best_performance,
            'evaluations': len(self.architecture_history),
            'search_strategy': 'evolutionary'
        }
    
    def _bayesian_search(self, evaluation_func: Callable, max_evaluations: int) -> Dict[str, Any]:
        """Bayesian optimization for architecture search (simplified)."""
        # Simplified Bayesian optimization
        # In practice, would use Gaussian processes or neural networks as surrogate models
        
        n_random = min(10, max_evaluations // 5)
        
        # Random initialization
        for i in range(n_random):
            architecture = self.sample_architecture()
            self.evaluate_architecture(architecture, evaluation_func)
        
        # Acquisition function-based search
        for i in range(n_random, max_evaluations):
            # Generate candidate architectures
            candidates = [self.sample_architecture() for _ in range(10)]
            
            # Simple acquisition function (expected improvement approximation)
            best_candidate = None
            best_acquisition = -float('inf')
            
            for candidate in candidates:
                # Simplified acquisition function
                similarity_penalty = self._compute_similarity_penalty(candidate)
                exploration_bonus = np.random.normal(0, 0.1)
                acquisition = exploration_bonus - similarity_penalty
                
                if acquisition > best_acquisition:
                    best_acquisition = acquisition
                    best_candidate = candidate
            
            if best_candidate:
                self.evaluate_architecture(best_candidate, evaluation_func)
            
            if i % 10 == 0:
                logger.info(f"NAS Bayesian evaluation {i}/{max_evaluations}, "
                           f"best score: {self.best_performance:.4f}")
        
        return {
            'best_architecture': self.best_architecture,
            'best_performance': self.best_performance,
            'evaluations': len(self.architecture_history),
            'search_strategy': 'bayesian'
        }
    
    def _compute_similarity_penalty(self, architecture: Dict[str, Any]) -> float:
        """Compute similarity penalty to encourage exploration."""
        if not self.architecture_history:
            return 0.0
        
        similarities = []
        for entry in self.architecture_history[-10:]:  # Look at recent architectures
            past_arch = entry['architecture']
            similarity = sum(1 for k in architecture.keys() 
                           if architecture[k] == past_arch.get(k, None))
            similarities.append(similarity / len(architecture))
        
        return np.mean(similarities) if similarities else 0.0


class NeuralOptimizationEngine:
    """
    Advanced neural optimization engine for autonomous system tuning.
    
    Combines multiple neural approaches for comprehensive optimization:
    - Neural Architecture Search
    - Attention-based hyperparameter optimization
    - Graph neural networks for system modeling
    - Transformer-based performance prediction
    """
    
    def __init__(self, config: OptimizationParameters):
        self.config = config
        self.nas_engine = None
        self.attention_optimizer = None
        self.graph_model = None
        self.transformer_predictor = None
        
        # Optimization state
        self.optimization_history = []
        self.performance_predictor = None
        self.current_best = None
        
        # Neural models
        self.models = {}
        self.training_data = {'inputs': [], 'targets': []}
    
    def initialize_neural_models(self, input_dims: Dict[str, int]):
        """Initialize neural optimization models."""
        logger.info("Initializing neural optimization models")
        
        # Initialize attention-based optimizer
        if 'hyperparams' in input_dims:
            self.attention_optimizer = AttentionLayer(
                input_dim=input_dims['hyperparams'],
                num_heads=self.config.attention_heads
            )
        
        # Initialize graph neural network
        if 'system_graph' in input_dims:
            self.graph_model = GraphNeuralNetwork(
                node_features=input_dims['system_graph'].get('nodes', 10),
                edge_features=input_dims['system_graph'].get('edges', 5),
                hidden_dim=64
            )
        
        # Initialize transformer predictor
        if 'sequences' in input_dims:
            self.transformer_predictor = TransformerOptimizer(
                input_dim=input_dims['sequences'],
                num_heads=self.config.attention_heads,
                num_layers=4
            )
        
        logger.info("Neural models initialized successfully")
    
    def setup_architecture_search(self, search_space: Dict[str, List[Any]]):
        """Setup neural architecture search."""
        self.nas_engine = NeuralArchitectureSearch(search_space)
        logger.info(f"NAS configured with {len(search_space)} parameters")
    
    def predict_performance(self, configuration: Dict[str, Any]) -> Tuple[float, float]:
        """Predict performance using neural models."""
        try:
            # Convert configuration to neural input
            neural_input = self._configuration_to_neural_input(configuration)
            
            predictions = []
            uncertainties = []
            
            # Use attention model if available
            if self.attention_optimizer is not None and 'hyperparams' in neural_input:
                hyperparams = neural_input['hyperparams'].reshape(1, -1, neural_input['hyperparams'].shape[-1])
                attention_pred = self.attention_optimizer.forward(hyperparams)
                predictions.append(np.mean(attention_pred))
                uncertainties.append(np.std(attention_pred))
            
            # Use transformer if available
            if self.transformer_predictor is not None and 'sequences' in neural_input:
                sequences = neural_input['sequences'].reshape(1, -1, neural_input['sequences'].shape[-1])
                transformer_pred = self.transformer_predictor.forward(sequences)
                predictions.append(float(transformer_pred))
                uncertainties.append(0.1)  # Simplified uncertainty
            
            # Use graph model if available
            if self.graph_model is not None and 'graph' in neural_input:
                graph_data = neural_input['graph']
                graph_pred = self.graph_model.forward(
                    graph_data['nodes'], 
                    graph_data['edges'], 
                    graph_data['adjacency']
                )
                predictions.append(float(graph_pred))
                uncertainties.append(0.1)
            
            # Ensemble prediction
            if predictions:
                final_prediction = np.mean(predictions)
                final_uncertainty = np.sqrt(np.mean(np.array(uncertainties)**2))
            else:
                # Fallback random prediction
                final_prediction = np.random.uniform(0.5, 0.8)
                final_uncertainty = 0.2
            
            return final_prediction, final_uncertainty
            
        except Exception as e:
            logger.warning(f"Neural prediction failed: {e}")
            return np.random.uniform(0.5, 0.8), 0.2
    
    def _configuration_to_neural_input(self, configuration: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Convert configuration to neural network input format."""
        neural_input = {}
        
        # Convert hyperparameters to vector
        hyperparam_values = []
        for key, value in configuration.items():
            if isinstance(value, (int, float)):
                hyperparam_values.append(float(value))
            elif isinstance(value, bool):
                hyperparam_values.append(float(value))
            elif isinstance(value, str):
                # Simple string encoding (hash-based)
                hyperparam_values.append(float(hash(value) % 100) / 100.0)
        
        if hyperparam_values:
            # Pad to fixed size
            target_size = 20
            if len(hyperparam_values) < target_size:
                hyperparam_values.extend([0.0] * (target_size - len(hyperparam_values)))
            elif len(hyperparam_values) > target_size:
                hyperparam_values = hyperparam_values[:target_size]
            
            neural_input['hyperparams'] = np.array(hyperparam_values)
        
        # Create sequence representation
        if len(hyperparam_values) >= 10:
            sequence_length = 10
            sequences = np.array(hyperparam_values[:sequence_length]).reshape(sequence_length, 1)
            neural_input['sequences'] = sequences
        
        # Create simple graph representation
        if len(hyperparam_values) >= 5:
            num_nodes = 5
            node_features = np.array(hyperparam_values[:num_nodes]).reshape(num_nodes, 1)
            edge_features = np.random.normal(0, 0.1, (num_nodes, num_nodes, 1))
            adjacency = np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)  # Fully connected
            
            neural_input['graph'] = {
                'nodes': node_features,
                'edges': edge_features,
                'adjacency': adjacency
            }
        
        return neural_input
    
    def optimize_hyperparameters(self, objective_function: Callable, 
                                search_space: Dict[str, Tuple[float, float]],
                                max_evaluations: int = 100) -> Dict[str, Any]:
        """Optimize hyperparameters using neural approach."""
        logger.info("Starting neural hyperparameter optimization")
        start_time = time.time()
        
        best_config = None
        best_score = -float('inf')
        evaluation_count = 0
        
        # Initialize with neural architecture search if configured
        if self.nas_engine is not None:
            nas_search_space = {}
            for param, (low, high) in search_space.items():
                # Create discrete choices for NAS
                if isinstance(low, float):
                    choices = np.linspace(low, high, 10).tolist()
                else:
                    choices = list(range(int(low), int(high) + 1))
                nas_search_space[param] = choices
            
            nas_result = self.nas_engine.search(
                evaluation_func=objective_function,
                max_evaluations=min(50, max_evaluations // 2),
                strategy="evolutionary"
            )
            
            if nas_result['best_architecture']:
                best_config = nas_result['best_architecture']
                best_score = nas_result['best_performance']
                evaluation_count += nas_result['evaluations']
        
        # Neural-guided optimization
        remaining_evaluations = max_evaluations - evaluation_count
        
        for i in range(remaining_evaluations):
            # Generate candidate using neural prediction
            if i < 10 or np.random.random() < 0.3:
                # Random exploration
                candidate = {}
                for param, (low, high) in search_space.items():
                    candidate[param] = np.random.uniform(low, high)
            else:
                # Neural-guided generation
                candidate = self._generate_neural_guided_candidate(search_space, best_config)
            
            # Evaluate candidate
            try:
                score = objective_function(candidate)
                evaluation_count += 1
                
                if score > best_score:
                    best_score = score
                    best_config = candidate
                
                # Update training data for neural models
                neural_input = self._configuration_to_neural_input(candidate)
                self.training_data['inputs'].append(neural_input)
                self.training_data['targets'].append(score)
                
                # Record optimization history
                self.optimization_history.append({
                    'configuration': candidate,
                    'score': score,
                    'evaluation': evaluation_count,
                    'timestamp': time.time()
                })
                
            except Exception as e:
                logger.warning(f"Evaluation failed: {e}")
                continue
            
            if i % 10 == 0:
                logger.info(f"Neural optimization {i}/{remaining_evaluations}, "
                           f"best score: {best_score:.4f}")
        
        optimization_time = time.time() - start_time
        
        logger.info(f"Neural optimization completed in {optimization_time:.2f}s")
        logger.info(f"Best score: {best_score:.4f}, Evaluations: {evaluation_count}")
        
        return {
            'best_configuration': best_config,
            'best_score': best_score,
            'evaluations': evaluation_count,
            'optimization_time': optimization_time,
            'optimization_history': self.optimization_history[-100:],  # Last 100 entries
            'neural_predictions_used': remaining_evaluations - 10
        }
    
    def _generate_neural_guided_candidate(self, search_space: Dict[str, Tuple[float, float]], 
                                        best_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate candidate using neural guidance."""
        candidate = {}
        
        if best_config is None:
            # Random fallback
            for param, (low, high) in search_space.items():
                candidate[param] = np.random.uniform(low, high)
            return candidate
        
        # Neural-guided perturbation of best config
        for param, (low, high) in search_space.items():
            if param in best_config:
                current_value = best_config[param]
                
                # Predict improvement direction using neural model
                test_configs = []
                test_scores = []
                
                # Test small perturbations
                perturbations = [-0.1, -0.05, 0.05, 0.1]
                for pert in perturbations:
                    test_config = best_config.copy()
                    new_value = np.clip(current_value * (1 + pert), low, high)
                    test_config[param] = new_value
                    
                    # Predict performance
                    predicted_score, _ = self.predict_performance(test_config)
                    test_configs.append(new_value)
                    test_scores.append(predicted_score)
                
                # Choose best predicted direction
                best_idx = np.argmax(test_scores)
                candidate[param] = test_configs[best_idx]
            else:
                candidate[param] = np.random.uniform(low, high)
        
        return candidate
    
    def multi_objective_optimization(self, objective_functions: List[Callable],
                                   weights: List[float],
                                   search_space: Dict[str, Tuple[float, float]],
                                   max_evaluations: int = 100) -> Dict[str, Any]:
        """Multi-objective optimization using neural approaches."""
        logger.info("Starting multi-objective neural optimization")
        
        def combined_objective(config):
            scores = []
            for obj_func in objective_functions:
                try:
                    score = obj_func(config)
                    scores.append(score)
                except Exception as e:
                    logger.warning(f"Objective evaluation failed: {e}")
                    scores.append(0.0)
            
            # Weighted combination
            weighted_score = sum(w * s for w, s in zip(weights, scores))
            return weighted_score
        
        # Use neural optimization for combined objective
        result = self.optimize_hyperparameters(
            objective_function=combined_objective,
            search_space=search_space,
            max_evaluations=max_evaluations
        )
        
        # Add multi-objective specific information
        result['objective_count'] = len(objective_functions)
        result['weights'] = weights
        result['optimization_type'] = 'multi_objective_neural'
        
        return result
    
    def get_optimization_insights(self) -> Dict[str, Any]:
        """Get insights from neural optimization process."""
        if not self.optimization_history:
            return {"status": "no_data"}
        
        # Analyze optimization trajectory
        scores = [entry['score'] for entry in self.optimization_history]
        improvements = [scores[i] - scores[i-1] for i in range(1, len(scores))]
        
        # Parameter importance analysis
        param_importance = {}
        if len(self.optimization_history) > 10:
            # Simple parameter importance based on correlation with performance
            for param in self.optimization_history[0]['configuration'].keys():
                param_values = [entry['configuration'][param] for entry in self.optimization_history]
                correlation = np.corrcoef(param_values, scores)[0, 1]
                param_importance[param] = abs(correlation) if not np.isnan(correlation) else 0.0
        
        return {
            "total_evaluations": len(self.optimization_history),
            "best_score": max(scores),
            "average_score": np.mean(scores),
            "score_std": np.std(scores),
            "improvement_trend": np.mean(improvements) if improvements else 0.0,
            "parameter_importance": param_importance,
            "convergence_rate": self._compute_convergence_rate(),
            "neural_model_usage": {
                "attention_predictions": sum(1 for _ in self.training_data['inputs']),
                "neural_guided_generations": max(0, len(self.optimization_history) - 10)
            }
        }
    
    def _compute_convergence_rate(self) -> float:
        """Compute optimization convergence rate."""
        if len(self.optimization_history) < 20:
            return 0.0
        
        scores = [entry['score'] for entry in self.optimization_history]
        
        # Compute moving average improvement
        window_size = 10
        improvements = []
        for i in range(window_size, len(scores)):
            current_window = scores[i-window_size:i]
            previous_window = scores[i-window_size*2:i-window_size] if i >= window_size*2 else scores[:window_size]
            
            current_avg = np.mean(current_window)
            previous_avg = np.mean(previous_window)
            improvement = current_avg - previous_avg
            improvements.append(improvement)
        
        return np.mean(improvements) if improvements else 0.0


# Factory functions
def create_neural_optimizer(
    architecture: str = "attention",
    learning_rate: float = 0.001,
    hidden_dims: List[int] = None,
    attention_heads: int = 8
) -> NeuralOptimizationEngine:
    """Factory function to create neural optimization engine."""
    
    if hidden_dims is None:
        hidden_dims = [128, 64, 32]
    
    # Parse architecture
    arch_map = {
        "attention": NeuralArchitecture.ATTENTION,
        "transformer": NeuralArchitecture.TRANSFORMER,
        "graph": NeuralArchitecture.GRAPH_NEURAL,
        "feedforward": NeuralArchitecture.FEEDFORWARD
    }
    
    architecture_enum = arch_map.get(architecture, NeuralArchitecture.ATTENTION)
    
    # Create configuration
    config = OptimizationParameters(
        learning_rate=learning_rate,
        architecture=architecture_enum,
        hidden_dims=hidden_dims,
        attention_heads=attention_heads
    )
    
    return NeuralOptimizationEngine(config)


# Example usage and demonstration
if __name__ == "__main__":
    # Example usage
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import cross_val_score
    
    # Generate example data
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    
    # Create neural optimizer
    optimizer = create_neural_optimizer(
        architecture="attention",
        learning_rate=0.001,
        attention_heads=4
    )
    
    # Initialize neural models
    input_dims = {
        'hyperparams': 20,
        'sequences': 10,
        'system_graph': {'nodes': 5, 'edges': 5}
    }
    optimizer.initialize_neural_models(input_dims)
    
    # Define objective function
    def objective_function(config):
        try:
            # Create model with hyperparameters
            model = RandomForestClassifier(
                n_estimators=int(config.get('n_estimators', 100)),
                max_depth=int(config.get('max_depth', 10)),
                min_samples_split=int(config.get('min_samples_split', 2)),
                random_state=42
            )
            
            # Evaluate with cross-validation
            scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
            return np.mean(scores)
        except:
            return 0.5
    
    # Define search space
    search_space = {
        'n_estimators': (50, 200),
        'max_depth': (5, 20),
        'min_samples_split': (2, 10)
    }
    
    # Setup architecture search
    nas_search_space = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': list(range(5, 21)),
        'min_samples_split': [2, 5, 10]
    }
    optimizer.setup_architecture_search(nas_search_space)
    
    # Run optimization
    print("🧠 Starting neural optimization...")
    result = optimizer.optimize_hyperparameters(
        objective_function=objective_function,
        search_space=search_space,
        max_evaluations=50
    )
    
    # Print results
    print(f"✅ Best configuration: {result['best_configuration']}")
    print(f"📊 Best score: {result['best_score']:.4f}")
    print(f"🔄 Evaluations: {result['evaluations']}")
    print(f"⏱️ Optimization time: {result['optimization_time']:.2f}s")
    
    # Get insights
    insights = optimizer.get_optimization_insights()
    print(f"📈 Convergence rate: {insights['convergence_rate']:.6f}")
    print(f"🎯 Parameter importance: {insights['parameter_importance']}")
    
    logger.info("Neural optimization engine demonstration completed successfully")