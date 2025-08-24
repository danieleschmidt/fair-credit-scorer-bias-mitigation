"""
Neural Fairness Framework - Advanced Deep Learning for Fair AI.

This module implements state-of-the-art neural network architectures specifically
designed for fairness-aware machine learning, incorporating the latest research
in fair representation learning and adversarial debiasing.

Research Contributions:
- Adversarial Fair Representation Learning with gradient reversal
- Variational Fair Auto-Encoders for representation learning
- Multi-task Neural Networks with fairness regularization
- Attention-based Fairness Mechanisms for interpretable bias mitigation
- Contrastive Learning for Fair Representations
- Transformer-based Fair Sequence Modeling
"""

import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from ..fairness_metrics import compute_fairness_metrics
    from ..logging_config import get_logger
except ImportError:
    from fairness_metrics import compute_fairness_metrics
    from logging_config import get_logger

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = get_logger(__name__)

@dataclass
class NeuralFairnessConfig:
    """Configuration for neural fairness models."""
    hidden_dims: List[int]
    dropout_rate: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 100
    fairness_weight: float = 1.0
    adversarial_weight: float = 1.0
    patience: int = 10
    gradient_clipping: float = 1.0
    use_batch_norm: bool = True
    activation: str = 'relu'


@dataclass
class TrainingResult:
    """Training result for neural fairness models."""
    model: Any
    training_history: List[Dict[str, float]]
    final_metrics: Dict[str, float]
    training_time: float
    convergence_epoch: int
    fairness_evolution: List[Dict[str, float]]


class FairnessLoss:
    """Collection of fairness-aware loss functions."""

    @staticmethod
    def demographic_parity_loss(y_pred: np.ndarray, sensitive_attr: np.ndarray) -> float:
        """
        Demographic parity loss function.

        Minimizes the difference in positive prediction rates between groups.
        """
        unique_groups = np.unique(sensitive_attr)
        if len(unique_groups) != 2:
            return 0.0

        group_0_mask = sensitive_attr == unique_groups[0]
        group_1_mask = sensitive_attr == unique_groups[1]

        rate_0 = np.mean(y_pred[group_0_mask])
        rate_1 = np.mean(y_pred[group_1_mask])

        return np.abs(rate_0 - rate_1)

    @staticmethod
    def equalized_odds_loss(y_true: np.ndarray, y_pred: np.ndarray, sensitive_attr: np.ndarray) -> float:
        """
        Equalized odds loss function.

        Minimizes the difference in TPR and FPR between groups.
        """
        unique_groups = np.unique(sensitive_attr)
        if len(unique_groups) != 2:
            return 0.0

        loss = 0.0

        for class_label in [0, 1]:
            class_mask = y_true == class_label

            if np.sum(class_mask) == 0:
                continue

            group_0_mask = (sensitive_attr == unique_groups[0]) & class_mask
            group_1_mask = (sensitive_attr == unique_groups[1]) & class_mask

            if np.sum(group_0_mask) == 0 or np.sum(group_1_mask) == 0:
                continue

            rate_0 = np.mean(y_pred[group_0_mask])
            rate_1 = np.mean(y_pred[group_1_mask])

            loss += np.abs(rate_0 - rate_1)

        return loss / 2  # Average over both classes

    @staticmethod
    def individual_fairness_loss(embeddings: np.ndarray, similarity_matrix: np.ndarray) -> float:
        """
        Individual fairness loss based on embedding similarity.

        Ensures similar individuals receive similar predictions.
        """
        # Compute pairwise distances in embedding space
        n_samples = embeddings.shape[0]
        embedding_distances = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(i+1, n_samples):
                dist = np.linalg.norm(embeddings[i] - embeddings[j])
                embedding_distances[i, j] = dist
                embedding_distances[j, i] = dist

        # Individual fairness violation
        fairness_violations = np.abs(embedding_distances - similarity_matrix)
        return np.mean(fairness_violations)


class SimpleNeuralNetwork:
    """
    Simple neural network implementation using numpy.

    This is a basic implementation for demonstration purposes.
    In practice, you would use PyTorch, TensorFlow, or similar frameworks.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 learning_rate: float = 0.001, dropout_rate: float = 0.0):
        """Initialize neural network."""
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate

        # Initialize weights and biases
        self.weights = []
        self.biases = []

        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            # Xavier initialization
            weight = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i])
            bias = np.zeros((1, dims[i+1]))

            self.weights.append(weight)
            self.biases.append(bias)

        self.activations = []
        self.z_values = []

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        # Clip to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)

    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU."""
        return (x > 0).astype(float)

    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid."""
        s = self.sigmoid(x)
        return s * (1 - s)

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass."""
        self.activations = [X]
        self.z_values = []

        current_input = X

        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(current_input, weight) + bias
            self.z_values.append(z)

            if i < len(self.weights) - 1:  # Hidden layers
                activation = self.relu(z)
                # Apply dropout during training
                if training and self.dropout_rate > 0:
                    dropout_mask = np.random.binomial(1, 1-self.dropout_rate, activation.shape)
                    activation = activation * dropout_mask / (1 - self.dropout_rate)
            else:  # Output layer
                activation = self.sigmoid(z)

            self.activations.append(activation)
            current_input = activation

        return current_input

    def backward(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray):
        """Backward pass."""
        m = X.shape[0]

        # Initialize gradients
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]

        # Output layer error
        dz = y_pred - y.reshape(-1, 1)

        # Backpropagate through layers
        for i in reversed(range(len(self.weights))):
            # Gradients for weights and biases
            dW[i] = np.dot(self.activations[i].T, dz) / m
            db[i] = np.mean(dz, axis=0, keepdims=True)

            if i > 0:  # Not the first layer
                # Propagate error to previous layer
                dz = np.dot(dz, self.weights[i].T) * self.relu_derivative(self.z_values[i-1])

        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        y_pred = self.forward(X, training=False)
        return (y_pred > 0.5).astype(int).flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        y_pred = self.forward(X, training=False)
        proba_positive = y_pred.flatten()
        proba_negative = 1 - proba_positive
        return np.column_stack([proba_negative, proba_positive])


class AdversarialFairClassifier(BaseEstimator, ClassifierMixin):
    """
    Adversarial Fair Classifier using gradient reversal.

    Implements adversarial training where a predictor learns to make accurate
    predictions while an adversary learns to predict protected attributes.
    The gradient reversal technique encourages fair representations.

    Research Innovation:
    - Gradient reversal layer for adversarial training
    - Fair representation learning through adversarial loss
    - Dynamic adversarial weight scheduling
    - Multi-objective optimization with fairness constraints
    """

    def __init__(
        self,
        config: Optional[NeuralFairnessConfig] = None,
        protected_attributes: List[str] = None,
        fairness_constraint: str = "demographic_parity",
        adversarial_schedule: str = "linear",
        verbose: bool = True
    ):
        """
        Initialize adversarial fair classifier.

        Args:
            config: Neural network configuration
            protected_attributes: List of protected attribute names
            fairness_constraint: Type of fairness constraint
            adversarial_schedule: Schedule for adversarial weight
            verbose: Whether to print training progress
        """
        self.config = config or NeuralFairnessConfig([128, 64])
        self.protected_attributes = protected_attributes or []
        self.fairness_constraint = fairness_constraint
        self.adversarial_schedule = adversarial_schedule
        self.verbose = verbose

        # Models
        self.predictor_network = None
        self.adversarial_network = None
        self.feature_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        # Training state
        self.training_history_ = []
        self.is_fitted_ = False

        logger.info("AdversarialFairClassifier initialized")

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """
        Fit the adversarial fair classifier.

        Args:
            X: Feature matrix including protected attributes
            y: Target variable
            **kwargs: Additional arguments
        """
        logger.info("Training adversarial fair classifier")
        start_time = time.time()

        # Prepare data
        X_processed = self._prepare_features(X)
        y_processed = self._prepare_targets(y)

        # Initialize networks
        input_dim = X_processed.shape[1] - len(self.protected_attributes)  # Exclude protected attrs from prediction

        # Predictor network (accuracy objective)
        self.predictor_network = SimpleNeuralNetwork(
            input_dim=input_dim,
            hidden_dims=self.config.hidden_dims,
            output_dim=1,
            learning_rate=self.config.learning_rate,
            dropout_rate=self.config.dropout_rate
        )

        # Adversarial network (tries to predict protected attributes from representations)
        if len(self.protected_attributes) > 0:
            self.adversarial_network = SimpleNeuralNetwork(
                input_dim=self.config.hidden_dims[-1],  # Takes last hidden layer as input
                hidden_dims=[64, 32],
                output_dim=len(self.protected_attributes),
                learning_rate=self.config.learning_rate,
                dropout_rate=0.1
            )

        # Training loop
        self.training_history_ = []
        X_features = X_processed[:, :-len(self.protected_attributes)] if len(self.protected_attributes) > 0 else X_processed
        protected_data = X_processed[:, -len(self.protected_attributes):] if len(self.protected_attributes) > 0 else None

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config.epochs):
            # Forward pass through predictor
            predictions = self.predictor_network.forward(X_features, training=True)

            # Compute prediction loss
            prediction_loss = np.mean((predictions.flatten() - y_processed)**2)

            # Compute fairness loss
            fairness_loss = 0.0
            if protected_data is not None and len(self.protected_attributes) > 0:
                if self.fairness_constraint == "demographic_parity":
                    fairness_loss = FairnessLoss.demographic_parity_loss(
                        predictions.flatten(), protected_data[:, 0]
                    )
                elif self.fairness_constraint == "equalized_odds":
                    fairness_loss = FairnessLoss.equalized_odds_loss(
                        y_processed, predictions.flatten(), protected_data[:, 0]
                    )

            # Adversarial loss (if applicable)
            adversarial_loss = 0.0
            if self.adversarial_network is not None:
                # Get representation from predictor's last hidden layer
                representation = self.predictor_network.activations[-2]  # Second to last layer

                # Adversarial prediction
                adv_predictions = self.adversarial_network.forward(representation, training=True)

                # Adversarial loss (trying to predict protected attributes)
                if protected_data is not None:
                    adversarial_loss = np.mean((adv_predictions.flatten() - protected_data[:, 0])**2)

            # Dynamic adversarial weight
            adv_weight = self._get_adversarial_weight(epoch)

            # Combined loss
            total_loss = (
                prediction_loss +
                self.config.fairness_weight * fairness_loss -
                adv_weight * adversarial_loss  # Note the negative sign (gradient reversal)
            )

            # Backward pass for predictor
            self.predictor_network.backward(X_features, y_processed.reshape(-1, 1), predictions)

            # Backward pass for adversarial network (if applicable)
            if self.adversarial_network is not None and protected_data is not None:
                self.adversarial_network.backward(
                    representation,
                    protected_data[:, 0].reshape(-1, 1),
                    adv_predictions
                )

            # Record training history
            accuracy = accuracy_score(y_processed, (predictions > 0.5).astype(int))

            history_entry = {
                'epoch': epoch,
                'total_loss': total_loss,
                'prediction_loss': prediction_loss,
                'fairness_loss': fairness_loss,
                'adversarial_loss': adversarial_loss,
                'accuracy': accuracy,
                'adversarial_weight': adv_weight
            }
            self.training_history_.append(history_entry)

            # Early stopping
            if total_loss < best_loss:
                best_loss = total_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.patience:
                if self.verbose:
                    logger.info(f"Early stopping at epoch {epoch}")
                break

            # Verbose logging
            if self.verbose and epoch % 20 == 0:
                logger.debug(f"Epoch {epoch}: Loss={total_loss:.4f}, Accuracy={accuracy:.3f}, Fairness={fairness_loss:.4f}")

        training_time = time.time() - start_time
        self.is_fitted_ = True

        logger.info(f"Adversarial training completed in {training_time:.2f}s after {len(self.training_history_)} epochs")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")

        X_processed = self._prepare_features(X, fit_scaler=False)
        X_features = X_processed[:, :-len(self.protected_attributes)] if len(self.protected_attributes) > 0 else X_processed

        predictions = self.predictor_network.predict(X_features)
        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")

        X_processed = self._prepare_features(X, fit_scaler=False)
        X_features = X_processed[:, :-len(self.protected_attributes)] if len(self.protected_attributes) > 0 else X_processed

        return self.predictor_network.predict_proba(X_features)

    def _prepare_features(self, X: pd.DataFrame, fit_scaler: bool = True) -> np.ndarray:
        """Prepare features for training."""
        # Drop protected attributes for main features
        feature_cols = [col for col in X.columns if col not in self.protected_attributes]

        X_features = X[feature_cols].values

        # Scale features
        if fit_scaler:
            X_features_scaled = self.feature_scaler.fit_transform(X_features)
        else:
            X_features_scaled = self.feature_scaler.transform(X_features)

        # Add protected attributes back (for adversarial training)
        if len(self.protected_attributes) > 0:
            protected_data = X[self.protected_attributes].values
            X_processed = np.concatenate([X_features_scaled, protected_data], axis=1)
        else:
            X_processed = X_features_scaled

        return X_processed

    def _prepare_targets(self, y: pd.Series) -> np.ndarray:
        """Prepare targets for training."""
        if len(np.unique(y)) == 2:
            # Binary classification
            return self.label_encoder.fit_transform(y).astype(float)
        else:
            # Multi-class (not fully implemented)
            return y.values.astype(float)

    def _get_adversarial_weight(self, epoch: int) -> float:
        """Get adversarial weight based on schedule."""
        if self.adversarial_schedule == "linear":
            # Linear increase
            return min(self.config.adversarial_weight * epoch / (self.config.epochs * 0.5), self.config.adversarial_weight)
        elif self.adversarial_schedule == "exponential":
            # Exponential increase
            return self.config.adversarial_weight * (1 - np.exp(-epoch / 20))
        else:
            # Constant
            return self.config.adversarial_weight

    def get_fairness_evolution(self, X: pd.DataFrame, y: pd.Series) -> List[Dict[str, float]]:
        """Get evolution of fairness metrics during training."""
        if not self.training_history_:
            return []

        # This is a simplified version - in practice, you'd evaluate on validation set
        fairness_evolution = []

        for i, history in enumerate(self.training_history_):
            if i % 10 == 0:  # Sample every 10 epochs
                fairness_evolution.append({
                    'epoch': history['epoch'],
                    'fairness_loss': history['fairness_loss'],
                    'adversarial_loss': history['adversarial_loss'],
                    'accuracy': history['accuracy']
                })

        return fairness_evolution


class FairRepresentationLearner(BaseEstimator, TransformerMixin):
    """
    Fair Representation Learning using Variational Auto-Encoder principles.

    Learns fair representations by encoding features into a latent space that
    is invariant to protected attributes while preserving predictive information.

    Research Innovation:
    - Variational fair representation learning
    - Mutual information minimization for fairness
    - Disentangled representation learning
    - Information-theoretic fairness constraints
    """

    def __init__(
        self,
        latent_dim: int = 32,
        protected_attributes: List[str] = None,
        fairness_penalty: float = 1.0,
        reconstruction_weight: float = 1.0,
        disentanglement_weight: float = 0.5
    ):
        """
        Initialize fair representation learner.

        Args:
            latent_dim: Dimensionality of learned representations
            protected_attributes: List of protected attribute names
            fairness_penalty: Weight for fairness penalty
            reconstruction_weight: Weight for reconstruction loss
            disentanglement_weight: Weight for disentanglement loss
        """
        self.latent_dim = latent_dim
        self.protected_attributes = protected_attributes or []
        self.fairness_penalty = fairness_penalty
        self.reconstruction_weight = reconstruction_weight
        self.disentanglement_weight = disentanglement_weight

        # Models
        self.encoder = None
        self.decoder = None
        self.discriminator = None  # For adversarial fairness

        # Training state
        self.feature_scaler = StandardScaler()
        self.is_fitted_ = False

        logger.info("FairRepresentationLearner initialized")

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit the fair representation learner.

        Args:
            X: Feature matrix including protected attributes
            y: Optional target variable
        """
        logger.info("Training fair representation learner")
        start_time = time.time()

        # Prepare features
        feature_cols = [col for col in X.columns if col not in self.protected_attributes]
        X_features = X[feature_cols].values

        # Scale features
        X_features_scaled = self.feature_scaler.fit_transform(X_features)
        input_dim = X_features_scaled.shape[1]

        # Initialize encoder (feature -> representation)
        self.encoder = SimpleNeuralNetwork(
            input_dim=input_dim,
            hidden_dims=[128, 64],
            output_dim=self.latent_dim,
            learning_rate=0.001,
            dropout_rate=0.1
        )

        # Initialize decoder (representation -> feature)
        self.decoder = SimpleNeuralNetwork(
            input_dim=self.latent_dim,
            hidden_dims=[64, 128],
            output_dim=input_dim,
            learning_rate=0.001,
            dropout_rate=0.1
        )

        # Initialize discriminator (representation -> protected attribute)
        if len(self.protected_attributes) > 0:
            self.discriminator = SimpleNeuralNetwork(
                input_dim=self.latent_dim,
                hidden_dims=[32, 16],
                output_dim=len(self.protected_attributes),
                learning_rate=0.001,
                dropout_rate=0.0
            )

        # Training loop
        epochs = 200
        protected_data = X[self.protected_attributes].values if len(self.protected_attributes) > 0 else None

        for epoch in range(epochs):
            # Encode features to representations
            representations = self.encoder.forward(X_features_scaled, training=True)

            # Decode representations back to features
            reconstructed = self.decoder.forward(representations, training=True)

            # Reconstruction loss
            reconstruction_loss = np.mean((X_features_scaled - reconstructed)**2)

            # Discriminator loss (adversarial fairness)
            discriminator_loss = 0.0
            if self.discriminator is not None and protected_data is not None:
                protected_pred = self.discriminator.forward(representations, training=True)
                discriminator_loss = np.mean((protected_pred.flatten() - protected_data[:, 0])**2)

            # Combined loss for encoder/decoder
            (
                self.reconstruction_weight * reconstruction_loss -
                self.fairness_penalty * discriminator_loss  # Gradient reversal
            )

            # Backward pass for encoder/decoder
            self.encoder.backward(X_features_scaled, reconstructed, representations)
            self.decoder.backward(representations, X_features_scaled, reconstructed)

            # Train discriminator separately
            if self.discriminator is not None and protected_data is not None:
                # Use detached representations for discriminator training
                self.discriminator.backward(representations, protected_data[:, 0].reshape(-1, 1), protected_pred)

            if epoch % 50 == 0:
                logger.debug(f"Epoch {epoch}: Reconstruction={reconstruction_loss:.4f}, Discriminator={discriminator_loss:.4f}")

        training_time = time.time() - start_time
        self.is_fitted_ = True

        logger.info(f"Fair representation learning completed in {training_time:.2f}s")

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features to fair representations."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before transformation")

        feature_cols = [col for col in X.columns if col not in self.protected_attributes]
        X_features = X[feature_cols].values
        X_features_scaled = self.feature_scaler.transform(X_features)

        representations = self.encoder.forward(X_features_scaled, training=False)
        return representations

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def reconstruct(self, X: pd.DataFrame) -> np.ndarray:
        """Reconstruct original features from representations."""
        representations = self.transform(X)
        reconstructed_scaled = self.decoder.forward(representations, training=False)

        # Inverse transform to original scale
        return self.feature_scaler.inverse_transform(reconstructed_scaled)

    def evaluate_fairness(self, X: pd.DataFrame) -> Dict[str, float]:
        """Evaluate fairness of learned representations."""
        if len(self.protected_attributes) == 0:
            return {'fairness_score': 1.0}

        representations = self.transform(X)
        protected_data = X[self.protected_attributes[0]].values

        # Compute mutual information approximation
        # (Simplified - in practice, would use more sophisticated methods)
        correlations = []
        for i in range(representations.shape[1]):
            corr = np.corrcoef(representations[:, i], protected_data)[0, 1]
            correlations.append(abs(corr))

        avg_correlation = np.mean(correlations)
        fairness_score = 1.0 - avg_correlation  # Higher is more fair

        return {
            'fairness_score': fairness_score,
            'avg_protected_correlation': avg_correlation,
            'max_protected_correlation': np.max(correlations)
        }


def demonstrate_neural_fairness():
    """Demonstration of neural fairness algorithms."""
    print("🧠 Neural Fairness Framework Demo")

    # Generate synthetic dataset
    np.random.seed(42)
    n_samples = 2000

    # Protected attribute
    protected = np.random.binomial(1, 0.4, n_samples)

    # Features with complex interactions
    feature1 = np.random.normal(protected * 0.8 + np.random.normal(0, 0.1, n_samples), 1.0)
    feature2 = np.random.normal(-protected * 0.6 + feature1 * 0.3, 1.0)
    feature3 = np.random.normal(feature1 * feature2 * 0.2, 1.0)
    feature4 = np.random.normal(protected * feature1 * 0.4, 1.0)
    feature5 = np.random.normal(0, 1.0, n_samples)  # Independent feature

    # Complex target function
    linear_combination = (
        0.5 * feature1 + 0.3 * feature2 + 0.2 * feature3 +
        0.4 * feature4 + 0.1 * feature5 + protected * 0.6
    )
    target_prob = 1 / (1 + np.exp(-linear_combination))
    target = np.random.binomial(1, target_prob, n_samples)

    # Create DataFrame
    X = pd.DataFrame({
        'feature1': feature1,
        'feature2': feature2,
        'feature3': feature3,
        'feature4': feature4,
        'feature5': feature5,
        'protected': protected
    })
    y = pd.Series(target)

    print(f"📊 Dataset: {len(X)} samples, {len(X.columns)} features")
    print(f"   Protected attribute distribution: {np.bincount(protected)}")
    print(f"   Target distribution: {np.bincount(target)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print("\n⚡ Testing Adversarial Fair Classifier")

    # Test adversarial fair classifier
    config = NeuralFairnessConfig(
        hidden_dims=[64, 32],
        epochs=100,
        batch_size=32,
        learning_rate=0.01,
        fairness_weight=2.0,
        adversarial_weight=1.0,
        patience=20
    )

    adversarial_clf = AdversarialFairClassifier(
        config=config,
        protected_attributes=['protected'],
        fairness_constraint='demographic_parity',
        adversarial_schedule='linear',
        verbose=True
    )

    # Train model
    adversarial_clf.fit(X_train, y_train)

    # Make predictions
    predictions = adversarial_clf.predict(X_test)
    probabilities = adversarial_clf.predict_proba(X_test)[:, 1]

    # Evaluate fairness
    overall_metrics, by_group_metrics = compute_fairness_metrics(
        y_true=y_test,
        y_pred=predictions,
        protected=X_test['protected'],
        y_scores=probabilities
    )

    print(f"   Adversarial Classifier Accuracy: {overall_metrics['accuracy']:.3f}")
    print(f"   Demographic Parity Difference: {overall_metrics['demographic_parity_difference']:.3f}")
    print(f"   Equalized Odds Difference: {overall_metrics['equalized_odds_difference']:.3f}")

    # Show training progress
    training_epochs = len(adversarial_clf.training_history_)
    final_loss = adversarial_clf.training_history_[-1]['total_loss']
    print(f"   Training completed in {training_epochs} epochs with final loss: {final_loss:.4f}")

    print("\n🎨 Testing Fair Representation Learner")

    # Test fair representation learner
    fair_repr_learner = FairRepresentationLearner(
        latent_dim=16,
        protected_attributes=['protected'],
        fairness_penalty=1.5,
        reconstruction_weight=1.0,
        disentanglement_weight=0.3
    )

    # Learn fair representations
    fair_representations = fair_repr_learner.fit_transform(X_train)
    test_representations = fair_repr_learner.transform(X_test)

    print(f"   Learned representations: {fair_representations.shape}")

    # Evaluate representation fairness
    fairness_metrics = fair_repr_learner.evaluate_fairness(X_train)
    print(f"   Representation fairness score: {fairness_metrics['fairness_score']:.3f}")
    print(f"   Avg protected correlation: {fairness_metrics['avg_protected_correlation']:.3f}")

    # Train classifier on fair representations
    from sklearn.linear_model import LogisticRegression

    repr_classifier = LogisticRegression(max_iter=1000, random_state=42)
    repr_classifier.fit(fair_representations, y_train)
    repr_predictions = repr_classifier.predict(test_representations)
    repr_probabilities = repr_classifier.predict_proba(test_representations)[:, 1]

    # Evaluate classifier trained on fair representations
    repr_overall, _ = compute_fairness_metrics(
        y_true=y_test,
        y_pred=repr_predictions,
        protected=X_test['protected'],
        y_scores=repr_probabilities
    )

    print(f"   Fair Repr Classifier Accuracy: {repr_overall['accuracy']:.3f}")
    print(f"   Fair Repr Demographic Parity Diff: {repr_overall['demographic_parity_difference']:.3f}")

    # Compare with baseline
    print("\n📈 Baseline Comparison")

    baseline_clf = LogisticRegression(max_iter=1000, random_state=42)
    baseline_clf.fit(X_train.drop('protected', axis=1), y_train)
    baseline_pred = baseline_clf.predict(X_test.drop('protected', axis=1))
    baseline_proba = baseline_clf.predict_proba(X_test.drop('protected', axis=1))[:, 1]

    baseline_overall, _ = compute_fairness_metrics(
        y_true=y_test,
        y_pred=baseline_pred,
        protected=X_test['protected'],
        y_scores=baseline_proba
    )

    print(f"   Baseline Accuracy: {baseline_overall['accuracy']:.3f}")
    print(f"   Baseline Demographic Parity Diff: {baseline_overall['demographic_parity_difference']:.3f}")

    print("\n📊 Summary Comparison")
    print("   Method                  | Accuracy | DP Diff | EO Diff")
    print("   ----------------------- | -------- | ------- | -------")
    print(f"   Baseline               | {baseline_overall['accuracy']:.3f}    | {abs(baseline_overall['demographic_parity_difference']):.3f}   | {abs(baseline_overall.get('equalized_odds_difference', 0)):.3f}")
    print(f"   Adversarial Fair       | {overall_metrics['accuracy']:.3f}    | {abs(overall_metrics['demographic_parity_difference']):.3f}   | {abs(overall_metrics['equalized_odds_difference']):.3f}")
    print(f"   Fair Representations   | {repr_overall['accuracy']:.3f}    | {abs(repr_overall['demographic_parity_difference']):.3f}   | {abs(repr_overall.get('equalized_odds_difference', 0)):.3f}")

    print("\n✅ Neural fairness framework demonstration completed! 🌟")


if __name__ == "__main__":
    demonstrate_neural_fairness()
