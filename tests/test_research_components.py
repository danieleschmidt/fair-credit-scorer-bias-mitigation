#!/usr/bin/env python3
"""
Comprehensive test suite for research components.
Validates emergent fairness consciousness and temporal fairness preservation.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from research.emergent_fairness_consciousness import (
    EmergentFairnessNeuron,
    EmergentFairnessConsciousness,
    demonstrate_emergent_fairness_consciousness
)
from research.temporal_fairness_preservation import (
    TemporalFairnessMemory,
    TemporalFairnessPreserver,
    demonstrate_temporal_fairness_preservation
)


class TestEmergentFairnessNeuron:
    """Test the EmergentFairnessNeuron component."""
    
    def test_neuron_initialization(self):
        """Test neuron initializes with correct parameters."""
        neuron = EmergentFairnessNeuron(neuron_id="test_1", fairness_weight=0.8)
        assert neuron.neuron_id == "test_1"
        assert neuron.fairness_weight == 0.8
        assert neuron.activation_threshold == 0.5
        assert isinstance(neuron.moral_reasoning_params, dict)
    
    def test_neuron_activation(self):
        """Test neuron activation with various stimuli."""
        neuron = EmergentFairnessNeuron("test", fairness_weight=0.7)
        
        # Test high bias input
        high_bias_input = {"bias_score": 0.9, "demographic_parity": 0.3}
        activation = neuron.process_fairness_stimulus(high_bias_input)
        assert activation > 0.5  # Should activate for high bias
        
        # Test fair input
        fair_input = {"bias_score": 0.1, "demographic_parity": 0.8}
        activation = neuron.process_fairness_stimulus(fair_input)
        assert activation < 0.5  # Should not activate for fair input
    
    def test_neuron_adaptation(self):
        """Test neuron adapts to repeated stimuli."""
        neuron = EmergentFairnessNeuron("adaptive", fairness_weight=0.5)
        initial_threshold = neuron.activation_threshold
        
        # Simulate repeated high bias inputs
        bias_input = {"bias_score": 0.8, "demographic_parity": 0.2}
        for _ in range(10):
            neuron.process_fairness_stimulus(bias_input)
        
        # Threshold should adapt (become more sensitive)
        assert neuron.activation_threshold != initial_threshold


class TestEmergentFairnessConsciousness:
    """Test the EmergentFairnessConsciousness system."""
    
    def test_consciousness_initialization(self):
        """Test consciousness system initializes properly."""
        consciousness = EmergentFairnessConsciousness(num_neurons=5)
        assert len(consciousness.neurons) == 5
        assert consciousness.global_fairness_state is not None
        assert hasattr(consciousness, 'emergence_tracker')
    
    def test_fairness_evaluation(self):
        """Test fairness evaluation across multiple metrics."""
        consciousness = EmergentFairnessConsciousness(num_neurons=3)
        
        # Create sample data
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        protected_attr = np.random.choice([0, 1], size=100)
        
        # Test evaluation
        fairness_state = consciousness.evaluate_fairness(X, y, protected_attr)
        
        assert 'demographic_parity' in fairness_state
        assert 'equalized_odds' in fairness_state
        assert 'fairness_score' in fairness_state
        assert 0 <= fairness_state['fairness_score'] <= 1
    
    def test_consciousness_evolution(self):
        """Test consciousness evolves over time."""
        consciousness = EmergentFairnessConsciousness(num_neurons=3)
        initial_state = consciousness.global_fairness_state.copy()
        
        # Generate sample data
        X, y = make_classification(n_samples=50, n_features=4, random_state=42)
        protected_attr = np.random.choice([0, 1], size=50)
        
        # Process multiple evaluations
        for _ in range(5):
            consciousness.evaluate_fairness(X, y, protected_attr)
        
        # State should have evolved
        final_state = consciousness.global_fairness_state
        assert final_state != initial_state
    
    def test_demonstrate_consciousness(self):
        """Test the demonstration function runs without errors."""
        result = demonstrate_emergent_fairness_consciousness()
        assert result is not None
        assert 'emergent_consciousness' in result
        assert 'fairness_evolution' in result


class TestTemporalFairnessMemory:
    """Test the TemporalFairnessMemory system."""
    
    def test_memory_initialization(self):
        """Test memory system initializes correctly."""
        memory = TemporalFairnessMemory(memory_size=10)
        assert memory.memory_size == 10
        assert len(memory.fairness_snapshots) == 0
        assert isinstance(memory.importance_weights, dict)
    
    def test_memory_storage(self):
        """Test storing fairness snapshots."""
        memory = TemporalFairnessMemory(memory_size=3)
        
        # Store snapshots
        for i in range(5):
            snapshot = {
                'timestamp': f'2024-01-0{i+1}',
                'fairness_metrics': {'accuracy': 0.8 + i*0.01},
                'model_version': f'v{i+1}'
            }
            memory.store_fairness_snapshot(snapshot)
        
        # Should only keep the most recent 3
        assert len(memory.fairness_snapshots) == 3
        assert memory.fairness_snapshots[-1]['model_version'] == 'v5'
    
    def test_memory_retrieval(self):
        """Test retrieving relevant memories."""
        memory = TemporalFairnessMemory(memory_size=5)
        
        # Store test snapshots
        snapshots = [
            {'timestamp': '2024-01-01', 'fairness_metrics': {'demographic_parity': 0.9}},
            {'timestamp': '2024-01-02', 'fairness_metrics': {'demographic_parity': 0.7}},
            {'timestamp': '2024-01-03', 'fairness_metrics': {'demographic_parity': 0.8}}
        ]
        
        for snapshot in snapshots:
            memory.store_fairness_snapshot(snapshot)
        
        # Retrieve similar memories
        current_metrics = {'demographic_parity': 0.75}
        similar = memory.retrieve_similar_memories(current_metrics, top_k=2)
        
        assert len(similar) <= 2
        assert all('fairness_metrics' in mem for mem in similar)


class TestTemporalFairnessPreserver:
    """Test the TemporalFairnessPreserver system."""
    
    def test_preserver_initialization(self):
        """Test preserver initializes with correct components."""
        preserver = TemporalFairnessPreserver()
        assert hasattr(preserver, 'memory')
        assert hasattr(preserver, 'drift_detector')
        assert hasattr(preserver, 'adaptive_thresholds')
    
    @patch('research.temporal_fairness_preservation.TemporalFairnessPreserver._detect_fairness_drift')
    def test_fairness_preservation(self, mock_drift_detection):
        """Test fairness preservation process."""
        preserver = TemporalFairnessPreserver()
        mock_drift_detection.return_value = {'drift_detected': False, 'drift_score': 0.1}
        
        # Create mock model and data
        mock_model = Mock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.6, 0.4]])
        
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        protected_attr = np.array([0, 1])
        
        # Test preservation
        result = preserver.preserve_fairness_over_time(
            model=mock_model,
            X=X,
            y=y,
            protected_attributes=protected_attr
        )
        
        assert 'fairness_preserved' in result
        assert 'preservation_actions' in result
        assert isinstance(result['preservation_actions'], list)
    
    def test_drift_detection(self):
        """Test fairness drift detection."""
        preserver = TemporalFairnessPreserver()
        
        # Simulate historical and current metrics
        historical_metrics = {'demographic_parity': 0.8, 'equalized_odds': 0.75}
        current_metrics = {'demographic_parity': 0.6, 'equalized_odds': 0.7}
        
        drift_result = preserver._detect_fairness_drift(historical_metrics, current_metrics)
        
        assert 'drift_detected' in drift_result
        assert 'drift_score' in drift_result
        assert isinstance(drift_result['drift_detected'], bool)
    
    def test_demonstrate_preservation(self):
        """Test the demonstration function runs without errors."""
        result = demonstrate_temporal_fairness_preservation()
        assert result is not None
        assert 'temporal_preserver' in result
        assert 'preservation_results' in result


class TestIntegrationScenarios:
    """Test integration between research components."""
    
    def test_consciousness_and_preservation_integration(self):
        """Test integration between consciousness and temporal preservation."""
        # Initialize both systems
        consciousness = EmergentFairnessConsciousness(num_neurons=3)
        preserver = TemporalFairnessPreserver()
        
        # Generate test data
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        protected_attr = np.random.choice([0, 1], size=100)
        
        # Get consciousness evaluation
        fairness_state = consciousness.evaluate_fairness(X, y, protected_attr)
        
        # Store in temporal memory
        snapshot = {
            'timestamp': '2024-01-01',
            'fairness_metrics': fairness_state,
            'consciousness_state': consciousness.global_fairness_state.copy()
        }
        preserver.memory.store_fairness_snapshot(snapshot)
        
        # Verify integration
        assert len(preserver.memory.fairness_snapshots) == 1
        stored_snapshot = preserver.memory.fairness_snapshots[0]
        assert 'consciousness_state' in stored_snapshot
    
    def test_end_to_end_research_pipeline(self):
        """Test complete research pipeline execution."""
        # This test validates the entire research component pipeline
        consciousness_result = demonstrate_emergent_fairness_consciousness()
        preservation_result = demonstrate_temporal_fairness_preservation()
        
        # Validate results structure
        assert consciousness_result is not None
        assert preservation_result is not None
        
        # Check key components exist
        assert 'emergent_consciousness' in consciousness_result
        assert 'temporal_preserver' in preservation_result
        
        # Verify systems can work together
        consciousness = consciousness_result['emergent_consciousness']
        preserver = preservation_result['temporal_preserver']
        
        assert hasattr(consciousness, 'neurons')
        assert hasattr(preserver, 'memory')


if __name__ == "__main__":
    # Run comprehensive tests
    pytest.main([__file__, "-v", "--tb=short"])