"""Tests for module docstring quality and presence."""

import ast
import importlib
import inspect
from pathlib import Path

import pytest


class TestModuleDocstrings:
    """Test that all modules have comprehensive docstrings."""
    
    def test_all_modules_have_docstrings(self):
        """Test that all Python modules in src/ have module-level docstrings."""
        src_path = Path(__file__).parent.parent / "src"
        
        # Modules that should have docstrings
        expected_modules = [
            "src.baseline_model",
            "src.bias_mitigator", 
            "src.data_loader_preprocessor",
            "src.evaluate_fairness",
            "src.fairness_metrics",
            "src.architecture_review",
            "src.run_tests"
        ]
        
        for module_name in expected_modules:
            try:
                module = importlib.import_module(module_name)
                docstring = inspect.getdoc(module)
                
                assert docstring is not None, f"Module {module_name} is missing a docstring"
                assert len(docstring.strip()) > 0, f"Module {module_name} has an empty docstring"
                
            except ImportError:
                pytest.fail(f"Could not import module {module_name}")
    
    def test_core_modules_have_comprehensive_docstrings(self):
        """Test that core modules have comprehensive docstrings with key elements."""
        core_modules = [
            "src.baseline_model",
            "src.bias_mitigator",
            "src.data_loader_preprocessor", 
            "src.evaluate_fairness",
            "src.fairness_metrics"
        ]
        
        required_elements = [
            "example",  # Should contain usage examples
            "function",  # Should mention main functions
        ]
        
        for module_name in core_modules:
            module = importlib.import_module(module_name)
            docstring = inspect.getdoc(module)
            
            assert docstring is not None, f"Core module {module_name} is missing a docstring"
            assert len(docstring) > 100, f"Module {module_name} docstring is too short (< 100 chars)"
            
            # Check for comprehensive content
            docstring_lower = docstring.lower()
            assert any(element in docstring_lower for element in required_elements), \
                f"Module {module_name} docstring should include examples or function descriptions"
    
    def test_fairness_metrics_module_docstring_content(self):
        """Test specific content requirements for fairness_metrics module."""
        from src import fairness_metrics
        
        docstring = inspect.getdoc(fairness_metrics)
        assert docstring is not None, "fairness_metrics module missing docstring"
        
        # Check for specific fairness concepts
        expected_content = [
            "fairness",
            "demographic parity",
            "equalized odds", 
            "compute_fairness_metrics"
        ]
        
        docstring_lower = docstring.lower()
        for content in expected_content:
            assert content in docstring_lower, \
                f"fairness_metrics docstring should mention '{content}'"
    
    def test_bias_mitigator_module_docstring_content(self):
        """Test specific content requirements for bias_mitigator module."""
        from src import bias_mitigator
        
        docstring = inspect.getdoc(bias_mitigator)
        assert docstring is not None, "bias_mitigator module missing docstring"
        
        # Check for bias mitigation concepts
        expected_content = [
            "bias mitigation",
            "expgrad",
            "reweight",
            "postprocess"
        ]
        
        docstring_lower = docstring.lower()
        for content in expected_content:
            assert content in docstring_lower, \
                f"bias_mitigator docstring should mention '{content}'"
    
    def test_evaluate_fairness_module_docstring_content(self):
        """Test specific content requirements for evaluate_fairness module."""
        from src import evaluate_fairness
        
        docstring = inspect.getdoc(evaluate_fairness)
        assert docstring is not None, "evaluate_fairness module missing docstring"
        
        # Check for evaluation pipeline concepts
        expected_content = [
            "evaluation",
            "pipeline",
            "run_pipeline",
            "cross-validation"
        ]
        
        docstring_lower = docstring.lower()
        for content in expected_content:
            assert content in docstring_lower, \
                f"evaluate_fairness docstring should mention '{content}'"
    
    def test_baseline_model_module_docstring_content(self):
        """Test specific content requirements for baseline_model module."""
        from src import baseline_model
        
        docstring = inspect.getdoc(baseline_model)
        assert docstring is not None, "baseline_model module missing docstring"
        
        # Check for model-specific concepts
        expected_content = [
            "logistic regression",
            "train_baseline_model",
            "evaluate_model"
        ]
        
        docstring_lower = docstring.lower()
        for content in expected_content:
            assert content in docstring_lower, \
                f"baseline_model docstring should mention '{content}'"
    
    def test_data_loader_module_docstring_content(self):
        """Test specific content requirements for data_loader_preprocessor module."""
        from src import data_loader_preprocessor
        
        docstring = inspect.getdoc(data_loader_preprocessor)
        assert docstring is not None, "data_loader_preprocessor module missing docstring"
        
        # Check for data loading concepts
        expected_content = [
            "data loading",
            "preprocessing", 
            "load_credit_dataset",
            "synthetic"
        ]
        
        docstring_lower = docstring.lower()
        for content in expected_content:
            assert content in docstring_lower, \
                f"data_loader_preprocessor docstring should mention '{content}'"
    
    def test_docstring_format_quality(self):
        """Test that docstrings follow good formatting practices."""
        modules_to_check = [
            "src.baseline_model",
            "src.bias_mitigator",
            "src.data_loader_preprocessor",
            "src.evaluate_fairness", 
            "src.fairness_metrics"
        ]
        
        for module_name in modules_to_check:
            module = importlib.import_module(module_name)
            docstring = inspect.getdoc(module)
            
            if docstring:
                # Check basic formatting
                assert not docstring.startswith(" "), \
                    f"Module {module_name} docstring should not start with whitespace"
                assert not docstring.endswith(" "), \
                    f"Module {module_name} docstring should not end with whitespace"
                
                # Should have multiple lines for comprehensive docstrings
                lines = docstring.split('\n')
                assert len(lines) >= 3, \
                    f"Module {module_name} docstring should have multiple lines for comprehensiveness"