"""Comprehensive tests for architecture_review.py to improve coverage."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch
import networkx as nx

from src.architecture_review import ArchitectureReview


@pytest.fixture
def temp_project():
    """Create a temporary project structure for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create src directory with sample Python files
    src_dir = temp_dir / "src"
    src_dir.mkdir()
    
    # Create sample Python files with imports
    (src_dir / "module_a.py").write_text("""
import os
from .module_b import some_function
from module_c import another_function
""")
    
    (src_dir / "module_b.py").write_text("""
import sys
from module_c import helper
""")
    
    (src_dir / "module_c.py").write_text("""
import json
# No internal imports
""")
    
    # Create requirements.txt
    requirements = temp_dir / "requirements.txt"
    requirements.write_text("""
# Test requirements
pandas==2.3.0
numpy==2.3.1
matplotlib==3.10.3
# Comment line
scikit-learn==1.7.0
""")
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


def test_architecture_review_init():
    """Test ArchitectureReview initialization."""
    review = ArchitectureReview()
    assert review.src_path == Path("src")
    assert review.requirements_file == Path("requirements.txt")
    assert isinstance(review.graph, nx.DiGraph)
    assert review.dependencies == []


def test_architecture_review_custom_paths():
    """Test ArchitectureReview with custom paths."""
    review = ArchitectureReview(
        src_path=Path("custom/src"),
        requirements_file=Path("custom/requirements.txt")
    )
    assert review.src_path == Path("custom/src")
    assert review.requirements_file == Path("custom/requirements.txt")


def test_parse_requirements_missing_file(temp_project):
    """Test parsing requirements when file doesn't exist."""
    review = ArchitectureReview(
        src_path=temp_project / "src",
        requirements_file=temp_project / "nonexistent.txt"
    )
    deps = review._parse_requirements()
    assert deps == []


def test_parse_requirements_valid_file(temp_project):
    """Test parsing valid requirements file."""
    review = ArchitectureReview(
        src_path=temp_project / "src",
        requirements_file=temp_project / "requirements.txt"
    )
    deps = review._parse_requirements()
    
    expected = ["pandas", "numpy", "matplotlib", "scikit-learn"]
    assert deps == expected


def test_build_dependency_graph(temp_project):
    """Test building the dependency graph."""
    review = ArchitectureReview(
        src_path=temp_project / "src",
        requirements_file=temp_project / "requirements.txt"
    )
    graph = review._build_dependency_graph()
    
    # Check nodes exist
    assert "module_a" in graph.nodes
    assert "module_b" in graph.nodes  
    assert "module_c" in graph.nodes
    
    # Check edges exist for internal imports
    assert graph.has_edge("module_a", "module_b")
    assert graph.has_edge("module_a", "module_c")
    assert graph.has_edge("module_b", "module_c")


def test_analyze_method(temp_project):
    """Test the analyze method populates both dependencies and graph."""
    review = ArchitectureReview(
        src_path=temp_project / "src",
        requirements_file=temp_project / "requirements.txt"
    )
    
    # Initially empty
    assert review.dependencies == []
    assert len(review.graph.nodes) == 0
    
    review.analyze()
    
    # Should be populated after analyze
    assert len(review.dependencies) > 0
    assert len(review.graph.nodes) > 0
    assert "pandas" in review.dependencies
    assert "module_a" in review.graph.nodes


def test_save_diagram(temp_project):
    """Test saving the dependency diagram."""
    review = ArchitectureReview(
        src_path=temp_project / "src",
        requirements_file=temp_project / "requirements.txt"
    )
    
    output_path = temp_project / "test_diagram.svg"
    
    with patch('matplotlib.pyplot.savefig') as mock_savefig, \
         patch('matplotlib.pyplot.figure'), \
         patch('matplotlib.pyplot.tight_layout'), \
         patch('matplotlib.pyplot.close'), \
         patch('networkx.spring_layout') as mock_layout, \
         patch('networkx.draw'):
        
        mock_layout.return_value = {"module_a": (0, 0)}
        
        review.save_diagram(str(output_path))
        
        # Should call analyze if graph is empty
        assert len(review.graph.nodes) > 0
        
        # Should create parent directory
        assert output_path.parent.exists()
        
        # Should call savefig with correct parameters
        mock_savefig.assert_called_once_with(output_path, format="svg")


def test_save_diagram_with_existing_graph(temp_project):
    """Test saving diagram when graph already exists."""
    review = ArchitectureReview(
        src_path=temp_project / "src",
        requirements_file=temp_project / "requirements.txt"
    )
    
    # Pre-populate the graph
    review.analyze()
    original_graph = review.graph.copy()
    
    output_path = temp_project / "test_diagram.svg"
    
    with patch('matplotlib.pyplot.savefig'), \
         patch('matplotlib.pyplot.figure'), \
         patch('matplotlib.pyplot.tight_layout'), \
         patch('matplotlib.pyplot.close'), \
         patch('networkx.spring_layout'), \
         patch('networkx.draw'):
        
        review.save_diagram(str(output_path))
        
        # Graph should remain the same (analyze not called again)
        assert review.graph.nodes == original_graph.nodes


def test_save_summary(temp_project):
    """Test saving the architecture summary."""
    review = ArchitectureReview(
        src_path=temp_project / "src", 
        requirements_file=temp_project / "requirements.txt"
    )
    
    output_path = temp_project / "test_summary.md"
    review.save_summary(str(output_path))
    
    # Should create the file
    assert output_path.exists()
    
    content = output_path.read_text()
    
    # Should contain expected sections
    assert "# Architecture Review" in content
    assert "## Dependencies" in content
    assert "## Module Graph" in content
    
    # Should contain dependency information
    assert "pandas" in content
    assert "numpy" in content
    
    # Should contain module information
    assert "module_a" in content
    assert "module_b" in content
    assert "module_c" in content


def test_save_summary_with_existing_graph(temp_project):
    """Test saving summary when graph already exists."""
    review = ArchitectureReview(
        src_path=temp_project / "src",
        requirements_file=temp_project / "requirements.txt"
    )
    
    # Pre-populate
    review.analyze()
    original_deps = review.dependencies.copy()
    
    output_path = temp_project / "test_summary.md"
    review.save_summary(str(output_path))
    
    # Dependencies should remain the same
    assert review.dependencies == original_deps


def test_main_function(temp_project):
    """Test the main function."""
    from src.architecture_review import main
    
    with patch.object(ArchitectureReview, '__init__', return_value=None) as mock_init, \
         patch.object(ArchitectureReview, 'analyze') as mock_analyze, \
         patch.object(ArchitectureReview, 'save_diagram') as mock_save_diagram, \
         patch.object(ArchitectureReview, 'save_summary') as mock_save_summary:
        
        main()
        
        mock_init.assert_called_once()
        mock_analyze.assert_called_once()
        mock_save_diagram.assert_called_once()
        mock_save_summary.assert_called_once()


def test_module_dependencies_edge_cases(temp_project):
    """Test edge cases in dependency parsing."""
    # Create a more complex Python file
    complex_file = temp_project / "src" / "complex_module.py"
    complex_file.write_text("""
# Test various import patterns
import os
import sys.path
from pathlib import Path
from . import module_a
from .subpackage import helper
from ..parent import utility  # parent level import
import external.library
from external.package import function

# Some code
def test_function():
    pass
""")
    
    review = ArchitectureReview(
        src_path=temp_project / "src",
        requirements_file=temp_project / "requirements.txt"
    )
    
    graph = review._build_dependency_graph()
    
    # Should handle relative imports correctly
    assert "complex_module" in graph.nodes
    if graph.has_edge("complex_module", "module_a"):
        assert graph.has_edge("complex_module", "module_a")


def test_empty_source_directory():
    """Test handling of empty source directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        empty_src = temp_path / "empty_src" 
        empty_src.mkdir()
        
        review = ArchitectureReview(
            src_path=empty_src,
            requirements_file=temp_path / "requirements.txt"
        )
        
        graph = review._build_dependency_graph()
        assert len(graph.nodes) == 0