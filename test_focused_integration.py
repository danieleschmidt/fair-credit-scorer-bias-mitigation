#!/usr/bin/env python3
"""
Focused integration test to verify the system works before running the full suite.
This tests the core integration functionality.
"""

import datetime
import os
import tempfile
import sys
sys.path.insert(0, '.')
sys.path.insert(0, './src')

from backlog_manager import BacklogManager, BacklogItem, TaskStatus, TaskType
from autonomous_backlog_executor import AutonomousBacklogExecutor
from security_quality_gates import SecurityQualityGateManager


def test_basic_integration():
    """Test basic integration functionality"""
    print("🧪 Testing Basic Integration")
    
    # Create temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test directories
        for subdir in ["DOCS", "src", "tests"]:
            os.makedirs(os.path.join(temp_dir, subdir), exist_ok=True)
        
        # Test BacklogManager
        manager = BacklogManager(temp_dir)
        
        # Create test item
        test_item = BacklogItem(
            id="integration_test",
            title="Integration Test Item",
            description="Test integration functionality",
            task_type=TaskType.FEATURE,
            business_value=8,
            effort=3,
            status=TaskStatus.READY,
            acceptance_criteria=["Test passes", "Code works"]
        )
        
        manager.backlog = [test_item]
        
        # Test scoring and ranking
        ranked = manager.score_and_rank()
        assert len(ranked) == 1
        assert ranked[0].wsjf_score > 0
        print(f"✓ WSJF Score: {ranked[0].wsjf_score:.2f}")
        
        # Test getting next ready item
        next_item = manager.get_next_ready_item()
        assert next_item is not None
        assert next_item.id == "integration_test"
        print("✓ Ready item selection works")
        
        # Test status report generation
        report = manager.generate_status_report()
        assert 'backlog_size' in report
        assert report['backlog_size'] == 1
        print("✓ Status report generation works")
        
        # Test save/load cycle
        manager.save_backlog()
        
        new_manager = BacklogManager(temp_dir)
        loaded = new_manager.load_backlog()
        assert len(loaded) == 1
        assert loaded[0].id == "integration_test"
        print("✓ Save/load cycle works")
        
        print("🎉 Basic integration test PASSED")


def test_executor_integration():
    """Test executor integration"""
    print("\n🧪 Testing Executor Integration")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test directories
        for subdir in ["DOCS", "src", "tests"]:
            os.makedirs(os.path.join(temp_dir, subdir), exist_ok=True)
        
        # Create executor in dry-run mode
        executor = AutonomousBacklogExecutor(temp_dir, dry_run=True)
        
        # Create simple backlog
        test_item = BacklogItem(
            id="executor_test",
            title="Executor Test Item",
            description="Test executor functionality",
            task_type=TaskType.FEATURE,
            business_value=5,
            effort=2,
            status=TaskStatus.READY,
            acceptance_criteria=["Execute successfully"]
        )
        
        executor.manager.backlog = [test_item]
        # Save it so get_backlog_summary can load it
        executor.manager.save_backlog()
        
        # Test backlog summary
        summary = executor.get_backlog_summary()
        print(f"Debug - Summary: {summary}")
        assert summary['total_items'] >= 1
        assert summary['ready_items'] >= 1
        print("✓ Backlog summary works")
        
        # Test single cycle execution (dry run)
        final_report = executor.execute_full_backlog_cycle(max_cycles=1, cycle_delay=0)
        print(f"Debug - Final report: {final_report}")
        assert final_report['cycles_completed'] >= 1  # Allow for some variation in execution
        print("✓ Execution cycle works")
        
        print("🎉 Executor integration test PASSED")


def test_gates_integration():
    """Test security and quality gates integration"""
    print("\n🧪 Testing Gates Integration")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test directories
        for subdir in ["src", "config"]:
            os.makedirs(os.path.join(temp_dir, subdir), exist_ok=True)
        
        # Create a test Python file
        test_py_content = '''
def documented_function():
    """This function is properly documented"""
    return "test"

def undocumented_function():
    return "no docs"
'''
        
        with open(os.path.join(temp_dir, "src", "test_file.py"), 'w') as f:
            f.write(test_py_content)
        
        # Test gate manager
        gate_manager = SecurityQualityGateManager(temp_dir)
        
        # Test documentation gate
        doc_result = gate_manager._run_documentation_gate()
        assert doc_result.gate_type.value == "documentation"
        assert len(doc_result.metrics) == 1
        print("✓ Documentation gate works")
        
        # Test overall quality evaluation
        gate_results = [doc_result]
        overall_passed, overall_score, recommendations = gate_manager.evaluate_overall_quality(gate_results)
        assert isinstance(overall_passed, bool)
        assert isinstance(overall_score, float)
        print(f"✓ Overall quality evaluation works (Score: {overall_score:.1f})")
        
        print("🎉 Gates integration test PASSED")


if __name__ == "__main__":
    print("🚀 Running Focused Integration Tests")
    print("=" * 50)
    
    try:
        test_basic_integration()
        test_executor_integration()
        test_gates_integration()
        
        print("\n" + "=" * 50)
        print("🎉 ALL FOCUSED INTEGRATION TESTS PASSED!")
        print("✅ System is ready for comprehensive testing")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)