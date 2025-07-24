#!/usr/bin/env python3
"""
Simple test runner for end-to-end integration tests.
Tests the new integration test functionality without requiring pytest.
"""

import sys
import traceback
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_test_method(test_class, method_name):
    """Run a specific test method and return result."""
    try:
        instance = test_class()
        if hasattr(instance, 'setup_method'):
            instance.setup_method()
        
        test_method = getattr(instance, method_name)
        test_method()
        
        if hasattr(instance, 'teardown_method'):
            instance.teardown_method()
        
        return True, None
    except Exception as e:
        if hasattr(instance, 'teardown_method'):
            try:
                instance.teardown_method()
            except:
                pass
        return False, str(e)

def main():
    """Run end-to-end integration tests."""
    print("🔧 Running End-to-End Integration Tests")
    print("=" * 60)
    
    try:
        # Import test classes
        from tests.test_end_to_end_workflows import (
            TestCompleteDataPipelineWorkflow,
            TestModelTrainingAndEvaluationWorkflow,
            TestCLIInterfaceEndToEnd,
            TestConfigurationChangesImpact,
            TestPerformanceRegressionTests
        )
        
        # Test cases to run
        test_cases = [
            (TestCompleteDataPipelineWorkflow, 'test_complete_data_pipeline_synthetic_generation'),
            (TestCompleteDataPipelineWorkflow, 'test_complete_data_pipeline_csv_loading_and_processing'),
            (TestModelTrainingAndEvaluationWorkflow, 'test_complete_model_training_workflow'),
            (TestModelTrainingAndEvaluationWorkflow, 'test_complete_model_evaluation_workflow'),
            (TestCLIInterfaceEndToEnd, 'test_cli_data_loading_interface'),
            (TestConfigurationChangesImpact, 'test_model_configuration_changes_impact'),
            (TestPerformanceRegressionTests, 'test_data_loading_performance_regression'),
        ]
        
        passed = 0
        failed = 0
        
        for test_class, method_name in test_cases:
            print(f"\nRunning {test_class.__name__}.{method_name}...")
            
            success, error = run_test_method(test_class, method_name)
            
            if success:
                print(f"✅ PASSED: {method_name}")
                passed += 1
            else:
                print(f"❌ FAILED: {method_name}")
                print(f"   Error: {error}")
                failed += 1
        
        print("\n" + "=" * 60)
        print(f"📊 Test Results: {passed} passed, {failed} failed")
        
        if failed == 0:
            print("🎉 ALL END-TO-END INTEGRATION TESTS PASSED!")
            print()
            print("Task Completion Summary:")
            print("✅ Complete data pipeline workflow tests")
            print("✅ Model training and evaluation workflow tests")  
            print("✅ CLI interface end-to-end tests")
            print("✅ Configuration changes impact tests")
            print("✅ Performance regression tests")
            print()
            print("Task 'end_to_end_tests' is COMPLETE!")
            return 0
        else:
            print("❌ Some integration tests failed")
            return 1
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Some required modules may not be available.")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())