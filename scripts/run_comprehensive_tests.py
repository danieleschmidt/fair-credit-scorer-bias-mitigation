#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class TestRunner:
    """Comprehensive test runner for the Fair Credit Scorer project."""
    
    def __init__(self, verbose: bool = False, dry_run: bool = False):
        self.verbose = verbose
        self.dry_run = dry_run
        self.project_root = Path(__file__).parent.parent
        self.results = {}
        self.start_time = datetime.now()
    
    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
        
        if self.verbose and level == "DEBUG":
            print(f"[{timestamp}] DEBUG: {message}")
    
    def run_command(self, command: List[str], description: str, 
                   capture_output: bool = True) -> Dict:
        """Run a command and capture results."""
        self.log(f"Running: {description}")
        self.log(f"Command: {' '.join(command)}", "DEBUG")
        
        if self.dry_run:
            self.log(f"DRY RUN: Would execute: {' '.join(command)}")
            return {"success": True, "stdout": "", "stderr": "", "duration": 0}
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=capture_output,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            
            duration = time.time() - start_time
            success = result.returncode == 0
            
            if not success:
                self.log(f"Command failed with return code {result.returncode}", "ERROR")
                if result.stderr:
                    self.log(f"Error output: {result.stderr}", "ERROR")
            
            return {
                "success": success,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": duration,
                "return_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            self.log(f"Command timed out after 30 minutes", "ERROR")
            return {
                "success": False,
                "stdout": "",
                "stderr": "Command timed out",
                "duration": 1800,
                "return_code": -1
            }
        except Exception as e:
            self.log(f"Command execution failed: {str(e)}", "ERROR")
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "duration": time.time() - start_time,
                "return_code": -1
            }
    
    def run_unit_tests(self) -> bool:
        """Run unit tests with coverage."""
        self.log("=== Running Unit Tests ===")
        
        command = [
            "python", "-m", "pytest",
            "-m", "unit",
            "--cov=src",
            "--cov-report=html:htmlcov/unit",
            "--cov-report=xml:coverage-unit.xml",
            "--cov-report=term-missing",
            "--junit-xml=junit-unit.xml",
            "-v"
        ]
        
        result = self.run_command(command, "Unit tests with coverage")
        self.results["unit_tests"] = result
        
        if result["success"]:
            self.log("✅ Unit tests passed")
        else:
            self.log("❌ Unit tests failed")
        
        return result["success"]
    
    def run_integration_tests(self) -> bool:
        """Run integration tests."""
        self.log("=== Running Integration Tests ===")
        
        command = [
            "python", "-m", "pytest",
            "-m", "integration",
            "--junit-xml=junit-integration.xml",
            "-v"
        ]
        
        result = self.run_command(command, "Integration tests")
        self.results["integration_tests"] = result
        
        if result["success"]:
            self.log("✅ Integration tests passed")
        else:
            self.log("❌ Integration tests failed")
        
        return result["success"]
    
    def run_e2e_tests(self) -> bool:
        """Run end-to-end tests."""
        self.log("=== Running End-to-End Tests ===")
        
        command = [
            "python", "-m", "pytest",
            "-m", "e2e",
            "--junit-xml=junit-e2e.xml",
            "-v"
        ]
        
        result = self.run_command(command, "End-to-end tests")
        self.results["e2e_tests"] = result
        
        if result["success"]:
            self.log("✅ End-to-end tests passed")
        else:
            self.log("❌ End-to-end tests failed")
        
        return result["success"]
    
    def run_performance_tests(self) -> bool:
        """Run performance tests and benchmarks."""
        self.log("=== Running Performance Tests ===")
        
        command = [
            "python", "-m", "pytest",
            "-m", "performance",
            "--benchmark-only",
            "--benchmark-json=benchmark-results.json",
            "-v"
        ]
        
        result = self.run_command(command, "Performance tests")
        self.results["performance_tests"] = result
        
        if result["success"]:
            self.log("✅ Performance tests passed")
        else:
            self.log("❌ Performance tests failed")
        
        return result["success"]
    
    def run_security_tests(self) -> bool:
        """Run security scans and tests."""
        self.log("=== Running Security Tests ===")
        
        # Bandit security scan
        bandit_command = [
            "bandit", "-r", "src/",
            "-f", "json", "-o", "bandit-report.json"
        ]
        
        bandit_result = self.run_command(bandit_command, "Bandit security scan")
        
        # Safety dependency check
        safety_command = [
            "safety", "check",
            "--json", "--output", "safety-report.json"
        ]
        
        safety_result = self.run_command(safety_command, "Safety dependency check")
        
        # Security-focused tests
        security_tests_command = [
            "python", "-m", "pytest",
            "-m", "security",
            "--junit-xml=junit-security.xml",
            "-v"
        ]
        
        security_tests_result = self.run_command(security_tests_command, "Security tests")
        
        overall_success = (bandit_result["success"] and 
                          safety_result["success"] and 
                          security_tests_result["success"])
        
        self.results["security_tests"] = {
            "bandit": bandit_result,
            "safety": safety_result,
            "security_tests": security_tests_result,
            "success": overall_success
        }
        
        if overall_success:
            self.log("✅ Security tests passed")
        else:
            self.log("❌ Security tests failed")
        
        return overall_success
    
    def run_code_quality_checks(self) -> bool:
        """Run code quality checks."""
        self.log("=== Running Code Quality Checks ===")
        
        # Ruff linting
        ruff_command = ["ruff", "check", "src/", "tests/", "--output-format=json"]
        ruff_result = self.run_command(ruff_command, "Ruff linting")
        
        # MyPy type checking
        mypy_command = ["mypy", "src/", "--json-report", "mypy-report"]
        mypy_result = self.run_command(mypy_command, "MyPy type checking")
        
        # Black formatting check
        black_command = ["black", "--check", "src/", "tests/"]
        black_result = self.run_command(black_command, "Black formatting check")
        
        overall_success = (ruff_result["success"] and 
                          mypy_result["success"] and 
                          black_result["success"])
        
        self.results["code_quality"] = {
            "ruff": ruff_result,
            "mypy": mypy_result,
            "black": black_result,
            "success": overall_success
        }
        
        if overall_success:
            self.log("✅ Code quality checks passed")
        else:
            self.log("❌ Code quality checks failed")
        
        return overall_success
    
    def run_contract_tests(self) -> bool:
        """Run contract tests."""
        self.log("=== Running Contract Tests ===")
        
        command = [
            "python", "-m", "pytest",
            "-m", "contract",
            "--junit-xml=junit-contract.xml",
            "-v"
        ]
        
        result = self.run_command(command, "Contract tests")
        self.results["contract_tests"] = result
        
        if result["success"]:
            self.log("✅ Contract tests passed")
        else:
            self.log("❌ Contract tests failed")
        
        return result["success"]
    
    def run_mutation_tests(self) -> bool:
        """Run mutation tests (if enabled)."""
        self.log("=== Running Mutation Tests ===")
        
        command = ["mutmut", "run", "--paths-to-mutate=src/"]
        
        result = self.run_command(command, "Mutation tests")
        self.results["mutation_tests"] = result
        
        if result["success"]:
            self.log("✅ Mutation tests passed")
        else:
            self.log("❌ Mutation tests failed")
        
        return result["success"]
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive test report."""
        self.log("=== Generating Comprehensive Report ===")
        
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        report = {
            "test_run_info": {
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_duration_seconds": total_duration,
                "project_root": str(self.project_root),
                "python_version": sys.version,
                "dry_run": self.dry_run
            },
            "results": self.results,
            "summary": self.generate_summary()
        }
        
        # Save JSON report
        report_file = self.project_root / "test-report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.log(f"Comprehensive report saved to: {report_file}")
        
        # Print summary
        self.print_summary()
        
        return report
    
    def generate_summary(self) -> Dict:
        """Generate test summary statistics."""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() 
                          if isinstance(r, dict) and r.get("success", False))
        failed_tests = total_tests - passed_tests
        
        total_duration = sum(
            r.get("duration", 0) for r in self.results.values() 
            if isinstance(r, dict) and "duration" in r
        )
        
        return {
            "total_test_suites": total_tests,
            "passed_test_suites": passed_tests,
            "failed_test_suites": failed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "total_duration_seconds": total_duration
        }
    
    def print_summary(self):
        """Print test execution summary."""
        summary = self.generate_summary()
        
        print("\n" + "="*80)
        print("COMPREHENSIVE TEST EXECUTION SUMMARY")
        print("="*80)
        
        print(f"Total Test Suites: {summary['total_test_suites']}")
        print(f"Passed: {summary['passed_test_suites']}")
        print(f"Failed: {summary['failed_test_suites']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Total Duration: {summary['total_duration_seconds']:.2f} seconds")
        
        print("\nDetailed Results:")
        for test_name, result in self.results.items():
            if isinstance(result, dict):
                status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
                duration = result.get("duration", 0)
                print(f"  {test_name}: {status} ({duration:.2f}s)")
        
        overall_success = summary['failed_test_suites'] == 0
        print(f"\nOverall Result: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
        print("="*80)
        
        return overall_success


def main():
    """Main entry point for the comprehensive test runner."""
    parser = argparse.ArgumentParser(
        description="Comprehensive test runner for Fair Credit Scorer project"
    )
    
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Show what would be executed without running tests"
    )
    
    parser.add_argument(
        "--skip", 
        nargs="+",
        choices=["unit", "integration", "e2e", "performance", "security", 
                "quality", "contract", "mutation"],
        default=[],
        help="Skip specific test categories"
    )
    
    parser.add_argument(
        "--only", 
        nargs="+",
        choices=["unit", "integration", "e2e", "performance", "security", 
                "quality", "contract", "mutation"],
        help="Run only specific test categories"
    )
    
    parser.add_argument(
        "--fail-fast", 
        action="store_true",
        help="Stop on first test failure"
    )
    
    args = parser.parse_args()
    
    runner = TestRunner(verbose=args.verbose, dry_run=args.dry_run)
    
    # Determine which tests to run
    all_tests = ["unit", "integration", "e2e", "performance", "security", 
                "quality", "contract", "mutation"]
    
    if args.only:
        tests_to_run = args.only
    else:
        tests_to_run = [t for t in all_tests if t not in args.skip]
    
    runner.log(f"Starting comprehensive test execution")
    runner.log(f"Tests to run: {', '.join(tests_to_run)}")
    
    # Test execution mapping
    test_functions = {
        "unit": runner.run_unit_tests,
        "integration": runner.run_integration_tests,
        "e2e": runner.run_e2e_tests,
        "performance": runner.run_performance_tests,
        "security": runner.run_security_tests,
        "quality": runner.run_code_quality_checks,
        "contract": runner.run_contract_tests,
        "mutation": runner.run_mutation_tests
    }
    
    # Run selected tests
    overall_success = True
    
    for test_name in tests_to_run:
        if test_name in test_functions:
            try:
                success = test_functions[test_name]()
                if not success:
                    overall_success = False
                    if args.fail_fast:
                        runner.log("Stopping due to --fail-fast flag", "ERROR")
                        break
            except KeyboardInterrupt:
                runner.log("Test execution interrupted by user", "ERROR")
                overall_success = False
                break
            except Exception as e:
                runner.log(f"Test execution failed: {str(e)}", "ERROR")
                overall_success = False
                if args.fail_fast:
                    break
    
    # Generate comprehensive report
    runner.generate_comprehensive_report()
    
    # Exit with appropriate code
    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    main()